# ADR-004: K Selection Method for Firm GMMs

**Status**: Accepted with sensitivity requirement — Week 3 conclusions must report K_max robustness  
**Date**: 2026-04-08 (revised after Bayesian audit + EDA diagnostics)  
**Authors**: Torrin Pataki, Claude Code  
**Reviewers**: Codex (accepted with conditions), STAT 405 methodology audit (incorporated)

## Context

Step 2 of the pipeline fits a Gaussian Mixture Model (GMM) to each firm's set of 50D patent vectors. Each GMM has K components — the number of distinct "technology areas" in the firm's portfolio. K must be determined per firm, since firms have vastly different technological breadths (a focused biotech startup vs. IBM).

The presentation (slide 7) identifies two approaches: a BIC sweep ("brute force loop") and a Bayesian GMM with automatic pruning. Arthur's methodology notes that the team "still need[s] method for determining optimal clusters for GMM given patent vectors." This ADR resolves that open question.

### Constraints

- **15,814 firms** to fit, with extreme size skew (50.8% have <10 patents, top firms have >100K)
- **50-dimensional** input vectors (from UMAP reduction)
- K must be meaningful for downstream comparison: Week 3 computes Bhattacharyya Coefficients across firms' GMM components, aggregated by mixing weights
- Computational budget: the full fitting run should complete in under 2 hours on a single machine (CPU)
- K selection must be reproducible (deterministic given random_state)

## Decision

### Method: Bayesian GMM with Dirichlet Process Prior

Use scikit-learn's `BayesianGaussianMixture` with `weight_concentration_prior_type="dirichlet_process"`. Set K_max (upper bound on components) and let the model automatically prune unused components by driving their mixing weights toward zero.

### Inference Method: Variational Bayes (sklearn)

sklearn uses **mean-field variational inference** (Blei & Jordan 2006), not MCMC. This is adequate for our downstream task because:

1. We need point estimates (μ_k, σ²_k, π_k) for BC computation, not posterior uncertainty
2. VI's variance underestimation affects credible intervals, which we don't need
3. VI sidesteps label switching entirely by converging to one mode — and BC is permutation-invariant over components
4. Computational budget: VI on 15K firms ≈ 1-3 hours; MCMC would be 10-30x slower

Stan/NUTS is explicitly **not** recommended for mixture models — Stan cannot sample discrete latent variables (cluster assignments z_i), and HMC gets trapped in one of K! equivalent posterior modes. If MCMC validation is needed, conjugate Gibbs sampling (Normal-Gamma for diagonal case) is the correct approach (see Bayesian Workflow section).

### Prior Specification: Global Empirical Bayes

sklearn's defaults use per-firm empirical Bayes (`mean_prior=mean(X_firm)`, `covariance_prior=empirical_cov(X_firm)`), which double-counts data. We use **global empirical Bayes**: compute hyperparameters from the pooled dataset of all ~1.45M patents, then apply as fixed priors to each firm's independent fit. This is a tractable approximation to a full hierarchical model — small firms' estimates are shrunk toward the global structure, large firms are nearly unaffected.

```python
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

# Step 1: Compute global hyperparameters from UNIQUE patent vectors (run ONCE)
# Use the patent_vectors_50d matrix directly — NOT grouped firm vectors,
# which would duplicate co-assigned patents and bias the prior.
global_mean = np.mean(patent_vectors_50d, axis=0)   # shape (50,)
global_var = np.var(patent_vectors_50d, axis=0)      # shape (50,)

# Step 2: Fit each firm with globally-calibrated priors
def fit_firm(X_firm, K_max=10):
    bgm = BayesianGaussianMixture(
        n_components=K_max,              # Provisional: 15 (see Sensitivity Condition)
        covariance_type="diag",                           # Per ADR-006
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1.0,                   # γ — see E[K] analysis below
        mean_prior=global_mean,                           # Global, NOT per-firm
        mean_precision_prior=1.0,                         # κ₀: weakly informative
        degrees_of_freedom_prior=52,                      # ν₀ = d+2 for finite posterior mean
        covariance_prior=global_var,                      # Global per-dimension variance
        max_iter=200,
        n_init=5,                                         # Multiple restarts for VI stability
        random_state=42,
        reg_covar=1e-6,
    )
    bgm.fit(X_firm)
    return bgm
```

**Effective K** is determined post-fit by counting components with mixing weight > 0.01 (1%). Components below this threshold are treated as pruned — their parameters are discarded from the serialized output.

**K_max = 15 (provisional).** EDA diagnostics on production data show that K=10 is binding: effective K does not plateau at K_max=10 but continues climbing to K=12-16 at K_max=20, with substantive (non-fragmenting) weights. BC ranking stability analysis shows Spearman ρ ≈ 0.77-0.81 across K_max settings — directionally stable but sensitive enough that K_max affects which specific firm pairs are surfaced as "most similar." K_max=15 is adopted as the operational default because K=10 is clearly too restrictive and K=20 is not obviously converged. **K_max is a provisional research parameter, not a settled constant.** See Sensitivity Condition below.

### Prior Hyperparameter Justification

**weight_concentration_prior γ = 1.0.** The Dirichlet Process prior expected number of components is E[K] ≈ γ · log(n). The E[K] table reveals that γ=0.1 (our initial choice) actively suppresses multi-component solutions:

| γ | n=10 | n=50 | n=100 | n=500 | n=1000 | n=10000 |
|---|------|------|-------|-------|--------|---------|
| **0.1** | **0.4** | **0.6** | **0.7** | **1.0** | **1.0** | **1.3** |
| 0.5 | 1.5 | 2.3 | 2.7 | 3.7 | 4.2 | 6.0 |
| **1.0** | **2.5** | **3.9** | **4.6** | **6.5** | **7.5** | **10.9** |
| 2.0 | 4.0 | 6.4 | 7.6 | 11.0 | 13.0 | 19.8 |

With γ=0.1, a firm with n=10,000 patents has E[K]≈1.3 — the prior assigns 91% of mass to the first component via stick-breaking (E[w₁] = 1/(1+γ) = 0.91). The posterior must overcome this strong sparsity preference before adding components. This explains part of the pruning failures in our synthetic experiments.

With γ=1.0, K grows logarithmically with firm size: E[K]≈3.9 at n=50, E[K]≈7.5 at n=1000. This is **economically meaningful** — larger firms genuinely operate in more technology areas. The logarithmic scaling provides natural regularization without size-stratified γ values.

**mean_precision_prior κ₀ = 1.0.** Controls shrinkage of component means toward the global mean. Posterior mean shrinkage is κ₀/(κ₀ + n_k): a cluster with n_k=5 has 17% shrinkage; n_k=50 has 2%. Each dimension's mean update is independent under diagonal covariance, so dimensionality does not amplify shrinkage.

**degrees_of_freedom_prior ν₀ = 52 (= d+2).** The minimum for a proper Wishart prior is ν₀ = d = 50, but the posterior expected covariance E[Σ_k | data] = Λ_n / (ν_n - d - 1) requires ν_n > d+1 to exist. Setting ν₀ = d+2 ensures finite posterior mean even for singleton clusters, providing a regularization anchor.

**mean_prior = global_mean, covariance_prior = global_var.** Computed once from the pooled dataset (~1.45M patents). This avoids per-firm double-counting while providing the same shrinkage-toward-global-structure benefit as a hierarchical model. The shrinkage factor κ₀/(κ₀ + n_k) is exactly the partial pooling weight in a hierarchical Normal model.

### Why Bayesian GMM over BIC Sweep?

| Factor | BIC Sweep | Bayesian GMM |
|--------|-----------|--------------|
| **Fits per firm** | K_max (10) | 1 |
| **Total fits** | 15,814 × 10 = 158,140 | 15,814 |
| **Handles firm-size skew** | No — BIC can favor K=1 for very small firms, K_max for large firms (monotonic decrease) | Yes — Dirichlet prior naturally regularizes toward fewer components for smaller datasets |
| **Hyperparameters** | K_max only | K_max + concentration prior + NIW hyperparameters |
| **Determinism** | Deterministic given random_state | Deterministic given random_state |
| **Interpretability** | BIC curve with clear elbow → interpretable | Effective K from weight threshold → slightly less interpretable |
| **BIC still available** | By definition | Can compute post-hoc for validation |

**The decisive factor is computational cost.** 158K fits vs 16K fits is a 10x difference. With `n_init=5`, the Bayesian approach does 79K fits total — still 2x faster than BIC sweep with `n_init=1`.

**The secondary factor is natural regularization.** The Dirichlet process prior provides a principled mechanism for small firms to auto-select K=1 or K=2 without requiring explicit thresholds in the BIC selection logic. BIC can behave poorly in high dimensions (50D), sometimes favoring too many components due to the weak penalty relative to the log-likelihood improvement from splitting.

### Bayesian Workflow

The Bayesian methodology audit (STAT 405) identified that our initial approach lacked the iterative prior-check-revise workflow. The following steps are required:

**Phase 0 — Prior Predictive Simulation (before any real data fitting):**
Draw parameters from the prior, simulate synthetic patent portfolios, check reasonableness. Run 500-1000 draws. Check: distribution of effective K (should concentrate at 1-8), per-dimension range (should match UMAP output scale), no degenerate draws.

**Phase 1 — Pilot Fit (200-300 stratified firms):**
Fit with calibrated priors. Run full posterior predictive checks (PPC) and sensitivity sweep on this subsample. Iterate prior revision if needed (expect 1-2 cycles).

**Phase 2 — Full-Scale Fit:**
Fit all 15,814 firms. Compute automated PPC scores per firm. Flag worst 2-5% for manual review.

**Phase 3 — Sensitivity Audit (200-500 firms):**
Sweep γ ∈ [0.1, 0.5, 1.0, 5.0], κ₀ ∈ [0.01, 0.1, 1.0, 10.0], ν₀ ∈ [50, 52, 60, 75, 100]. Stability criterion: effective K changes < 0.5 across one log-decade of γ for >90% of subsample.

**Optional — MCMC Validation:**
Run conjugate Gibbs sampling (Normal-Gamma for diagonal case) on ~500 highest-difficulty firms. Compare VI vs Gibbs posterior means. If Spearman r > 0.90, VI is validated for the full dataset.

### Empirical Validation (Synthetic Tests)

Pre-ADR experiments on synthetic data in 50D with diagonal covariance:

| Test | Data | K_max | Result |
|------|------|-------|--------|
| 3 clusters (200 pts, 10D) | Well-separated | 10 | Effective K=3 across all priors (0.001–10.0) |
| 2 clusters (100 pts, 50D) | Well-separated | 10 | Effective K=2, weights ≈ [0.495, 0.475] |
| 2 clusters (50 pts, 50D) | Well-separated | 10 | Effective K=6 — overfitting at small n |
| 2 clusters (30 pts, 50D) | Well-separated | 10 | Effective K=10 — severe overfitting |
| Random (100 pts, 50D) | No structure | 10 | Effective K=1 — correctly collapses |

**Note**: These experiments used γ=0.1. The overfitting at small n is partly attributable to the prior suppressing multi-component exploration (see E[K] table). With γ=1.0, the prior provides more room for the data to express multi-component structure — but the fundamental data scarcity issue at n<50 in 50D remains (ADR-005).

### EDA Validation Required

The EDA (Phase 1) must validate this decision on real data:
1. **Prior predictive simulation**: Generate 500 synthetic portfolios from the prior. Check for plausibility.
2. **K sensitivity on subsample**: For 20 firms of varying sizes, fit Bayesian GMM and BIC sweep. Compare effective K selections. Do they agree for well-sized firms?
3. **Concentration prior sensitivity**: Sweep γ ∈ [0.1, 0.5, 1.0, 5.0] on 50 firms. Is effective K stable?
4. **Posterior predictive checks**: Per-dimension KS statistics, pairwise distance distributions, assignment entropy.
5. **BIC monotonicity check**: For 10 large firms (>500 patents), does BIC have a clear elbow, or is it monotonically decreasing?

If the EDA reveals that Bayesian GMM produces unreliable K estimates, we fall back to BIC sweep.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| **BIC sweep (K=1..K_max)** | 10x computational cost; BIC can behave poorly in high dimensions; no natural regularization for small firms |
| **AIC instead of BIC** | AIC penalizes model complexity less than BIC, tends to overfit (selects higher K). BIC is more conservative and appropriate when we want parsimonious models. |
| **Silhouette-based K selection** | Silhouette score requires hard cluster assignments; GMMs produce soft assignments. Using argmax of responsibilities discards the probabilistic nature. Also O(n²) per evaluation. |
| **Elbow method on inertia** | K-means concept, not applicable to GMMs. No probabilistic interpretation. |
| **Fixed K for all firms** | Ignores the fundamental variation in firm technological breadth. A focused biotech firm should have K=1-2; a diversified conglomerate should have K=5-8. |
| **Cross-validation** | Computationally prohibitive at 15K firms. Adds complexity without clear benefit over BIC/Bayesian for mixture models. |
| **MCMC (Stan/NUTS)** | Stan cannot sample discrete latent variables (cluster assignments). HMC gets trapped in one of K! equivalent posterior modes. Explicitly not recommended for mixture models per the methodology audit. |
| **MCMC (Gibbs)** | Correct approach for mixture models, but 10-30x slower than VI. Reserved for validation subsample, not production. |

## Sensitivity Condition

**K_max sensitivity is elevated from a tuning detail to a reported methodological condition.**

BC ranking stability analysis on 99 pilot firms (4,851 pairs) revealed:

| Comparison | Spearman ρ | Top-50 overlap | Top-5 NN overlap |
|:----------:|:----------:|:--------------:|:----------------:|
| K=10 vs K=15 | 0.807 | 48% | 62% |
| K=10 vs K=20 | 0.774 | 22% | 53% |
| K=15 vs K=20 | 0.792 | 36% | 64% |

Bulk rankings are moderately stable (ρ ≈ 0.80), but the top tail — the only part that matters for candidate-pair discovery — is materially unstable. This means K_max affects which specific firm pairs would be flagged as M&A candidates.

**Week 3 contract requirement**: Any BC-based findings must be accompanied by K_max robustness checks. For the production run:
1. Fit GMMs at K_max ∈ {10, 15, 20}
2. Compute BC rankings under each setting
3. Report firm pairs as either **robust** (appear consistently across K_max settings) or **model-sensitive** (ranking depends on K_max)

This is a methodological limitation, not a model failure. The diagonal GMM with Bayesian K selection is the best tractable approximation for 15K firms in 50D, but downstream conclusions about specific firm-pair similarity must acknowledge sensitivity to K_max.

## Consequences

- Each firm gets a single Bayesian GMM fit with K_max=15 (provisional). Effective K is determined by weight thresholding.
- Components with weight < 0.01 are discarded from serialized output (not stored in `firm_gmm_parameters.parquet`).
- The Dirichlet process prior (γ=1.0) provides E[K] ≈ log(n) scaling — larger firms naturally express more components.
- Prior hyperparameters are computed from the global pooled dataset (global empirical Bayes), not per-firm.
- ELBO (variational lower bound on log-evidence) is stored per firm for validation. sklearn's `BayesianGaussianMixture` does not provide BIC; ELBO serves the same purpose as a model evidence proxy.
- The Bayesian workflow (prior predictive simulation, PPC, sensitivity analysis) is a required part of the EDA, not optional.
- The presentation's "Other?" option (slide 7, red text) is addressed: Bayesian GMM IS the "other" — it subsumes the BIC approach by providing automatic model selection.
- **K_max sensitivity is a reported methodological condition.** Week 3 must report robustness of BC rankings across K_max ∈ {10, 15, 20}. Firm pairs whose rankings are unstable across K_max settings are flagged as model-sensitive.
- **Revisit trigger**: If Week 3 BC validation shows that K_max sensitivity undermines the M&A prediction task (e.g., known merger pairs are not consistently ranked as similar), K_max selection must be revisited — potentially with per-firm BIC-based K selection for large firms.
