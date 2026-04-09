# Request for Information: Bayesian GMM Workflow for Firm Patent Portfolios

**From**: Firm-Pair Merger Prediction development team  
**To**: Bayesian statistics expert (STAT 405 context)  
**Date**: 2026-04-08  
**Purpose**: Audit our planned Bayesian Gaussian Mixture Model approach, identify gaps in our statistical methodology, and propose a principled Bayesian workflow grounded in academic best practices.

---

## 1. Problem Statement

We are building a research pipeline to predict M&A (merger and acquisition) pairs among technology firms based on their patent portfolios. The pipeline has four stages:

1. **Vectorize patents** → 50-dimensional vectors per patent (COMPLETE)
2. **Firm portfolio representation** → fit a GMM per firm on its patent vectors (THIS STAGE)
3. **Compare distributions** → Bhattacharyya Coefficient between firm GMMs
4. **Extensions** → synthetic portfolio matching

**Stage 2 is the critical modeling step.** Each firm owns a set of patents, each represented as a 50D real-valued vector (produced by a UMAP dimensionality reduction of 1536D neural embeddings). We want to represent each firm's patent portfolio as a probability distribution — specifically, a Gaussian Mixture Model — so that downstream we can compute distributional overlap (Bhattacharyya Coefficient) between pairs of firms.

The GMM mixing weights have direct economic interpretation: a firm with weights [0.6, 0.3, 0.1] has 60% of its patents in one technology area, 30% in another, 10% in a third. The GMM decomposition captures **how many** distinct technology areas a firm operates in (K), **what** each area looks like (component means and covariances), and **how much** of the portfolio is in each area (mixing weights).

---

## 2. Data Characteristics

### 2.1 The Vectors

Each patent is a 50-dimensional real-valued vector produced by UMAP reduction of PatentSBERTa embeddings. Key properties:

- **Dimensionality**: d = 50
- **Total patents**: ~1.45 million across 15,814 firms
- **UMAP output characteristics** (known from 768D pre-UMAP analysis, 50D not yet measured — production run still in progress):
  - Pre-UMAP L2 norms are NOT unit-normalized (mean ≈ 6.8 in 768D)
  - UMAP uses cosine metric internally (L2-normalizes before learning the manifold)
  - 50D output dimensions are expected to be weakly correlated (UMAP optimizes local structure, not global linear relationships)
  - 2D UMAP projection shows meaningful firm-level clustering (patents from focused firms cluster tightly; diversified mega-firms span the space)
- **Continuous, real-valued, non-negative not guaranteed**: UMAP output can be positive or negative
- **No known distributional form**: We do not know a priori that patent vectors follow Gaussian mixtures. The Gaussian assumption is a modeling choice justified by: (a) the methodology literature (Choi et al. 2019), (b) the economic interpretability of mixture weights, (c) the closed-form BC between Gaussians

### 2.2 The Firm-Size Distribution

This is the central statistical challenge. The distribution of patents-per-firm is extremely right-skewed:

| Quantile | Estimated Patent Count |
|----------|----------------------|
| 50th percentile | ~10 |
| 75th percentile | ~50 |
| 90th percentile | ~300 |
| 99th percentile | ~5,000 |
| Max | >100,000 (e.g., IBM, Samsung) |

- ~50% of firms have fewer than 10 patents
- ~20-30% of firms have 50+ patents (candidates for multi-component GMM)
- Top firms have 10,000-100,000+ patents

**This means we are fitting the same model class across a sample-size range spanning 4 orders of magnitude.** A firm with 5 patents in 50D is a fundamentally different statistical problem than a firm with 50,000 patents.

### 2.3 Co-Assigned Patents

~85,000 patents are co-assigned (owned by multiple firms). These appear in full in each firm's patent set. This is by design — a jointly-owned patent should contribute fully to both firms' portfolio distributions.

---

## 3. Current Implementation Plan (What We've Decided — Pending Audit)

We have drafted four Architecture Decision Records (ADRs) that are pending review. Here is a condensed summary of what we currently plan:

### 3.1 K Selection (ADR-004): Bayesian GMM with Dirichlet Process Prior

**Plan**: Use scikit-learn's `BayesianGaussianMixture` with `weight_concentration_prior_type="dirichlet_process"` to auto-determine K per firm. Set K_max=10 as the upper bound. Count "effective K" as components with weight > 0.01 after fitting.

**Preliminary evidence**: Synthetic experiments show Bayesian GMM correctly identifies K=3 in 10D with 200 points, and K=2 in 50D with 100 points. Robust across concentration priors (0.001 to 10.0) for well-separated clusters.

### 3.2 Minimum Patent Threshold (ADR-005): Three Tiers

| Tier | n (patents) | Treatment |
|------|-------------|-----------|
| Exclude | < 5 | No distributional representation |
| Single Gaussian | 5 – 49 | K=1, diagonal covariance |
| Full GMM | ≥ 50 | Bayesian GMM, K_max=10 |

**Preliminary evidence**: Synthetic experiments in 50D show Bayesian GMM cannot reliably prune spurious components until n ≈ 50, even with well-separated clusters.

### 3.3 Covariance Type (ADR-006): Diagonal

**Plan**: Use `covariance_type="diag"` for all components. Each dimension gets its own variance but no cross-dimension correlations are modeled.

**Rationale**: Full covariance in 50D requires 1,275 parameters per component (vs 50 for diagonal). Most firms don't have enough data. UMAP dimensions are expected to be weakly correlated.

### 3.4 Pre-Normalization (ADR-007): Pending EDA

**Plan**: Default to raw (no normalization) but run a sensitivity check comparing raw, L2-normalized, and z-score standardized vectors before finalizing.

---

## 4. What scikit-learn's BayesianGaussianMixture Actually Does

This is critical context for the audit, because sklearn makes many default choices that may or may not be appropriate for our problem.

### 4.1 Inference Method: Variational Bayes (NOT MCMC)

sklearn uses **mean-field variational inference** to approximate the posterior. It optimizes a lower bound on the model evidence (ELBO) rather than sampling from the true posterior. This means:

- It finds a single "best" approximate posterior (a product of independent factors)
- It does NOT produce posterior samples or credible intervals
- It cannot represent multi-modal posteriors (if two clusterings are equally good, VI picks one)
- It is fast (one optimization per firm) but may underestimate posterior uncertainty
- It converges to a local optimum (hence `n_init=3` to try multiple starting points)

### 4.2 Prior Structure (Normal-Inverse-Wishart conjugate family)

sklearn's `BayesianGaussianMixture` uses the following prior structure:

| Parameter | Prior | sklearn Default | What It Controls |
|-----------|-------|----------------|------------------|
| **Mixing weights** π | Dirichlet Process (stick-breaking) | `weight_concentration_prior = 1/K_max` | Sparsity of component usage. Lower → fewer active components. |
| **Component means** μ_k | Gaussian: N(μ₀, Σ_k / κ₀) | `mean_prior = mean(X)`, `mean_precision_prior = 1.0` | Where cluster centers can be placed. κ₀=1 is weakly informative; means are loosely anchored to global data mean. |
| **Component covariances** Σ_k | Wishart: W(ν₀, S₀) | `degrees_of_freedom_prior = d = 50`, `covariance_prior = empirical_cov(X)` | Shape and scale of clusters. ν₀=d is the minimum for a proper prior. S₀ set from data (empirical Bayes). |

**Key observation**: sklearn's defaults use **empirical Bayes** for the mean and covariance priors — it sets them from the data itself (global mean and empirical covariance of X). This is computationally convenient but:

- Conflates the prior with the data (using the data twice)
- May not reflect genuine prior knowledge about patent technology spaces
- The `mean_precision_prior = 1.0` and `degrees_of_freedom_prior = 50` are essentially "let the data speak" defaults with minimal regularization

### 4.3 What sklearn Does NOT Do

- **No prior predictive simulation**: There is no built-in way to simulate from the prior and check if it produces reasonable data
- **No posterior predictive checks**: No built-in PPC
- **No posterior uncertainty quantification**: VI returns point estimates of the approximate posterior mode, not samples
- **No label switching handling**: VI sidesteps this (it finds one mode), but this means it may miss equally valid clusterings
- **No hierarchical structure**: Each firm is fitted independently. There is no sharing of information across firms.

---

## 5. What We Need From You (Specific Questions)

### 5.1 Prior Choice

**Q1: Are the sklearn defaults reasonable for our problem?** Specifically:
- `mean_prior = mean(X)` (global mean of firm's patent vectors) — is empirical Bayes acceptable here, or should we specify an external prior?
- `mean_precision_prior = 1.0` — how does this interact with 50D data? Is this "weakly informative" or effectively uninformative in 50D?
- `degrees_of_freedom_prior = 50` (= d) — this is the minimum for a proper Wishart. Should we set it higher for more regularization?
- `covariance_prior = empirical_cov(X)` — again empirical Bayes. Should we use a scaled identity matrix instead? What are the tradeoffs?

**Q2: What does principled prior elicitation look like here?** We have domain knowledge:
- Patent vectors come from UMAP, so they have specific scale properties
- Technology areas within a firm should be "separable" (distinct clusters)
- We expect K to range from 1 to ~8 for most firms
- Smaller firms should naturally have fewer components
- The Dirichlet concentration prior controls sparsity — how should it be set relative to K_max?

**Q3: For the Dirichlet Process concentration parameter**, we currently use 0.1 (encourages sparsity). Is this appropriate? Should it vary by firm size? The literature suggests α < 1 encourages fewer components, α > 1 encourages more. What is the principled way to choose this?

**Q4: Should we consider a hierarchical prior structure?** Rather than fitting each firm independently, could we share prior information across firms (e.g., a global estimate of typical within-cluster covariance, or a shared Dirichlet concentration)? This could help stabilize estimates for small firms. Is this tractable for 15K firms?

### 5.2 Inference Method

**Q5: Is variational inference adequate for our problem, or do we need MCMC?**
- Our data is 50-dimensional with diagonal covariance (100 parameters per component, not 1,325)
- We have 15,814 firms to fit — MCMC per firm may be computationally prohibitive
- We care about point estimates of GMM parameters for downstream BC computation, not posterior uncertainty per se
- But: VI may produce overconfident estimates, especially for small firms where the posterior is broad

**Q6: If MCMC, what flavor?** Given 50D diagonal covariance:
- Gibbs sampling with conjugate NIW priors? (Conditionals are tractable for diagonal case)
- HMC via Stan or PyMC? (More flexible but slower; the 50D argument in the conversation above)
- Could we use VI for the bulk of firms and MCMC for a validation subsample to check VI's adequacy?

**Q7: Is there a middle ground?** E.g., variational inference with post-hoc posterior predictive checks to validate the VI approximation? Or Laplace approximation?

### 5.3 Bayesian Workflow

**Q8: What should our prior predictive simulation look like?** Before fitting any real data, we should:
- Draw parameters from our priors
- Simulate "fake patent portfolios" from the GMM with those parameters
- Check: Do the simulated portfolios look reasonable? (Cluster separation, number of clusters, spread)
- What "reasonable" means here: patent vectors should look like they could come from a technology firm's portfolio

**Q9: What posterior predictive checks should we perform?** After fitting, we should:
- Simulate new data from the fitted GMM
- Compare to the real data
- What metrics should we compare? (Cluster assignment entropy? Marginal distributions per dimension? Inter-point distance distributions?)

**Q10: What does prior sensitivity analysis look like at our scale?** With 15K firms:
- We can't re-run the full pipeline with different priors multiple times
- Can we do sensitivity analysis on a subsample (e.g., 100 firms spanning the size distribution)?
- What hyperparameters should we vary and what stability metrics should we monitor?

**Q11: What is the iterative workflow?**
1. Choose initial priors
2. Prior predictive check → iterate if unreasonable
3. Fit model (VI or MCMC)
4. Posterior predictive check → iterate if model fails to capture data features
5. Prior sensitivity analysis → check robustness
6. At what point do we declare the model "good enough"?

### 5.4 Model Adequacy

**Q12: Is a Gaussian mixture the right model class?** The Gaussianity assumption gives us closed-form BC, but:
- UMAP output may not be well-modeled by Gaussians (it could have non-convex cluster shapes)
- Are there diagnostic tests for "how Gaussian" our clusters are?
- What happens to BC computation if the Gaussian assumption is violated?

**Q13: Should we consider alternative mixture models?** E.g., t-distributions (heavier tails, more robust to outliers), or non-parametric density estimation (KDE)?

### 5.5 The Small-Firm Problem

**Q14: What is the principled Bayesian treatment of firms with 5-49 patents in 50D?** Our current plan forces K=1 for these firms. But:
- With 5 patents in 50D, the posterior is dominated by the prior
- Is it more honest to use the prior predictive distribution rather than a "fitted" model?
- Should we report posterior uncertainty for small firms (even if we use point estimates for downstream BC)?
- Is there a minimum sample size below which Bayesian updating is not meaningful?

**Q15: For the exclusion threshold (n < 5)**, is this justified from a Bayesian perspective? In principle, Bayesian methods can update even with n=1 — the posterior is just close to the prior. Should we include all firms and let the posterior reflect the uncertainty?

---

## 6. Constraints and Practical Considerations

### 6.1 Computational Budget
- The full fitting run should complete in a reasonable time on a single machine (CPU, 32-128 GB RAM)
- 15,814 firms, ranging from 5 to 100K+ patents each
- sklearn VI on all firms is estimated at 30 min to 2 hours
- MCMC per firm would need to be budgeted carefully (seconds per firm for small firms; possibly minutes for large firms)

### 6.2 Downstream Use
- The output is a set of GMM parameters (means, covariances, weights) per firm
- These are consumed by the Bhattacharyya Coefficient computation (Week 3), which has a closed form for Gaussian mixtures
- BC between two GMMs with K₁ and K₂ components involves K₁ × K₂ pairwise component comparisons, weighted by the mixing weights
- Point estimates of GMM parameters are sufficient for BC computation; posterior samples would enable uncertainty propagation but add significant complexity

### 6.3 Team Expertise
- The development lead (Torrin Pataki) has formal Bayesian statistics training (STAT 405 with Alexandre Bouchard-Côté: Bayesian workflow, model selection, MCMC, advanced inference)
- The implementation agent (Claude Code) can implement whatever methodology is recommended
- The team includes economics researchers who need interpretable results
- Codex (impartial reviewer) will audit the implementation for correctness

### 6.4 Tools Available
- **scikit-learn**: `GaussianMixture`, `BayesianGaussianMixture` (variational inference)
- **PyMC**: Full probabilistic programming (MCMC via NUTS/HMC, VI, prior/posterior predictive simulation)
- **Stan (via CmdStanPy)**: HMC with NUTS; excellent for high-dimensional posteriors; Torrin has cmdstan installed at `~/.cmdstan/`
- **scipy**: Wishart, Dirichlet distributions for manual prior predictive simulation
- All available in our Python venv

### 6.5 What We've Already Done
- 4 ADRs drafted (K selection, min patents, covariance type, normalization)
- Interface spec written (PortfolioBuilder, GMMFitter)
- Config structure defined
- Synthetic experiments run (sklearn BayesianGMM on random and structured 50D data)
- All Week 1 pipeline code complete and tested (45 tests)

---

## 7. Summary of What We're Asking

In priority order:

1. **Audit our prior choices** — are the sklearn defaults adequate, or do we need principled prior elicitation?
2. **Recommend an inference approach** — VI (sklearn), MCMC (PyMC/Stan), or a hybrid?
3. **Propose a Bayesian workflow** — prior predictive checks, sensitivity analysis, posterior predictive checks, iteration criteria
4. **Address the small-firm problem** — what is the principled treatment for firms with very few patents relative to dimensionality?
5. **Assess model adequacy** — is GMM the right model class for UMAP output? What diagnostics should we run?

We are not looking for a "just use sklearn defaults" answer. We want a principled, research-grade Bayesian methodology that we can defend in an academic paper. At the same time, it must be computationally tractable for 15K firms.

---

## 8. Reference Architecture

For context, here is the complete pipeline architecture:

```
Raw Patent Text (1.5M patents, 15,814 firms)
    │
    ├── PatentSBERTa(title + abstract) → 768D
    ├── PatentSBERTa(cited abstracts) → mean pool → 768D
    │
    └── Concatenate → 1536D → UMAP → 50D per patent
                                        │
                                   Group by firm
                                        │
                              ┌─────────┼─────────┐
                              │         │         │
                           n < 5     5 ≤ n ≤ 49   n ≥ 50
                          EXCLUDE   SINGLE GMM   BAYESIAN GMM
                                   (K=1)        (K=1..10)
                                        │
                                   GMM Parameters
                              (means, covs, weights)
                                        │
                              Bhattacharyya Coefficient
                              (pairwise firm comparison)
                                        │
                              M&A Prediction Features
```

---

## Appendix A: sklearn BayesianGaussianMixture Full Prior Specification

For the reviewer's convenience, the complete prior structure as implemented by sklearn:

**Weights** (Dirichlet Process, stick-breaking representation):
```
v_k ~ Beta(1, γ)                    for k = 1, ..., K_max - 1
π_k = v_k ∏_{j<k} (1 - v_j)        stick-breaking construction
γ = weight_concentration_prior       (our current setting: 0.1)
```

**Means** (Gaussian, conditioned on covariance):
```
μ_k | Σ_k ~ N(μ_0, Σ_k / κ_0)
μ_0 = mean_prior                     (sklearn default: mean(X))
κ_0 = mean_precision_prior           (sklearn default: 1.0)
```

**Covariances** (for diagonal case, Gamma prior on precisions):
```
For covariance_type="diag":
λ_{k,j} ~ Gamma(ν_0 / 2, ν_0 * s_{0,j} / 2)    per dimension j
σ²_{k,j} = 1 / λ_{k,j}

ν_0 = degrees_of_freedom_prior       (sklearn default: d = 50)
s_{0,j} = covariance_prior[j]        (sklearn default: empirical variance of X along dimension j)
```

**Note**: For the diagonal case, the Wishart prior reduces to independent Gamma priors on each dimension's precision. This is simpler than the full Wishart and may make MCMC more tractable if we go that route.

## Appendix B: Synthetic Experiment Results

Results from pre-ADR experiments (conducted in Python, reproducible):

### B.1 Bayesian GMM Auto-Pruning (10D, 200 points, 3 true clusters)

| Concentration Prior | Effective K | Top Weights |
|--------------------:|:-----------:|-------------|
| 0.001 | 3 | [0.390, 0.351, 0.249] |
| 0.010 | 3 | [0.390, 0.351, 0.249] |
| 0.100 | 3 | [0.390, 0.351, 0.249] |
| 1.000 | 3 | [0.388, 0.350, 0.247] |
| 10.000 | 3 | [0.369, 0.350, 0.248] |

Conclusion: Correct K identified across 4 orders of magnitude of concentration prior.

### B.2 Bayesian GMM in 50D (2 true clusters, diagonal covariance)

| n (points) | Effective K (>0.01 weight) | Top 2 Weights | Behavior |
|-----------:|:--------------------------:|---------------|----------|
| 5 | 5 | [0.329, 0.265] | Every point = its own component |
| 10 | 10 | [0.180, 0.162] | Severe overfitting |
| 20 | 10 | [0.252, 0.225] | Cannot prune |
| 30 | 10 | [0.404, 0.223] | Largest weight concentrating but no pruning |
| 50 | 6 | [0.499, 0.427] | Beginning to prune; approaching correct K |
| 100 | 2 | [0.495, 0.475] | Correct K, correct weights |

Conclusion: In 50D, Bayesian GMM (VI) needs n ≈ 50-100 to reliably determine K. Below n=50, the approximate posterior cannot overcome the inherent ambiguity.

### B.3 Single Gaussian (K=1) on Small Samples

GaussianMixture with n_components=1, diagonal covariance:

| n | Converged | BIC |
|--:|:---------:|----:|
| 3 | True | 427.2 |
| 5 | True | 743.2 |
| 10 | True | 1,538.2 |
| 20 | True | 3,026.8 |

Conclusion: K=1 with diagonal covariance is always fittable (even n=3), but the estimate quality is unknown — the BIC scale is not comparable across n.
