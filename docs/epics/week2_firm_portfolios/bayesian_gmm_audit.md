# Bayesian GMM Methodology Audit: Complete Analysis

## Executive Summary

Your current ADR decisions are **directionally sound but underspecified on the Bayesian side**. The three most consequential issues, in priority order:

1. **γ = 0.1 is too sparse** — it implies E[K] ≈ 0.6 for n=50, which means the DP prior actively suppresses multi-component solutions even when the data clearly supports them. Raise to γ = 0.5–1.0.

2. **sklearn's empirical Bayes defaults double-count data** — setting `mean_prior = mean(X_firm)` and `covariance_prior = empirical_cov(X_firm)` per firm uses each firm's data twice (once to set the prior, once to compute the posterior). Switch to **global** empirical Bayes: compute hyperparameters from the pooled dataset of all firms, then apply as fixed priors to each independent fit.

3. **VI is adequate for your downstream task** — you need point estimates of GMM parameters for Bhattacharyya coefficient computation, not posterior uncertainty. sklearn's variational Bayes gives you this. The label-switching problem that makes MCMC difficult for mixtures is irrelevant because BC is computed from the mixture density, which is permutation-invariant.

---

## Part 1: Prior Choice Audit (Q1–Q4)

### Q1: Are sklearn defaults reasonable?

**Partially. Three of four need adjustment.**

| Parameter                  | sklearn Default         | Problem                                                                         | Recommendation                                           |
| -------------------------- | ----------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `mean_prior`               | `mean(X_firm)`          | Per-firm EB double-counts data                                                  | `mean(X_pool)` — global centroid across all firms        |
| `mean_precision_prior`     | 1.0                     | Actually reasonable — κ₀=1 gives 1/(1+n) shrinkage toward µ₀                    | Keep at 1.0                                              |
| `degrees_of_freedom_prior` | d = 50                  | Boundary of proper prior; posterior mean of Σ undefined for clusters with n_k=1 | Set to d+2 = **52** (minimum for finite posterior mean)  |
| `covariance_prior`         | `empirical_cov(X_firm)` | Per-firm EB double-counts data                                                  | `np.var(X_pool, axis=0)` — global per-dimension variance |

**Why κ₀ = 1 is fine**: The posterior mean shrinkage is κ₀/(κ₀ + n_k). For κ₀=1: a cluster with n_k=5 has 17% shrinkage toward µ₀; n_k=50 has 2% shrinkage. This scales naturally with cluster size, not dimensionality. In 50D the dimensionality does not amplify or dampen mean shrinkage — each dimension's mean update is independent under diagonal covariance.

**Why ν₀ = 50 is problematic**: The posterior expected covariance E[Σ_k | data] = Λ_n / (ν_n - d - 1). With ν₀ = d = 50, this requires n_k > 1 for the expectation to exist. Setting ν₀ = 52 ensures the prior itself has a finite mean, providing a regularization anchor even for singleton clusters. For small firms where you want stronger regularization, ν₀ = 60–75 (1.2d–1.5d) adds meaningful shrinkage of covariance estimates toward the prior scale.

**Why global EB is principled**: Setting hyperparameters from the pooled dataset (all 1.45M patents) and applying them as fixed priors to each firm's independent fit is a tractable approximation to a full hierarchical model. It avoids within-firm double-counting while providing the same shrinkage-toward-global-structure benefit. This is sometimes called "Type II maximum likelihood" or "global empirical Bayes."

### Q2: Principled prior elicitation

The concrete prior specification I recommend:

```python
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

# Step 1: Compute global hyperparameters from pooled data (run once)
X_pool = np.concatenate([firm_vectors[gvkey] for gvkey in all_gvkeys])
global_mean = np.mean(X_pool, axis=0)         # shape (50,)
global_var = np.var(X_pool, axis=0)            # shape (50,)

# Step 2: Fit each firm with globally-calibrated priors
def fit_firm(X_firm, K_max=10):
    bgm = BayesianGaussianMixture(
        n_components=K_max,
        covariance_type='diag',
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1.0,        # gamma — see Q3
        mean_prior=global_mean,                # anchored globally
        mean_precision_prior=1.0,              # kappa_0: weakly informative
        degrees_of_freedom_prior=52,           # nu_0 = d + 2
        covariance_prior=global_var,           # per-dimension scale from global pool
        max_iter=200,
        n_init=5,                              # up from 3 for stability
        random_state=42,
    )
    bgm.fit(X_firm)
    return bgm
```

**The IW prior's known pathology in high d**: Even with ν₀ = d+1, the Inverse-Wishart creates artificial coupling between variance and correlation estimates — high-variance dimensions are biased toward high correlations. For d=50 this is not just a theoretical concern. The diagonal covariance assumption (ADR-006) actually _solves_ this by decomposing the IW into 50 independent Gamma priors on precisions, eliminating the variance-correlation coupling entirely. **This is an additional theoretical justification for ADR-006 beyond the computational argument.**

### Q3: Dirichlet Process concentration parameter

**γ = 0.1 is too aggressive.** Here is the E[K] table computed from E[K] = γ · (ψ(γ + n) − ψ(γ)):

| γ       | n=10    | n=50    | n=100   | n=500   | n=1000  | n=10000  |
| ------- | ------- | ------- | ------- | ------- | ------- | -------- |
| **0.1** | **0.4** | **0.6** | **0.7** | **1.0** | **1.0** | **1.3**  |
| 0.5     | 1.5     | 2.3     | 2.7     | 3.7     | 4.2     | 6.0      |
| **1.0** | **2.5** | **3.9** | **4.6** | **6.5** | **7.5** | **10.9** |
| 2.0     | 4.0     | 6.4     | 7.6     | 11.0    | 13.0    | 19.8     |

With γ=0.1, even a firm with n=10,000 patents has a prior expectation of only 1.3 active components. The stick-breaking prior weight E[w₁] = 1/(1+γ) = 0.91, meaning the first component gets 91% of prior mass. This is so sparse that the posterior has to overcome a strong prior preference for K=1 before adding components.

**Your synthetic experiments in Appendix B.2 are consistent with this**: the failure to prune in 50D at n<50 is partly because γ=0.1 doesn't have enough prior mass on multi-component solutions to help VI explore them, and partly because n<50 genuinely lacks information.

**Recommendation: γ = 1.0** (fixed across all firms). This gives:

- n=50: E[K] ≈ 3.9 — allows the data to support 2–5 components
- n=500: E[K] ≈ 6.5 — large firms can express more complexity
- n=10,000: E[K] ≈ 10.9 — naturally bumps against K_max=10

The logarithmic growth E[K] ≈ γ·log(n) means that with γ=1.0, K grows naturally with firm size. This is _economically meaningful_: larger firms genuinely operate in more technology areas. A fixed γ that allows this natural scaling is more defensible than size-stratified γ values, which would require justifying the tier boundaries.

**Should γ vary by firm size?** Three defensible options exist:

1. **Fixed γ=1.0** (recommended for simplicity and defensibility)
2. **Size-stratified**: γ=0.5 for n<100, γ=1.0 for 100≤n<1000, γ=2.0 for n≥1000
3. **Gamma hyperprior**: place Gamma(2,2) on γ — but this requires MCMC, not VI

The 2026 Diatchkova et al. paper (arxiv 2602.06301) warns that Gamma(1,1) as a hyperprior on γ causes >60% posterior collapse. If you ever go the hyperprior route, use a calibrated Gamma with mass away from zero.

### Q4: Hierarchical prior structure

**Full hierarchical modeling across 15K firms is computationally intractable for MCMC.** A joint model over 15K firms × 10 components × 100 parameters/component ≈ 15M parameters; MCMC cannot sample this.

**Global empirical Bayes (recommended above) is the practical approximation to a hierarchical model.** It provides the same directional benefit: small firms' estimates are shrunk toward the global structure, large firms are nearly unaffected. The mathematical mechanism is identical to partial pooling — the shrinkage factor κ₀/(κ₀ + n_k) is exactly the partial pooling weight in a hierarchical Normal model.

**The incremental benefit of full hierarchical over global EB is largest when**:

- Between-firm covariance heterogeneity is high (biotech vs. semiconductor firms have very different within-cluster structures)
- Most firms are small (prior-dominated)
- Calibrated uncertainty for small firms matters

For your downstream task (BC rankings, not calibrated uncertainty), global EB is sufficient.

---

## Part 2: Inference Method (Q5–Q7)

### Q5: Is VI adequate?

**Yes, for your specific downstream task.** The key reasoning:

1. **You need point estimates (µ_k, σ²_k, π_k) for BC computation, not posterior samples.** VI gives you the variational posterior mode, which is a good point estimate when the posterior is approximately unimodal (which it is for well-separated clusters with sufficient data).

2. **VI's known failure modes are not critical for your use case:**
   - _Variance underestimation_ — affects credible intervals, which you don't need
   - _Mode-seeking (reverse KL)_ — may undercount K when clusters overlap, but this is conservative for BC (merging overlapping clusters increases BC, making firm pairs look _more_ similar, not less)
   - _Mean-field independence assumption_ — drops µ-σ² correlation within components, but for diagonal covariance the correlation is less severe

3. **VI's structural advantage for mixtures**: it sidesteps label switching entirely by converging to one mode. Since BC is permutation-invariant over components, this is fine.

4. **Computational budget**: VI on all 15,814 firms ≈ 1–3 hours on a 32-core machine. MCMC ≈ 30–70+ hours (Gibbs in NumPy) or 3–10 hours (JAX-accelerated Gibbs). The 10–30× cost difference is real.

### Q6: If MCMC, what flavor?

**Gibbs sampling with conjugate Normal-Gamma priors, NOT Stan/NUTS.** Here's why:

**Stan is the wrong tool for mixture models.** Stan cannot sample discrete latent variables (cluster assignments z_i). The standard workaround marginalizes out z via log_sum_exp, but this leaves K! equivalent posterior modes in the continuous parameter space. NUTS gets trapped in one mode and cannot cross the energy barriers between modes. Stan's own documentation explicitly warns about this: "the sampler gets stuck in one of the bowls around the modes."

**Gibbs with data augmentation (explicit z_i) is the correct MCMC approach for diagonal GMMs.** The conditional posteriors are all closed-form:

| Step | Parameter             | Conditional Posterior                                 |
| ---- | --------------------- | ----------------------------------------------------- |
| 1    | z_i (assignments)     | Categorical(softmax(log π_k + log N(x_i; µ_k, σ²_k))) |
| 2    | π (weights)           | Dirichlet(α₀ + n₁, ..., α₀ + n_K)                     |
| 3    | µ\_{k,j} (means)      | Normal((κ₀µ₀ + n*k x̄_k)/(κ₀+n_k), σ²*{k,j}/(κ₀+n_k))  |
| 4    | τ\_{k,j} (precisions) | Gamma(α₀ + n*k/2, β₀ + ½Σ(x*{ij}−µ\_{k,j})²)          |

Each Gibbs iteration is O(N × K × D). For N=1000, K=5, D=50: ~250K operations per iteration. At 2000 iterations: 0.5–5 seconds per firm in NumPy, 0.05–0.5 seconds in JAX.

### Q7: Middle ground?

**The recommended hybrid**: VI for bulk fitting + MCMC validation on a diagnostic subsample.

```
Step 1: Fit all 15,814 firms with sklearn BayesianGaussianMixture (VI)
        → Wall time: 1–3 hours

Step 2: Compute "difficulty scores" per firm:
        - ELBO variance across n_init restarts (high = unstable VI)
        - converged_ flag
        - effective K at K_max boundary
        - firm size (small n = unreliable VI)

Step 3: Select ~500 difficult firms for Gibbs validation

Step 4: Compare VI point estimates vs Gibbs posterior means
        - If Spearman correlation r > 0.90 for µ_k, σ²_k: VI validated
        - If correlation is lower: investigate systematic bias pattern
```

This is a recognized approach in genomics (fitting thousands of cell-type GMMs) and has the same structure as your problem.

---

## Part 3: Bayesian Workflow (Q8–Q11)

### Q8: Prior predictive simulation

Draw parameters from your prior, simulate synthetic patent portfolios, check if they look reasonable.

```python
from scipy.stats import dirichlet, gamma as gamma_dist
import numpy as np

def prior_predictive_sample(n_patents, K_max=10, D=50,
                            gamma=1.0, kappa_0=1.0, nu_0=52,
                            mu_0=None, s_0=None):
    """Simulate one firm's patent portfolio from the prior."""
    # weights: stick-breaking DP
    v = np.random.beta(1, gamma, size=K_max - 1)
    pi = np.zeros(K_max)
    remaining = 1.0
    for k in range(K_max - 1):
        pi[k] = v[k] * remaining
        remaining *= (1 - v[k])
    pi[K_max - 1] = remaining

    # component parameters
    means, variances = [], []
    for k in range(K_max):
        # precision per dimension ~ Gamma(nu_0/2, nu_0*s_0[j]/2)
        tau_k = gamma_dist.rvs(a=nu_0/2, scale=2.0/(nu_0 * s_0), size=D)
        sigma2_k = 1.0 / tau_k
        # mean ~ Normal(mu_0, sigma2_k / kappa_0)
        mu_k = np.random.normal(mu_0, np.sqrt(sigma2_k / kappa_0))
        means.append(mu_k)
        variances.append(sigma2_k)

    # generate data
    z = np.random.choice(K_max, size=n_patents, p=pi)
    X = np.array([np.random.normal(means[zi], np.sqrt(variances[zi])) for zi in z])
    return X, pi, means, variances
```

**What to check** (run 500–1000 prior predictive draws):

- Distribution of effective K (components with π_k > 0.01): Should be concentrated at 1–8
- Per-dimension range of X: Should match the scale of your actual UMAP output range
- Distribution of inter-point distances: Should look like real patent distances
- No degenerate draws (all points at the same location, or spread across ±10⁶)

**With sklearn's default κ₀ = 0.01 (not 1.0 — sklearn actually defaults to a very small mean precision)**: component means can scatter to ±10σ from the data mean, producing wildly implausible portfolios. This is the first thing prior predictive simulation will catch, and it's why I recommend κ₀ = 1.0 explicitly.

### Q9: Posterior predictive checks

After fitting, run automated PPCs. Since visual inspection of 15K firms is impossible, compute scalar discrepancy metrics:

**Tier 1 (always run):**

1. **Marginal KS statistic per dimension**: For each dimension d, compare observed X[:,d] to samples from the fitted GMM. Compute KS statistic. Report mean and max across dimensions. A `ks_mean > 0.10` flags marginal misfit.

2. **Pairwise distance distribution**: Compute pdist(X_obs) vs pdist(X_rep). The Wasserstein distance between these two distributions detects global structural misfit that marginal checks miss (e.g., correct marginals but wrong correlation structure).

3. **Component weight calibration**: Check if the proportion of data assigned to each component (via predict_proba) matches the fitted weights. Severe imbalance suggests label degeneracy.

**Tier 2 (run on subsample):**

4. **Assignment entropy**: H_i = −Σ_k q(z_i=k) log q(z_i=k). High average entropy means soft/ambiguous assignments (overlapping clusters). Compare to what the prior predicts.

5. **Per-component Mahalanobis Q-Q plot**: For each active component, compute Mahalanobis distances of assigned points and compare to χ²(d) distribution. Detects heavy tails (non-Gaussianity) within clusters.

**Scaling to 15K firms**: You cannot visually inspect 15K PPCs. Compute scalar scores per firm, build a distribution across firms, flag the worst 2–5% for manual review. Systematic failures that correlate with firm size or technology class indicate model misspecification for that subpopulation.

**Automated PPC code pattern:**

```python
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import pdist

def compute_ppc_scores(X_obs, model, n_rep=100):
    """
    Returns a dict of scalar PPC discrepancy scores for one firm's fitted model.
    """
    scores = {}

    # generate replicate datasets
    X_rep_list = [model.sample(len(X_obs))[0] for _ in range(n_rep)]

    # marginal KS statistic (mean and max over dims)
    ks_per_dim = []
    for d in range(X_obs.shape[1]):
        ks_stats = [ks_2samp(X_obs[:,d], X_rep[:,d]).statistic
                    for X_rep in X_rep_list]
        ks_per_dim.append(np.mean(ks_stats))
    scores['ks_marginal_mean'] = np.mean(ks_per_dim)
    scores['ks_marginal_max'] = np.max(ks_per_dim)

    # pairwise distance discrepancy
    obs_dists = pdist(X_obs[:200])  # subsample for speed
    rep_dists_list = [pdist(X_rep[:200]) for X_rep in X_rep_list[:10]]
    scores['wasserstein_pairwise'] = np.mean(
        [wasserstein_distance(obs_dists, rd) for rd in rep_dists_list]
    )

    # assignment entropy
    resp = model.predict_proba(X_obs)
    entropy = -np.sum(resp * np.log(resp + 1e-10), axis=1)
    scores['mean_assignment_entropy'] = np.mean(entropy)

    # effective K
    weights = model.weights_
    scores['effective_K'] = 1.0 / np.sum(weights**2)  # perplexity-style

    return scores
```

**Flag thresholds (calibrate on your data):**

- `ks_marginal_mean > 0.10` → marginal misfit
- `ks_marginal_max > 0.20` → at least one dimension severely misfit
- `effective_K` at K_max ceiling → model wants more components
- `effective_K < 1.5` for a firm that should be diverse → posterior collapse

### Q10: Sensitivity analysis at scale

**Run on a stratified subsample of 200–500 firms** (not all 15K):

- 50 smallest firms (most sensitive to priors)
- 50 largest firms (should be robust — confirm this)
- 50 random mid-size firms
- 50 firms with worst PPC scores
- Remaining stratified by technology class

**Hyperparameters to sweep (log-spaced grid):**

| Parameter                        | Grid                                   | Expected sensitivity                    |
| -------------------------------- | -------------------------------------- | --------------------------------------- |
| `weight_concentration_prior` (γ) | [0.01, 0.1, 0.5, 1.0, 5.0]             | **Highest** — directly controls K       |
| `mean_precision_prior` (κ₀)      | [0.01, 0.1, 1.0, 10.0]                 | Low for large firms; moderate for small |
| `degrees_of_freedom_prior` (ν₀)  | [50, 52, 60, 75, 100]                  | Low unless ν₀ very close to d           |
| `covariance_prior` scale factor  | [0.1×, 0.5×, 1×, 2×, 10×] × global_var | Moderate                                |

**Stability metrics per perturbation:**

- Δ effective_K (acceptable: < 0.5 across one log-decade)
- Δ posterior predictive log-likelihood (acceptable: < 0.5 nats/obs)
- Δ BC values for 100 test firm-pairs (acceptable: Spearman rank correlation > 0.95 vs. baseline)

**When results are "sensitive"**: If effective K changes by >1.5 (a whole cluster appears/disappears) when γ moves by one order of magnitude, the data are insufficiently informative for that firm. This is expected for small firms and is not a model failure — it's honest uncertainty.

### Q11: Iterative workflow

```
Phase 0: Prior Design (before any fitting)
├─ Prior predictive simulation (1000 draws, revised sklearn defaults)
├─ Check: synthetic portfolios look plausible?
└─ Iterate until prior predictive is reasonable (1-2 cycles)

Phase 1: Pilot (200-300 stratified firms)
├─ Fit with revised priors
├─ Full PPC battery + convergence diagnostics
├─ Sensitivity sweep on this subsample
├─ Identify systematic failure modes
└─ Iterate prior revision if needed (1-2 cycles)

Phase 2: Full-Scale Fit (15,814 firms)
├─ Fit all firms with finalized priors
├─ Automated PPC scores + convergence flags
├─ Flag ~2-5% problematic firms
└─ Optional: MCMC validation on flagged subset

Phase 3: Sensitivity Audit (200-500 firms)
├─ Full hyperparameter sweep
├─ Report which firm classes are sensitive
└─ Compute downstream BC sensitivity

Phase 4: Accept
├─ PPC medians within thresholds
├─ Sensitivity: effective K stable within ±0.5 for >90% of subsample
├─ Convergence: <3% of full sample flagged
└─ No systematic PPC failure correlated with scientifically important covariates
```

**Typical iteration count**: 2–4 prior revision cycles in Phase 0–1 before full-scale deployment.

**"Good enough" criteria:**

- PPC: median `ks_marginal_mean < 0.08` across firms
- Sensitivity: effective K stable within ±0.5 under 1 log-decade of γ perturbation for >90% of subsample
- Convergence: <3% of firms flagged as non-converged or degenerate
- No systematic PPC failure correlated with scientifically important covariates

---

## Part 4: Model Adequacy (Q12–Q13)

### Q12: Is GMM the right model class for UMAP output?

**Conditionally yes, with caveats that are partially mitigated by your 50D choice.**

**What supports GMM:**

- UMAP at 50D preserves substantially more global structure than 2–3D; cluster shapes are more elliptical and more faithfully reflect original topology
- GMM is the dominant tractable approach for parametric density estimation in embedding spaces
- For well-specialized firms with homogeneous portfolios, per-firm clusters are likely approximately Gaussian in 50D
- EM convergence under misspecification goes to the KL-projection (closest Gaussian in KL sense), which is a systematic bias that likely preserves BC ranking

**What poses risk:**

- UMAP does not preserve density — cluster sizes in the embedding don't reflect true portfolio concentration
- UMAP does not guarantee global distance preservation — inter-firm distances in embedding space are not linearly related to original cosine distances
- UMAP can produce artificial cluster tears (splitting one true cluster into disconnected blobs), creating spurious components
- The UMAP output kernel is heavy-tailed (generalized Student-t-like), meaning the optimization implicitly encourages clusters with heavier tails than Gaussian

**Diagnostics to run:**

1. **Mardia's test** (multivariate skewness + kurtosis) on per-component clusters for a subsample of firms. Available in R's `MVN` package.
2. **Per-dimension Q-Q plots** against Normal for individual cluster members
3. **Per-component Mahalanobis distance** vs. χ²(d) distribution — systematic heavy tails indicate the Gaussian assumption is costing you

**The ranking-preservation question is key**: Even if absolute BC values are biased by the Gaussian assumption, if the _ranking_ of firm pairs by BC is preserved (i.e., truly similar firms still rank higher than dissimilar firms), the model is adequate for your M&A prediction task. Validate this by computing a non-parametric alternative (Monte Carlo BC from kernel density estimates) on ~200 firm pairs and checking Spearman correlation with the Gaussian BC. If ρ > 0.85–0.90, the Gaussian approximation is practically adequate for ranking.

### Q13: Alternative mixture models

| Alternative          | Pros                                                          | Cons for Your Use Case                                                        |
| -------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **t-mixture**        | Heavier tails, robust to outliers, reduces to Gaussian as ν→∞ | No simple closed-form BC; requires numerical integration                      |
| **KDE**              | Non-parametric, no distributional assumption                  | No closed-form BC; curse of dimensionality at d=50; no compact representation |
| **von Mises-Fisher** | Natural for L2-normalized embeddings                          | Requires L2 normalization (destroys magnitude information); no simple BC      |
| **HDBSCAN**          | Handles non-convex clusters; UMAP docs recommend it           | Not a density model; no parameters for BC computation                         |

**Assessment**: The t-mixture is the most scientifically defensible upgrade from GMM, but the implementation cost is significant (custom EM, no sklearn implementation, no closed-form BC). For a first-pass pipeline, GMM with multi-component fits is defensible. If the Gaussianity diagnostics reveal systematic problems, upgrade to t-mixtures in Week 4.

**Recommendation**: Start with GMM. Run Gaussianity diagnostics (Mardia's test on within-cluster residuals, per-dimension Q-Q plots) on a sample of fitted firms. If >20% of clusters show significant non-Gaussianity (heavy tails, skew), consider the t-mixture upgrade for those specific firms.

---

## Part 5: The Small-Firm Problem (Q14–Q15)

### Q14: Principled treatment of firms with 5–49 patents in 50D

**The critical insight: diagonal covariance decomposes the 50D problem into 50 independent 1D problems.** You are never estimating a 50D posterior; you are running 50 independent Normal-Gamma updates, each with n observations. For n=10 in any single dimension, the Normal-Gamma posterior is perfectly well-defined and data-informed.

**Normal-Gamma conjugate posterior per dimension j:**

```
Prior:      µ_j | σ²_j ~ Normal(µ₀_j, σ²_j / κ₀)
            τ_j = 1/σ²_j ~ Gamma(α₀, β₀)

Posterior:  κ_n = κ₀ + n
            µ_n = (κ₀ µ₀ + n x̄_j) / (κ₀ + n)
            α_n = α₀ + n/2
            β_n = β₀ + ½[n·s²_j + κ₀·n·(x̄_j - µ₀)²/(κ₀ + n)]

Posterior predictive:  x_new | data ~ t(2α_n, µ_n, √(β_n/(κ_n·α_n)))
```

With κ₀=1, α₀=1 (weakly informative), the prior effective sample size (ESS) is:

- ESS for mean: κ₀ = 1
- ESS for variance: 2α₀ = 2

**This means even n=5 produces a posterior that is 83% data-driven for the mean and 71% data-driven for the variance.** The posterior is genuinely informed by data, not prior-dominated.

**Your K=1 restriction for n=5–49 is correct.** The synthetic experiments confirm that in 50D, the Bayesian GMM cannot reliably distinguish K=2 from K=1 until n ≈ 50. Forcing K=1 is not a limitation of Bayesian methods — it reflects the genuine information content of the data. The Bayesian Occam factor naturally penalizes K=2 when n is small relative to the model complexity.

**Should you report posterior uncertainty?** For the BC computation, you use point estimates (posterior means of µ and σ²). But you _could_ use the posterior predictive (Student-t per dimension) instead of the plug-in Gaussian. The difference matters for n < 30: the t-distribution has heavier tails, producing modestly lower BC values (5–15% for n=10 firms). Whether this matters depends on whether you need absolute BC values or just rankings. For rankings, plug-in Gaussian is sufficient.

### Q15: Is n < 5 exclusion justified?

**Yes, and it's more conservative than necessary.**

The pure Bayesian argument: n=1 produces a valid posterior. Nothing breaks mathematically. But for _downstream BC computation_, both firms' distributions need to reflect their actual patent profiles, not the shared prior. If both firms are prior-dominated, their BC reflects prior similarity (which is identical for all small firms), not firm similarity.

**Principled threshold derivation**: The posterior is "prior-dominated" when n < ESS_prior = κ₀ + 2α₀. With κ₀=1, α₀=1: ESS=3. So n < 3 is prior-dominated; n ≥ 4 is data-informed. Your n=5 threshold provides a safety margin and is defensible.

**A more sophisticated alternative**: Instead of a hard threshold, compute KL(posterior ‖ prior) per firm. Exclude firms where KL < ε (e.g., 0.1 nats), which directly measures whether the data moved the posterior. This would automatically adapt to prior strength and could include some n=3–4 firms while excluding pathological n=10 firms (if all 10 patents are identical, the posterior barely moves from the prior along most dimensions).

---

## Part 6: Concrete Recommendations Summary

### Changes to ADRs

**ADR-004 (K Selection)**: Change `weight_concentration_prior` from 0.1 to **1.0**. Change `n_init` from 3 to **5**.

**ADR-005 (Min Patents)**: Current tiers are well-justified. No change needed. Optionally replace the hard n<5 exclusion with a KL-based criterion.

**ADR-006 (Covariance Type)**: Diagonal is correct. Add the theoretical justification that diagonal avoids the IW prior's variance-correlation coupling pathology in d=50.

**ADR-007 (Normalization)**: Pending EDA is the right call. Default to raw. Run the sensitivity check as planned.

### Changes to Config

```yaml
portfolio:
  weight_concentration_prior: 1.0 # was 0.1 — see E[K] analysis
  n_init: 5 # was 3 — more restarts for VI stability
```

### Changes to Implementation

```python
# In GMMFitter.__init__ or fit_firm:
# Compute global hyperparameters ONCE from pooled data
self.global_mean = np.mean(X_pool, axis=0)
self.global_var = np.var(X_pool, axis=0)

# Pass to BayesianGaussianMixture:
bgm = BayesianGaussianMixture(
    mean_prior=self.global_mean,           # NOT mean(X_firm)
    covariance_prior=self.global_var,       # NOT empirical_cov(X_firm)
    degrees_of_freedom_prior=52,            # NOT 50
    weight_concentration_prior=1.0,         # NOT 0.1
    # ... rest unchanged
)
```

### New Workflow Steps (Add to Phase 2)

1. **Prior predictive simulation** (500 draws) before fitting any real data
2. **Automated PPC scores** computed for every firm after fitting
3. **Sensitivity sweep** on 200–500 stratified subsample
4. **MCMC validation** on ~500 highest-difficulty firms (optional but recommended)
5. **BC ranking validation**: compare Gaussian BC to Monte Carlo BC on ~200 firm pairs

### Inference Path

**Primary: sklearn VI** (BayesianGaussianMixture) for all 15,814 firms. This is computationally tractable and gives adequate point estimates for BC.

**Secondary: Gibbs sampling** (custom Normal-Gamma conjugate sampler) on the ~500 most difficult firms as a validation check. If VI and Gibbs agree (r > 0.90 on parameter estimates), VI is validated for the entire dataset.

**Not recommended: Stan/NUTS** — label switching is a fundamental structural problem for mixture models in HMC, and Stan cannot sample discrete latent variables.

---

## Appendix: Supporting Technical Details

### A.1 NIW Prior Mathematical Structure

The Normal-Inverse-Wishart prior over (µ, Σ) is parameterized by:

```
NIW(µ₀, κ₀, ν₀, Λ₀)

Σ_k  ~ InverseWishart(Λ₀, ν₀)
µ_k | Σ_k ~ Normal(µ₀, Σ_k / κ₀)
```

The conjugate posterior after observing n_k points is:

```
κ_n  = κ₀ + n_k
ν_n  = ν₀ + n_k
µ_n  = (κ₀ · µ₀ + n_k · x̄_k) / (κ₀ + n_k)
Λ_n  = Λ₀ + S_k + (κ₀ · n_k)/(κ₀ + n_k) · (x̄_k - µ₀)(x̄_k - µ₀)ᵀ
```

For diagonal covariance, this reduces to d independent Normal-Gamma updates — one per dimension.

### A.2 The IW Prior's Three Known Problems in High Dimensions

1. **Single ν₀ controls all variance components**: uniform shrinkage across all 50 dimensions, even if dimensions have very different natural scales
2. **Variance-correlation coupling**: high-variance dimensions are biased toward high absolute correlations (physically unmotivated in UMAP space)
3. **Concentration effect**: even with ν₀ = d+1, off-diagonal correlations concentrate near zero as d grows

Diagonal covariance eliminates all three problems by decomposing into independent per-dimension Gamma priors.

### A.3 Recommended sklearn Parameter Reference

| Parameter                         | Small firm (n<50)     | Large firm (n>200)    |
| --------------------------------- | --------------------- | --------------------- |
| `n_components`                    | 10                    | 10                    |
| `covariance_type`                 | `'diag'`              | `'diag'`              |
| `weight_concentration_prior_type` | `'dirichlet_process'` | `'dirichlet_process'` |
| `weight_concentration_prior`      | 1.0                   | 1.0                   |
| `mean_precision_prior`            | 1.0                   | 1.0                   |
| `mean_prior`                      | `global_mean`         | `global_mean`         |
| `degrees_of_freedom_prior`        | 52                    | 52                    |
| `covariance_prior`                | `global_var`          | `global_var`          |
| `reg_covar`                       | 1e-4                  | 1e-6                  |
| `n_init`                          | 5                     | 5                     |

### A.4 Key Citations

- Murphy, K.P. (2007). "Conjugate Bayesian analysis of the Gaussian distribution." UBC Tech Report.
- Gelman, A. et al. (2013). _Bayesian Data Analysis, 3rd Edition._ Chapman & Hall. Chapters 3, 5, 6.
- Bishop, C.M. (2006). _Pattern Recognition and Machine Learning._ Springer. Chapter 10 (Variational GMM).
- Chung, Y., Gelman, A. et al. (2015). "Weakly Informative Prior for Point Estimation of Covariance Matrices in Hierarchical Models." JEBS.
- Diatchkova et al. (2026). "Design-Conditional Prior Elicitation for Dirichlet Process Mixtures." arxiv 2602.06301.
- Teh, Y.W. (2010). "Dirichlet Process." Encyclopedia of Machine Learning.
- Gelman et al. (2020). "Bayesian Workflow." Columbia University preprint.
- Gabry et al. (2019). "Visualization in Bayesian Workflow." JRSS-A.
- Kallioinen et al. (2024). "Detecting and Diagnosing Prior and Likelihood Sensitivity." Statistics and Computing.
- Yao et al. (2018). "Yes, but Did It Work? Evaluating Variational Inference." ICML.
- McInnes, Healy & Melville (2018). "UMAP: Uniform Manifold Approximation and Projection." arxiv 1802.03426.
- Blei & Jordan (2006). "Variational Inference for Dirichlet Process Mixtures."
- Morita, Thall, Muller (2008). "Determining the Effective Sample Size of a Parametric Prior." PMC.
- Löffler et al. (2021). "Bayesian Sparse Gaussian Mixture Model in High Dimensions." arxiv 2207.10301.
- Vehtari, Gelman, Gabry (2017). "Practical Bayesian Model Evaluation Using LOO-CV and WAIC." Statistics and Computing.
- Stan User's Guide: Identifying Mixture Models, Label Switching, Posterior Predictive Checks.
- sklearn documentation: BayesianGaussianMixture, Concentration Prior Analysis.
