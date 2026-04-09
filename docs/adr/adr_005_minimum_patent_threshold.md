# ADR-005: Minimum Patent Threshold and Firm Tiering

**Status**: Accepted  
**Date**: 2026-04-08 (validated by EDA on production data)  
**Authors**: Torrin Pataki, Claude Code  
**Reviewers**: Codex (accepted)

## Context

The firm-size distribution is extremely skewed: 50.8% of the 15,814 firms in v3 have fewer than 10 patents. Fitting a GMM in 50D requires sufficient data — the parameter count per component (even with diagonal covariance) is 101 (50 means + 50 variances + 1 weight). Firms with very few patents cannot support meaningful distributional estimation.

We need to decide:
1. What is the minimum number of patents to include a firm in the analysis?
2. How should we handle firms that have enough patents for a simple representation but not enough for a multi-component GMM?
3. How do these thresholds affect the coverage of our analysis (firms and patents)?

### Statistical Foundation

For a single Gaussian component with diagonal covariance in 50D:
- **Parameters per component**: 50 (mean) + 50 (variance) + 1 (weight) = 101
- **Rule of thumb**: 5-10x parameters for reliable estimation = 500-1,000 data points
- **With Bayesian regularization**: The prior stabilizes estimates with far fewer data points, but the estimates become increasingly dominated by the prior rather than the data

For multi-component GMMs (K>1), the data requirement scales linearly with K. A K=3 model needs ~3x the data of K=1.

### Empirical Evidence (Pre-ADR Experiments)

Bayesian GMM fitting experiments on synthetic 50D data with 2 true clusters:

| n (patents) | Effective K | Behavior |
|-------------|-------------|----------|
| 5 | 5 (= n) | Every point becomes its own component — severe overfitting |
| 10 | 10 (= K_max) | Still overfitting — cannot distinguish structure from noise |
| 20 | 10 (= K_max) | Bayesian prior insufficient to regularize at this ratio |
| 30 | 10 | Largest weights beginning to concentrate (0.40, 0.22) but no pruning |
| 50 | 6 | Starting to prune; top 2 weights = [0.50, 0.43] — approaching correct K |
| 100 | 2 | Correct K identified; weights ≈ [0.50, 0.48] |

**Critical finding**: In 50D with diagonal covariance, Bayesian GMM cannot reliably determine K until n ≈ 50. Below n=50, the model either overfits (one component per point) or fails to prune spurious components.

Single Gaussian (K=1) fitting works reliably even at n=3 due to the trivial structure (just compute mean and variance, regularized by prior).

## Decision

### Three-Tier Firm Classification

| Tier | Patent Count | Treatment | Rationale |
|------|-------------|-----------|-----------|
| **Exclude** | n < 5 | Not included in analysis | Fewer than 5 points in 50D cannot produce even a regularized single-Gaussian estimate that is meaningfully different from the prior. These firms have negligible patent presence. |
| **Single Gaussian** | 5 ≤ n ≤ 49 | K=1, diagonal covariance | Sufficient for a regularized mean + variance estimate. The firm is represented as a single technology focus area. Multi-component fitting would overfit (see empirical evidence above). |
| **Full GMM** | n ≥ 50 | Bayesian GMM with K_max per ADR-004 (default 15) | Sufficient data for the Bayesian GMM to reliably prune and determine effective K. |

### Why n=5 as the Exclusion Boundary?

- With n < 5 in 50D, the sample covariance matrix is rank-deficient even for a single component. The estimate is entirely regularization (prior), not data.
- 5 points provide a minimally meaningful cluster: a centroid with some (heavily regularized) spread estimate.
- Firms with <5 patents are marginal in the technology landscape and unlikely to be M&A targets or acquirers based on patent portfolio overlap.

### Why n=50 as the GMM Boundary (Not n=20)?

The bootstrap prompt suggested n=20. Our synthetic experiments show this is too aggressive:

- At n=20 in 50D, Bayesian GMM with `weight_concentration_prior=0.1` retains all 10 components even when only 2 true clusters exist.
- At n=50, the model begins pruning correctly (effective K=6 → approaching true K=2).
- At n=100, K selection is reliable (effective K=2, correct).

**We set the boundary at n=50 as a conservative threshold.** Firms with 20-49 patents get a well-estimated single Gaussian. Firms with 50+ patents get a multi-component GMM that the Bayesian prior can actually regularize.

### EDA Validation (Production Data)

The tier thresholds were validated on 1,447,673 production patent vectors across 15,696 firms:

| Tier | Firms | % of Firms | Unique Patents | % of Patents |
|------|:-----:|:----------:|:--------------:|:------------:|
| Exclude (<5) | 7,747 | 49.4% | 14,938 | ~1.0% |
| Single Gaussian (5-49) | 6,304 | 40.2% | 91,498 | ~6.3% |
| Full GMM (50+) | 1,645 | 10.5% | 1,343,476 | 92.8% |

**The tiers are well-justified**: excluding half the firms loses only ~1% of patents. The 1,645 GMM-tier firms hold 93% of all patents — these are where modeling quality matters most. The median firm has exactly 5 patents (right at the exclusion boundary).

### Single Gaussian Implementation

For the single-Gaussian tier, use scikit-learn's `GaussianMixture` with `n_components=1`:

```python
from sklearn.mixture import GaussianMixture

model = GaussianMixture(
    n_components=1,
    covariance_type="diag",
    max_iter=200,
    random_state=42,
    reg_covar=1e-6,
)
model.fit(firm_vectors)
# Output: means_ shape (1, 50), covariances_ shape (1, 50), weights_ = [1.0]
```

This produces the same output schema as the multi-component GMM (means, covariances, weights), just with K=1. Downstream BC computation treats it uniformly.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| **Exclude <10, GMM for 10+** | n=10 is far too few for GMM in 50D (see experiments). Would produce unreliable distributional estimates for ~50% of firms. |
| **Exclude <5, GMM for 5+** | Bayesian GMM severely overfits at n=5-49. Single Gaussian is more honest. |
| **Exclude <20, GMM for 20+** (bootstrap prompt suggestion) | n=20 in 50D cannot prune correctly (experiments show all 10 components retained). Wastes computational effort on unreliable fits. |
| **No exclusion — represent all firms** | A firm with 1-4 patents in 50D produces a degenerate distribution. Including it adds noise to downstream pairwise comparison. |
| **PCA pre-reduction for small firms** | Reduce 50D → 10D for firms with few patents, then fit GMM. Adds complexity, violates the principle that all firms should be compared in the same feature space. |
| **Pooled covariance for small firms** | Use the global covariance from all patents as the covariance for small-firm single Gaussians. Reasonable but adds a dependency on global statistics and makes the single-firm representation less self-contained. Could be a future refinement. |

## Consequences

- Firms with <5 patents are excluded from the `firm_gmm_parameters.parquet` output. They are flagged in a separate exclusion log with their gvkey and patent count.
- Firms with 5-49 patents get K=1, diagonal covariance. Their GMM parameters are identical in schema to multi-component firms (means shape (1, 50), covariances shape (1, 50), weights = [1.0]).
- Firms with 50+ patents get full Bayesian GMM (ADR-004).
- The `firm_gmm_parameters.parquet` schema includes a `tier` column ("single_gaussian" or "gmm") for downstream analysis.
- Week 3 BC computation does not need special handling — a single Gaussian is a valid GMM with K=1.
- **Revisit trigger**: If the EDA shows that real patent vectors have stronger cluster structure than synthetic data (enabling correct K selection at lower n), the n=50 GMM threshold can be lowered to 30 or 20.
