# ADR-006: GMM Covariance Type

**Status**: Accepted  
**Date**: 2026-04-08 (validated by covariance diagnostic on production data)  
**Authors**: Torrin Pataki, Claude Code  
**Reviewers**: Codex (accepted)

## Context

Each GMM component models a "technology area" as a Gaussian distribution in 50D space. The covariance matrix captures how patents spread within each technology area. The choice of covariance parameterization directly affects:

1. **Parameter count** per component (and therefore data requirements)
2. **Modeling capacity** (whether the model can capture correlations between dimensions)
3. **Computational cost** of fitting and of downstream Bhattacharyya Coefficient computation
4. **Numerical stability** (full covariance in high dimensions is prone to singularity)

scikit-learn supports four covariance types: `full`, `tied`, `diag`, and `spherical`.

### Parameter Counts Per Component (50D)

| Covariance Type | Parameters | Formula | Per-Component Total (+ 50 mean + 1 weight) |
|----------------|------------|---------|---------------------------------------------|
| Full | 1,275 | d(d+1)/2 | 1,326 |
| Diagonal | 50 | d | 101 |
| Spherical | 1 | 1 | 52 |
| Tied (shared) | 1,275 | d(d+1)/2 (shared across K) | 52 per component + 1,275 shared |

For a K=5 model:
- **Full**: 5 × 1,326 = 6,630 free parameters
- **Diagonal**: 5 × 101 = 505 free parameters
- **Spherical**: 5 × 52 = 260 free parameters

## Decision

### Covariance Type: Diagonal (`"diag"`)

All GMM components use diagonal covariance — each dimension has its own variance, but cross-dimension correlations are assumed zero.

### Rationale

**1. Data sufficiency.** The 10x rule of thumb requires ~1,010 data points per diagonal component vs ~13,260 per full-covariance component. With our firm-size distribution (many firms with 50-500 patents), diagonal is fittable; full covariance would require firms with thousands of patents per component — limiting multi-component GMMs to a handful of mega-firms.

| Cov Type | Min Patents for K=1 | Min Patents for K=3 | Min Patents for K=5 |
|----------|--------------------|--------------------|---------------------|
| Full | ~13,260 | ~39,780 | ~66,300 |
| Diagonal | ~1,010 | ~3,030 | ~5,050 |

**2. Diagonal is the best practical choice under data/compute constraints.** EDA on production vectors revealed that UMAP dimensions are in fact moderately to strongly correlated (mean |r|=0.33, 48% of pairs |r|>0.3, top pairs |r|=0.95). The "weakly correlated" assumption was empirically falsified. However, a direct covariance diagnostic comparing diagonal, full, and tied covariance on 20 firms (n=200-1000) showed that **full covariance produces 62% worse PPC scores than diagonal, and tied is 20% worse.** The reason: full covariance in 50D has 1,275 parameters per component, forcing K=2 for most firms, while diagonal allows K=8-10 with far better density coverage. The parameter budget trade-off decisively favors more diagonal components over fewer full-covariance components.

**3. Numerical stability.** Full covariance matrices in 50D are prone to singularity when the number of data points per component is comparable to the dimensionality. scikit-learn's `reg_covar` parameter adds a small diagonal regularization, but this is a patch — diagonal covariance eliminates the problem structurally.

**4. BC closed-form availability.** The Bhattacharyya Coefficient between two multivariate Gaussians has a closed form for both diagonal and full covariance. Diagonal is computationally cheaper:

For diagonal Gaussians with means μ₁, μ₂ and diagonal covariances σ₁², σ₂²:

```
BC = exp(-D_B)

D_B = (1/8)(μ₁-μ₂)ᵀ Σ⁻¹ (μ₁-μ₂) + (1/2) ln(det(Σ) / √(det(Σ₁)·det(Σ₂)))

where Σ = (Σ₁ + Σ₂) / 2
```

For diagonal matrices, the determinant and inverse are trivially computed from element-wise operations — no matrix inversion needed. This makes the Week 3 pairwise comparison (15K × 15K firms) significantly faster.

**5. Storage efficiency.** Each component stores 50 variance values (400 bytes as float64) vs 1,275 upper-triangle values (10,200 bytes) for full covariance. Across 15K firms with average K=3, this is ~1.8 MB vs ~460 MB — a 250x difference.

**6. Avoids the Inverse-Wishart prior pathology in high dimensions.** The IW prior has three known problems at d=50 (Murphy 2007, Chung et al. 2015): (a) a single ν₀ controls uniform shrinkage across all 50 dimensions, even if dimensions have different natural scales; (b) variance-correlation coupling — high-variance dimensions are biased toward high absolute correlations, which is physically unmotivated in UMAP embedding space; (c) off-diagonal correlations concentrate near zero as d grows, making the prior increasingly informative about correlation structure regardless of ν₀. The diagonal covariance assumption eliminates all three problems by decomposing the IW into 50 independent Gamma priors on per-dimension precisions. This is not merely a computational convenience — it is a theoretically superior prior structure for our problem.

### When Full Covariance Would Be Better

Full covariance captures inter-dimension correlations. If:
- The 50 UMAP dimensions have strong pairwise correlations (r > 0.3 for many pairs)
- Technology areas within firms have ellipsoidal shapes aligned off-axis in the 50D space
- Diagonal covariance produces visually poor cluster assignments on large firms

then full covariance may be warranted for a subset of firms. This is a potential Week 4 refinement, not a Week 2 baseline requirement.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| **Full covariance** | 1,326 parameters per component; requires >13K patents per component for reliable estimation; numerically unstable for typical firm sizes; 250x storage overhead. Could be a refinement for mega-firms (>10K patents) in Week 4. |
| **Spherical covariance** | Only 1 parameter per component (uniform variance in all dimensions). Too restrictive — dimensions may have genuinely different variances after UMAP. Loses information. |
| **Tied covariance** | All components share the same covariance matrix. Inappropriate — different technology areas within a firm may have different spreads (a focused area vs. an exploratory one). |
| **Tiered: diagonal for small firms, full for large** | Adds complexity to the pipeline and downstream BC computation (would need to handle mixed covariance types). Violates parsimony. If full covariance is needed, it should be for all firms or none. |

## Consequences

- All GMM components use `covariance_type="diag"` in both `GaussianMixture` and `BayesianGaussianMixture`.
- Covariances are stored as shape (K, 50) arrays in `firm_gmm_parameters.parquet` (not (K, 50, 50)).
- Downstream BC computation uses the diagonal closed-form — no matrix inversion needed.
- The `covariance_type` column in the output parquet is always `"diagonal"` for Week 2. If full covariance is introduced later, it would require a separate ADR and a schema-compatible extension.
- **EDA validation complete**: Inter-dimension correlations are strong (mean |r|=0.33, max |r|=0.95). Despite this, direct covariance diagnostic showed diagonal produces better PPC than full or tied covariance due to parameter budget trade-offs. The original "weakly correlated" revisit trigger has fired and been resolved in favor of keeping diagonal on empirical grounds.
- **Revisit trigger**: If Week 3 BC comparisons produce poor M&A prediction accuracy and a targeted experiment shows full covariance with sufficiently large K (requiring firms with n > 5,000) improves rankings.
