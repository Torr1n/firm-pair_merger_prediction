# ADR-007: Pre-Normalization Strategy for GMM Input Vectors

**Status**: Accepted (raw)  
**Date**: 2026-04-08 (finalized by EDA sensitivity check on production data)  
**Authors**: Torrin Pataki, Claude Code  
**Reviewers**: Codex (accepted)

## Context

Week 1 produced 50D patent vectors via UMAP reduction of 1536D concatenated embeddings (768D title+abstract + 768D citations). These vectors are the input to per-firm GMM fitting. The question is whether to normalize these vectors before GMM fitting, and if so, how.

This ADR is unique among the Week 2 ADRs: **the final decision is deferred to EDA evidence.** The normalization sensitivity check is a Codex-committed deliverable from Week 1 and must be completed with empirical data before the decision is finalized.

### What We Know from Week 1

- **PatentSBERTa embeddings are NOT L2-normalized**: In 768D, title+abstract L2 norms have mean=6.79, std=0.14. Citation norms are bimodal (mean pooling effect, r=-0.317 with citation count).
- **UMAP transforms the scale**: UMAP with cosine metric first L2-normalizes internally, then learns a low-dimensional embedding. The 50D output has its own scale characteristics that we have NOT yet measured (blocked on production results).
- **GMMs are scale-sensitive**: The EM algorithm computes Mahalanobis distances. If dimensions have vastly different scales, dimensions with larger variance will dominate cluster assignment. Diagonal covariance (ADR-006) mitigates this somewhat (each dimension gets its own variance), but extreme scale differences can still cause numerical issues.

## Options Under Evaluation

### Option A: Raw (No Normalization)

Use 50D vectors as-is from UMAP output.

| Aspect | Assessment |
|--------|-----------|
| **Preserves UMAP structure** | Fully — UMAP's learned manifold structure is untouched |
| **Scale balance** | Unknown — depends on UMAP output dimension scales |
| **Magnitude information** | Preserved — if UMAP encodes meaningful magnitude differences, these inform GMM clustering |
| **Risk** | If dimensions have wildly different scales, GMM fitting may be numerically unstable or dominated by high-variance dimensions |

### Option B: L2 Normalization

Normalize each 50D vector to unit length: `x_norm = x / ||x||₂`

| Aspect | Assessment |
|--------|-----------|
| **Preserves UMAP structure** | Partially — projects onto unit hypersphere; relative angles preserved, magnitudes destroyed |
| **Scale balance** | Yes — all vectors have ||x||=1 |
| **Magnitude information** | Destroyed — a patent "close to the origin" (low confidence/mixed signal) becomes indistinguishable from one "far from origin" (strong signal) |
| **Risk** | Hyperspherical projection may distort cluster shapes; diagonal covariance on a hypersphere is a poor fit |

### Option C: Z-Score Standardization

Per-dimension standardization: `x_std = (x - μ_dim) / σ_dim` using global (all patents) mean and std per dimension.

| Aspect | Assessment |
|--------|-----------|
| **Preserves UMAP structure** | Approximately — linear rescaling per dimension doesn't change topology, but changes relative scale relationships |
| **Scale balance** | Yes — each dimension has mean=0, std=1 |
| **Magnitude information** | Rescaled but not destroyed — relative magnitudes within each dimension are preserved |
| **Risk** | If UMAP intentionally produced different scales per dimension (encoding importance), z-score destroys that signal |

## Decision: Raw (Confirmed by EDA)

**Default to raw (Option A)** unless the EDA reveals a problem. Rationale:

1. **UMAP with cosine metric already normalizes internally.** The 50D output is already a learned transformation that accounts for the original scale of the 1536D input. Adding another normalization step risks double-correction.

2. **Diagonal covariance per-dimension already adapts to scale.** Each GMM component estimates per-dimension variance independently. If dimension 3 has range [-2, 2] and dimension 17 has range [-10, 10], the diagonal covariance will capture this — it's literally what diagonal covariance does.

3. **Parsimony.** Fewer transformations = fewer places for subtle bugs. The raw approach has the simplest pipeline.

4. **If normalization is needed, z-score is preferred over L2.** Z-score preserves within-dimension relative magnitudes and doesn't project onto a hypersphere. L2 normalization in 50D concentrates all vectors near the surface of a sphere, which is a poor geometry for diagonal-covariance GMMs.

## EDA Sensitivity Check (Required Before Finalizing)

This is a Codex-committed deliverable. The EDA must perform:

### 1. Dimension Scale Analysis
- Compute per-dimension mean, std, min, max, range of the 50D vectors
- If all dimensions have comparable scales (std within 2x of each other), raw is well-justified
- If some dimensions have 10x+ larger variance, z-score may be needed

### 2. Inter-Dimension Correlation
- Compute the 50×50 Pearson correlation matrix
- If dimensions are weakly correlated (|r| < 0.3 for most pairs), diagonal covariance is well-justified
- If strong correlations exist, this also informs ADR-006

### 3. Normalization Sensitivity on Subsample
- Select ~50 mid-sized firms (100-500 patents)
- Fit Bayesian GMM under all three normalizations (raw, L2, z-score)
- Compare:
  - **BIC scores** (lower is better)
  - **Silhouette scores** on GMM cluster assignments (higher is better)
  - **Effective K** (stability across normalizations)
  - **Cluster assignment stability** (do the same patents end up in the same clusters?)
- If all three normalizations produce similar results, raw is preferred (simplest)
- If one normalization produces significantly better BIC/silhouette, adopt it

### 4. Visual Inspection
- For 5 firms, plot 2D PCA projection of patent vectors colored by GMM cluster assignment under each normalization
- Do the clusters look visually sensible? Does normalization change the story?

## EDA Results (Production Data)

The sensitivity check was run on 50 mid-sized firms (100-500 patents):

| Normalization | Silhouette | Effective K | Convergence | Cross-norm K agreement |
|:-------------:|:----------:|:-----------:|:-----------:|:---------------------:|
| Raw | 0.586 | 8.4 | 100% | — |
| L2 | 0.585 | 8.2 | 100% | 90% within ±1 vs raw |
| Z-score | 0.587 | 8.4 | 100% | 96% within ±1 vs raw |

**All three normalizations produce nearly identical results** — silhouette scores within 0.002, effective K within 0.2, and 90-96% K agreement within ±1. Dimension scales vary 16x (std range [0.14, 2.32]), which exceeds the "within 2x" threshold, but diagonal covariance handles per-dimension scale by construction — making explicit normalization redundant.

**Decision: Raw.** Per the decision criteria: "If all three normalizations produce similar results, raw is preferred (simplest)."

## Alternatives Considered

| Alternative | Why Not Default |
|-------------|----------------|
| **Whitening (PCA + scaling)** | Full decorrelation. Overkill if dimensions are already weakly correlated (expected from UMAP). Adds a PCA step that changes the coordinate system — harder to interpret GMM means in original UMAP space. |
| **Robust scaling (IQR-based)** | More robust to outliers than z-score, but adds complexity. If outliers are a problem, they should be investigated, not masked by scaling. |
| **Per-firm normalization** | Normalize vectors within each firm independently. Destroys cross-firm comparability — a firm's patent at [1, 2, 3] should mean the same thing regardless of what other patents the firm has. |

## Consequences

- The normalization choice is recorded in `config.yaml` as `portfolio.normalization` (options: `"raw"`, `"l2"`, `"zscore"`).
- If z-score is selected, the global mean and std vectors (shape (50,)) must be computed once on the full dataset and stored (not recomputed per firm). These become part of the pipeline's reproducibility artifacts.
- The serialized GMM parameters are always in the normalized space (if normalization is applied). Week 3 BC computation must use the same normalization.
- EDA sensitivity check is complete. Decision finalized as raw.
- **Revisit trigger**: If Week 3 BC comparisons show poor M&A prediction accuracy and normalization is hypothesized as a contributing factor.
