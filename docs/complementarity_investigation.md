# Directional Complementarity — Investigation Log

**Date**: 2026-04-21
**Author**: Duncan Harrop, Claude Code
**Branch**: `feature/directional-complementarity`
**Script**: `scripts/compute_complementarity.py`

---

## Formula Evolution Summary

| Attempt | Formula | Symmetric? | Usable? | Why not |
| --- | --- | --- | --- | --- |
| 1 | `Σ_k (1−p_{A,k}) · p_{B,k}` on per-firm GMM weights | Yes — always `= 1−BC` | No | Algebraic identity; normalised weights make it symmetric by construction |
| 2 | `1 − Σ_i Σ_j p_{A,i}·p_{B,j}·exp(−KL(B_j‖A_i))` on per-firm GMMs | No | No | 99.96% of pairs saturate at 1.0; 50D KL divergence explodes across non-overlapping firm clusters |
| 3 | `Σ_k (1−p_{A,k}) · p_{B,k}` on global K-means weights | Yes — always `= 1−BC` | No | Same algebraic identity applies regardless of clustering approach |
| **Final** | **`Σ_k max(0, p_market_k − p_{A,k}) · p_{B,k}` on global K-means weights** | **No** | **Yes** | **Market-relative gap breaks the normalisation constraint → genuine asymmetry** |

---

## Background

The BC score already measures symmetric technological overlap between two firms. The goal of this investigation was to compute a *directional* complementarity score — Comp(A→B) ≠ Comp(B→A) — that captures whether firm B fills gaps in firm A's patent portfolio. This is motivated by capability-acquisition M&A theory: acquirers often buy targets not because they overlap, but because the target covers technology areas the acquirer lacks.

The proposed formula from CONTEXT.md:
```
Comp(A→B) = Σ_k (1 - p_{A,k}) · p_{B,k}
```

---

## Attempt 1: Weight-only formula on per-firm GMM weights

### Setup
Use the Bayesian GMM mixture weights directly from `firm_gmm_parameters_k10.parquet`. Each firm has up to K=10 weights summing to 1. Build a weight matrix P (7,485 × 10) and compute:

```python
Comp = (1 - P) @ P.T    # intended: directional complementarity
BC   = P @ P.T          # symmetric Bhattacharyya
```

### Issue 1: float32/float64 decode mismatch
Weights were stored as raw bytes (`float64`, 8 bytes per value). The initial script decoded them as `float32` (4 bytes), producing 16 floats per 8-component firm instead of 8. All weights were garbage. Fix: change `dtype=np.float32` → `dtype=np.float64` in `np.frombuffer`.

### Issue 2: Wrong self-complementarity assertion
The validation checked `assert np.allclose(np.diag(Comp), 0)`. This was wrong. For a firm with multiple components (e.g. weights [0.3, 0.7]):

```
Comp(A→A) = (1-0.3)·0.3 + (1-0.7)·0.7 = 0.21 + 0.21 = 0.42
```

The diagonal is only 0 for single-component firms. The assertion was removed.

### Issue 3: Formula is symmetric — mathematical proof

After fixing the decode, the validation reported:
```
Directional (Comp != Comp.T): False
Comp(IBM->Intel): 0.8993   Comp(Intel->IBM): 0.8993
```

**Proof of symmetry:** When both weight vectors sum to 1,

```
Comp(A→B) = Σ_k (1 - p_{A,k}) · p_{B,k}
           = Σ_k p_{B,k}  −  Σ_k p_{A,k} · p_{B,k}
           = 1 − BC(A,B)
```

`BC(A,B)` is symmetric, so `Comp(A→B) = 1 − BC(A,B) = Comp(B→A)` always. The formula **cannot** be asymmetric with normalised weights, regardless of how the clusters are defined.

---

## Attempt 2: KL divergence using GMM means and covariances

### Motivation
The GMM parameters include not just weights but means (K × 50) and covariances (K × 50, diagonal). The KL divergence KL(B_j ‖ A_i) uses A's variance in the denominator — a focused firm A (narrow σ²) penalises distance more, creating genuine asymmetry.

```
Comp(A→B) = 1 − Σ_i Σ_j p_{A,i} · p_{B,j} · exp(−KL(B_j ‖ A_i))

KL(N_j ‖ N_i) = ½ · [ Σ_d σ²_{j,d}/σ²_{i,d}  +  Σ_d (μ_{i,d}−μ_{j,d})²/σ²_{i,d}  −  D  +  Σ_d ln(σ²_{i,d}/σ²_{j,d}) ]
```

This is genuinely asymmetric: swapping A and B changes which variance appears in the denominator.

### Result: saturation
```
Directional (Comp != Comp.T): True  (max = 0.1854)
Fraction of pairs with Comp > 0.999: 0.9996
Comp(IBM->Intel): 1.0000   Comp(Intel->IBM): 1.0000
```

The metric IS directional for 0.04% of pairs — but 99.96% of all pairs saturate at exactly 1.0.

**Root cause:** In 50D UMAP space, per-firm GMM components are extremely localised relative to inter-firm distances. Any cross-firm component pair has enormous KL divergence (driven by the distance term `Σ_d (μ_{i,d}−μ_{j,d})² / σ²_{i,d}`), so `exp(−KL) ≈ 0` for essentially all pairs → coverage ≈ 0 → Comp ≈ 1.

The only pairs that showed variation were firms with BC = 1.0 (near-identical portfolios), which are not useful for M&A prediction. IBM and Intel — two large tech firms — had BC = 0.10, meaning their component means are far apart in 50D space. At D=50, the KL formula amplifies this distance 50-fold.

**Conclusion:** KL-based complementarity on per-firm GMMs is not viable. The 50D geometry is too sparse for this formula.

---

## Attempt 3: Weight-only formula on global K-means clusters

### Motivation
If all firms share a common set of K global technology zones (instead of each firm having its own private clusters), then cluster k means the same technology area for every firm. The formula `Comp(A→B) = Σ_k (1−p_{A,k}) · p_{B,k}` could then be interpreted meaningfully.

### Setup
1. Fit `MiniBatchKMeans(K=50)` on all 1,447,673 patent embeddings → 50 global technology zones
2. Each firm's portfolio = fraction of its patents in each zone
3. Compute Comp and BC

### Result: still symmetric
```
Directional (max |Comp[i,j]−Comp[j,i]|): 0.000000
```

**Same algebraic proof applies.** Regardless of how clusters are defined, when weights sum to 1, `Comp(A→B) = 1 − BC(A,B)`. The formula is symmetric by construction.

However, the saturation improved significantly: only 63.64% of pairs at 1.0 (vs 99.96% for KL-based). The global cluster approach produced more spread in the BC scores because the shared reference frame lets similar firms score higher.

---

## Final approach: Market-relative gaps + global K-means

### Key insight
To break symmetry without changing the fundamental structure of the formula, redefine what "A's gap" means. Instead of `(1 − p_{A,k})`, use how far A is **below the global market average** in each zone:

```
gap_{A,k} = max(0,  p_market_k − p_{A,k})
```

where `p_market_k` = fraction of ALL patents globally in zone k (from the K-means label counts).

Then:
```
Comp(A→B) = Σ_k max(0, p_market_k − p_{A,k}) · p_{B,k}
```

**Why this is genuinely asymmetric:** A's and B's market-relative gaps are different vectors. Switching A and B uses a different gap vector, producing a different result.

**Economic interpretation:** Firm A's "gaps" are the technology zones where it invests less than the market as a whole. Comp(A→B) measures how much B concentrates in those under-served areas. High Comp(A→B) means B fills A's market-relative weaknesses.

### Results
```
comp_dir max |M[i,j]−M[j,i]|: 0.039376   ← genuinely directional
BC + diss symmetric: True                  ← as expected

Comp(IBM→Intel):    0.0021   Comp(Intel→IBM):    0.0061
Comp(IBM→Qualcomm): 0.0103   Comp(Qualcomm→IBM): 0.0156
```

IBM→Intel and Intel→IBM are different. The direction is confirmed.

The raw scale of `comp_dir` is `[0, max(p_market_k)] ≈ [0, 0.041]` because it is bounded by the largest global cluster (4.1% of all patents). The values are small in absolute terms but have meaningful spread and the downstream model will handle scaling.

---

## Output

**File**: `output/comparison/complementarity_matrix_global_k50.npz`

| Array | Shape | Description |
|---|---|---|
| `gvkeys` | (7485,) | Firm identifiers |
| `bc_matrix` | (7485, 7485) | Symmetric BC scores |
| `dissimilarity_matrix` | (7485, 7485) | `1 − BC`, symmetric |
| `comp_dir_matrix` | (7485, 7485) | Market-relative directional comp, asymmetric |
| `p_market` | (50,) | Global patent distribution over 50 zones |
| `k_global` | scalar | K=50 |

**Recommended usage in the downstream model:**
- `bc_matrix[i,j]` — technological similarity (already validated in Week 2)
- `comp_dir_matrix[i,j]` — does j fill i's market-relative gaps? (acquirer = i)
- `comp_dir_matrix[j,i]` — does i fill j's market-relative gaps? (acquirer = j)
- Use both directions as separate features if the model treats pair ordering as meaningful

---

## Lessons

1. **The formula `Comp(A→B) = Σ_k (1−p_{A,k}) · p_{B,k}` is always symmetric when weights sum to 1.** This is an algebraic identity, not a data issue. No clustering approach changes this.

2. **KL divergence is not viable for comparing 50D UMAP-embedded GMMs.** The 50D geometry causes KL to diverge for all cross-firm component pairs. `exp(−KL) ≈ 0` universally.

3. **Asymmetry requires breaking the weight normalisation constraint.** The market-relative gap `max(0, p_market_k − p_{A,k})` is not a normalised probability distribution, so the algebraic symmetry proof no longer applies.

4. **Global clusters are necessary for the formula to be economically meaningful.** Per-firm GMMs have private cluster indices — cluster 1 for IBM is not the same technology as cluster 1 for Intel. The formula only makes sense with a shared reference frame.
