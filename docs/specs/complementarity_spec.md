# Directional Technology Complementarity — Specification

**Status**: Draft — pending ADR-008 approval before implementation of `src/comparison/complementarity.py`
**Date**: 2026-04-16
**Authors**: Duncan Harrop, Claude Code
**Scope**: Directional complementarity metric, `scripts/compute_complementarity.py`, and future production module `src/comparison/complementarity.py`
**ADR Dependencies**: ADR-004 (K_max=10 production), ADR-006 (diagonal covariance)

---

## Motivation

The Bhattacharyya Coefficient (BC) measures **technological overlap** between two firms' patent portfolios. It is symmetric — BC(A, B) = BC(B, A) — and captures substitutability: firms that patent in the same clusters score high.

But M&A activity is not just about similarity. Bloom, Schankerman & Van Reenen (2013) show that firms interact through both technological proximity *and* knowledge spillovers. A firm may acquire another precisely because the target fills technological gaps the acquirer lacks. This is a **gap-filling / synergy** motive, and it is directional: the answer to "does B fill A's gaps?" is not the same as "does A fill B's gaps?"

BC cannot capture this. We need a second feature.

---

## The Formula

Each firm's patent portfolio is represented as a probability distribution over K GMM clusters via its mixture weights:

```
p_{A,k}  =  firm A's weight on technology cluster k  (sums to 1 across k)
```

The directional complementarity of B into A — how well B's strengths fill A's weaknesses — is:

```
Comp(A→B) = Σ_k  (1 - p_{A,k}) · p_{B,k}
```

Term by term:
- `(1 - p_{A,k})` is firm A's **gap** in cluster k: where A is weak, this is large
- `p_{B,k}` is firm B's **strength** in cluster k: where B is strong, this is large
- The product is large when B is strong exactly where A is weak
- Summing across all clusters gives the total gap-filling score

**Key property: asymmetry**

```
Comp(A→B) ≠ Comp(B→A)  in general
```

This lets us distinguish acquirer from target — something BC cannot do:

| BC | Comp(A→B) | Interpretation |
|---|---|---|
| High | High | B overlaps A *and* fills A's gaps — strong acquisition candidate |
| Low | High | Capability acquisition: B brings what A lacks (no overlap, pure synergy) |
| High | Low | Near-clone: B overlaps A but adds nothing new |
| Low | Low | Firms are unrelated |

---

## Relation to BC

For the weight-only (mixture-level) BC formula used in this project:

```
BC(A, B) = Σ_k  p_{A,k} · p_{B,k}
```

Note that:

```
Comp(A→B) = Σ_k p_{B,k} - Σ_k p_{A,k} · p_{B,k}
           = 1 - BC(A, B)
```

So when firms share uniform weights, Comp = 1 − BC and the two metrics are perfectly inversely related. But once weights are heterogeneous (which is always the case for GMM-tier firms), directionality emerges: the *distribution* of gaps matters, not just their average depth.

This is why Comp is a complement to BC in the feature set, not a substitute.

---

## Implementation: `scripts/compute_complementarity.py`

This script is the production-ready prototype. It runs end-to-end on the GMM output and saves the complementarity and BC matrices.

### Inputs

| File | Location | Description |
|---|---|---|
| `firm_gmm_parameters_k10.parquet` | `data/` | GMM results per firm. Rows have `gvkey`, `weights`, `n_components`, `n_patents`, `tier` |
| `deduplication_decisions.csv` | `data/` | 464 firms to exclude (subsidiaries, aliases, predecessor records). Has a `dropped` column with gvkey values |

**Always apply the deduplication list first.** Without exclusion, subsidiaries (e.g. Alphabet/Waymo, Qualcomm/Snaptrack) produce spurious high complementarity scores — they are already related, making them false positives in any downstream M&A model.

### Algorithm

**Step 1 — Build the weight matrix P**

Collect the mixture weights for every non-excluded firm into a matrix `P` of shape `(n_firms, K)`:

```
P[i, k]  =  weight of firm i on cluster k
```

Firms with fewer than K components are zero-padded on the right. Each row sums to 1.

**Step 2 — Vectorized complementarity**

```python
one_minus_P = 1 - P               # (n_firms, K)  — each firm's gaps
Comp = one_minus_P @ P.T          # (n_firms, n_firms)
```

`Comp[i, j]` = `Comp(firm_i → firm_j)` = how well firm j fills firm i's gaps.

This is O(n² · K) but expressed as a single matrix multiply — fast and exact.

**Step 3 — BC matrix (for reference)**

```python
BC = P @ P.T                      # (n_firms, n_firms), symmetric
```

Uses the corrected linear-weight formula (not the √-weighted variant, which exceeds 1.0 for multi-component mixtures and was the source of the original ranking instability bug — see `comparison_spec.md`).

**Step 4 — Validate**

Three invariants that must hold:

```python
# 1. Self-complementarity is zero: a firm fills none of its own gaps
assert np.allclose(np.diag(Comp), 0, atol=1e-5)

# 2. All values in [0, 1]
assert Comp.min() >= -1e-5 and Comp.max() <= 1 + 1e-5

# 3. Matrix is genuinely asymmetric (directionality is working)
assert not np.allclose(Comp, Comp.T, atol=1e-4)
```

**Step 5 — Save**

```
output/comparison/complementarity_matrix_k10.npz
```

Stored arrays:

| Key | Shape | Description |
|---|---|---|
| `gvkeys` | `(n_firms,)` | Firm identifiers, order matches matrix rows/columns |
| `comp_matrix` | `(n_firms, n_firms)` | `Comp[i,j]` = how well j fills i's gaps |
| `bc_matrix` | `(n_firms, n_firms)` | Symmetric BC scores for reference |

### Running the script

```bash
source venv/bin/activate
python scripts/compute_complementarity.py
```

Requires `data/firm_gmm_parameters_k10.parquet` and `data/deduplication_decisions.csv`. Output goes to `output/comparison/`.

---

## Spot Checks

The script prints spot checks for five named firms:

```python
DEEP_DIVE_FIRMS = {
    "006066": "IBM",
    "012141": "Intel",
    "024800": "Qualcomm",
    "160329": "Google/Alphabet",
    "020779": "Cisco Systems",
}
```

The IBM↔Intel pair confirms directionality is working:

```
Comp(IBM → Intel): should differ from Comp(Intel → IBM)
```

IBM (diversified portfolio, many clusters) should have lower self-gaps than a focused semiconductor firm, meaning its gap-filling relationship with Intel differs in magnitude from the reverse.

---

## Downstream Usage

Week 3 hypothesis testing will consume both BC and Comp as features in a predictive model for M&A pairs. To load and query:

```python
import numpy as np

data = np.load("output/comparison/complementarity_matrix_k10.npz", allow_pickle=True)
gvkeys = data["gvkeys"].tolist()
comp_matrix = data["comp_matrix"]
bc_matrix = data["bc_matrix"]

gvkey_to_idx = {gk: i for i, gk in enumerate(gvkeys)}

def get_complementarity(acquirer_gvkey, target_gvkey):
    """How well does target fill acquirer's gaps?"""
    i = gvkey_to_idx[str(acquirer_gvkey)]
    j = gvkey_to_idx[str(target_gvkey)]
    return float(comp_matrix[i, j])

def get_bc(gvkey_a, gvkey_b):
    i = gvkey_to_idx[str(gvkey_a)]
    j = gvkey_to_idx[str(gvkey_b)]
    return float(bc_matrix[i, j])
```

Comp is an asymmetric feature: `get_complementarity(acquirer, target)` is not the same as `get_complementarity(target, acquirer)`. Use both directions as separate features if the model treats pair ordering as meaningful.

---

## Future Production Module

When ADR-008 is approved, `scripts/compute_complementarity.py` will be lifted into `src/comparison/complementarity.py` following the same pattern as `BhattacharyyaComputer` in `comparison_spec.md`:

- Config-driven paths and K_max
- `ComplementarityComputer` class with `compute_comp_matrix()` and `load_comp_matrix()` methods
- Full test suite under `tests/unit/test_complementarity.py`
- Canonical `.npz` metadata schema (K_max, formula, n_firms, computed_at_iso, dedup_source)

The script is intentionally kept as a standalone prototype until the formula decision is finalized and Codex has reviewed the ADR.

---

## Out of Scope

- **Full covariance complementarity** — the formula operates on mixture weights only; it does not use means or covariances. This is by design: it measures *where* each firm concentrates probability mass, not the spread within each cluster.
- **Normalized complementarity** — dividing by a baseline (e.g. uniform weights) would remove scale dependence but change the economic interpretation. Not planned.
- **Synthetic portfolio matching** (A + B ≈ C) — this is the Step 4 extension described in `methodology.md` and is a separate workstream.

---

## References

- Bloom, N., Schankerman, M., & Van Reenen, J. (2013). Identifying technology spillovers and product market rivalry. *Econometrica*, 81(4), 1347–1393.
- Bena, J., & Li, K. (2014). Corporate innovations and mergers and acquisitions. *Journal of Finance*, 69(5), 1923–1960.
- `methodology.md` — original pipeline description (Arthur Khamkhosy, email to Jan Bena)
- `docs/specs/comparison_spec.md` — BC module spec; complementarity is the asymmetric complement to BC
