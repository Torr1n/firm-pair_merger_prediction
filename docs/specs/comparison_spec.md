# Comparison Pipeline — BhattacharyyaComputer Specification

**Status**: Reviewed — revisions in progress (Codex 2026-04-14); not yet approved for implementation
**Date**: 2026-04-14 (revised 2026-04-15 for Codex review findings)
**Authors**: Torrin Pataki, Claude Code
**Reviewers**: Codex (first pass returned 5 findings; second pass pending after these revisions)
**ADR Dependencies**: ADR-004 (K_max=15 production), ADR-006 (diagonal covariance)
**Scope**: Bhattacharyya Coefficient computation only. Directional complementarity (asymmetric "gap-filling") is specified separately in `complementarity_spec.md` after `adr_008_directional_complementarity_metric.md` is approved.

---

## Overview

This spec defines `BhattacharyyaComputer`, the production module responsible for Step 3a of the pipeline: pairwise Bhattacharyya Coefficients (BC) between every firm-pair's GMM portfolio.

**Pipeline flow**:
```
firm_gmm_parameters_k15.parquet ──→ BhattacharyyaComputer.compute_bc_matrix() ──→ bc_matrix_k15.npz
                                                          │
                                                          ↓
                                              Week 3 hypothesis testing
```

The implementation lifts validated logic from `scripts/recompute_bc_corrected.py:66-200` and `scripts/run_kmax_sweep.py:434-470` into a tested, contract-driven module under `src/comparison/bhattacharyya.py`.

**Critical correctness note.** The formula uses **linear** mixing weights πᵢπⱼ, NOT the √-weighted variant `√(πᵢ·πⱼ)`. The √-weighted variant is mathematically an upper bound that exceeds 1.0 for multi-component mixtures (max observed: 5.39 in the original sweep). Re-introducing it caused the original K_max=15→20 ranking instability bug. See `bc_mixture()` in `scripts/run_kmax_sweep.py:473` (do not use) versus `bc_mixture_linear()` in `scripts/recompute_bc_corrected.py:66` (use this).

---

## Module: BhattacharyyaComputer

**File**: `src/comparison/bhattacharyya.py`

Computes pairwise BC matrices over a fitted GMM dataset. Stateless beyond its config; safe to instantiate per-run.

### Mathematical contract

For two firm GMMs with diagonal covariance:

```
GMM_A = Σᵢ πᵢᴬ · N(μᵢᴬ, diag(σ²ᵢᴬ))     i = 1..K_A
GMM_B = Σⱼ πⱼᴮ · N(μⱼᴮ, diag(σ²ⱼᴮ))     j = 1..K_B
```

The mixture-level BC (linear-weighted, bounded in [0, 1]):

```
BC(A, B) = Σᵢ Σⱼ πᵢᴬ · πⱼᴮ · BC(Nᵢᴬ, Nⱼᴮ)
```

The component-pair BC under diagonal covariance (closed form):

```
D_B(Nᵢ, Nⱼ) = (1/8) · Σ_d (μᵢ_d - μⱼ_d)² / σ̄²_d  +  (1/2) · Σ_d ln(σ̄²_d / √(σ²ᵢ_d · σ²ⱼ_d))
where  σ̄²_d = (σ²ᵢ_d + σ²ⱼ_d) / 2

BC(Nᵢ, Nⱼ) = exp(-D_B(Nᵢ, Nⱼ))
```

**Properties** (must hold under tests):
- Bounded: 0 ≤ BC(A, B) ≤ 1 for all valid GMM inputs (Cauchy-Schwarz on linear weights)
- Symmetric: BC(A, B) = BC(B, A)
- Self-similarity: BC(A, A) ≈ 1.0 (within float64 numerical tolerance)
- K-invariant in expectation: BC values do not systematically inflate with K (the original √-weighted bug failed this property)

### Interface

```python
class BhattacharyyaComputer:
    def __init__(self, config: dict):
        """
        Args:
            config: parsed config dict; reads from config['comparison'] section.
                Required keys:
                  - output_dir: str — directory for BC matrix output
                  - sg_block_cache: bool — whether to precompute the SG-vs-SG block
                  - sg_block_chunk_size: int — chunking for SG block computation (default: 200)
        """

    def compute_bc(self, gmm_a: GMMResult, gmm_b: GMMResult) -> float:
        """Compute BC between two firm GMMs.

        Args:
            gmm_a, gmm_b: GMMResult objects (per firm_portfolio_spec.md). Must have
                the same covariance_type ('diag' for production) AND the same
                dedup_source (the deduplication_decisions.csv file they came from).

        Returns:
            float in [0, 1]. Self-comparison returns ≈ 1.0.

        Raises:
            ValueError: if covariance_type differs between inputs or is not 'diag',
                if either input is missing a dedup_source field, or if dedup_sources
                differ between the two inputs (silent mixing of dedup policies would
                reintroduce the original 2026-04-12 top-pair duplication bug).
        """

    def compute_sg_block(
        self,
        gmm_results: list[GMMResult],
    ) -> tuple[list[str], np.ndarray]:
        """Precompute the K_max-invariant single-Gaussian-vs-single-Gaussian block.

        Optimization: SG firms (K=1) have weights = [1.0], so their pairwise BC is
        independent of K_max. Computing this block once and caching it accelerates
        subsequent K_max sweeps and reduces compute on the dominant block of the
        full pairwise matrix (typically ~85% of all firms are SG-tier).

        Args:
            gmm_results: full list of GMMResult objects (any tier).

        Returns:
            (sg_gvkeys, sg_block):
                sg_gvkeys: list of gvkeys with tier == 'single_gaussian', sorted
                    string-ascending (matches compute_bc_matrix ordering)
                sg_block: (N_sg, N_sg) float64 BC matrix; diagonal == 1.0

        Raises:
            ValueError: if any input is missing dedup_source, or if inputs reference
                heterogeneous dedup_sources.
        """

    def compute_bc_matrix(
        self,
        gmm_results: list[GMMResult],
        output_path: str,
        sg_block: np.ndarray | None = None,
        sg_gvkeys: list[str] | None = None,
        progress_label: str = "all",
    ) -> tuple[list[str], np.ndarray]:
        """Compute pairwise BC matrix for all firms; write to .npz.

        Row/column order is deterministic: inputs are internally sorted by gvkey
        (string-ascending) before computation. The returned `gvkeys` list and the
        bc_matrix row/column indices correspond to this sorted order. Regression
        tests may therefore compare matrices element-wise without reordering.

        Args:
            gmm_results: full list of GMMResult objects (any tier). All inputs must
                share the same dedup_source (enforced; raises ValueError on mismatch).
            output_path: full path to output .npz file (compressed numpy archive).
            sg_block, sg_gvkeys: optional precomputed cache from compute_sg_block().
                When provided, SG-vs-SG pairs are looked up rather than recomputed.
                Cache gvkey order must match this method's sorted ordering.
            progress_label: label printed in progress messages.

        Returns:
            (gvkeys, bc_matrix):
                gvkeys: list of gvkeys in string-ascending order (row/column order)
                bc_matrix: (N, N) float64 symmetric matrix; diagonal == 1.0

        Raises:
            ValueError: if any input is missing dedup_source, or if inputs reference
                heterogeneous dedup_sources.

        Side effect:
            Writes .npz file containing arrays 'gvkeys' (object array) and
            'bc_matrix' (float64), plus dict-style metadata. See Storage format
            section below for the canonical metadata key schema.
        """

    def load_bc_matrix(self, path: str) -> tuple[list[str], np.ndarray, dict]:
        """Load a previously-computed BC matrix.

        Returns:
            (gvkeys, bc_matrix, metadata): the gvkey order (string-ascending), the
            BC matrix, and the stored metadata dict. See Storage format section
            for the canonical metadata key schema.
        """
```

### Storage format

Output `.npz` schema (numpy compressed archive):

| Key | Type | Shape | Notes |
|---|---|---|---|
| `gvkeys` | object array (string) | (N,) | Row/column order of bc_matrix (string-ascending) |
| `bc_matrix` | float64 | (N, N) | Symmetric; diagonal = 1.0 |
| `metadata` | dict (pickled) | — | See canonical metadata schema below |

**Canonical metadata schema** (required keys, identical between `compute_bc_matrix` and `load_bc_matrix`):

| Key | Type | Meaning |
|---|---|---|
| `k_max` | int | K_max value used during GMM fitting (15 for production) |
| `formula` | str | Always `"linear_mixture"` for this spec; future variants would introduce new identifiers |
| `n_firms` | int | Number of firms in `gvkeys` (= bc_matrix side length) |
| `computed_at_iso` | str | ISO-8601 UTC timestamp of the compute run |
| `sg_cache_hit_count` | int | Number of SG-vs-SG pairs looked up from the cached block (0 if cache disabled) |
| `source_gmm_path` | str | Absolute or repo-relative path to the input `firm_gmm_parameters_k{K_max}.parquet` |
| `dedup_source` | str | Path or identifier of the `deduplication_decisions.csv` referenced by every input GMMResult |

Why `.npz` rather than parquet: the matrix is dense and symmetric (~28M values for 7,485 firms). NumPy compressed archives outperform parquet for dense numerical matrices and load directly into a single ndarray. Existing corrected matrices on S3 already use this format.

---

## Configuration

Added to `src/config/config.yaml`:

```yaml
# --- Comparison (Week 3) ---
comparison:
  output_dir: "output/comparison"
  bc_matrix_template: "output/comparison/bc_matrix_k{k_max}.npz"
  sg_block_cache: true                  # Precompute and reuse SG-vs-SG block
  sg_block_chunk_size: 200              # Chunk size for SG block (memory mgmt)
```

---

## Testing Strategy

`tests/unit/test_bhattacharyya.py`. Tests are written BEFORE implementation (TDD).

### Mathematical correctness

- **test_bc_self_equals_one**: For any GMM A, `compute_bc(A, A)` returns 1.0 within `1e-9` tolerance. Cover single-Gaussian, multi-component, K=10, K=15.
- **test_bc_orthogonal_near_zero**: Two well-separated single-Gaussian GMMs (means 100 std apart) yield BC < 1e-10.
- **test_bc_symmetry**: For random GMM pairs (synthetic), `compute_bc(A, B) == compute_bc(B, A)` within float64 tolerance.
- **test_bc_bounded**: For 100 random synthetic GMMs (varying K, weights, means, variances), all pairwise BC values lie in [0, 1].
- **test_bc_rejects_wrong_covariance**: ValueError if either GMM has covariance_type != 'diag', or if A and B differ.
- **test_bc_rejects_missing_dedup_source**: ValueError if either GMM has no `dedup_source` field.
- **test_bc_rejects_heterogeneous_dedup_source**: ValueError if two inputs to `compute_bc_matrix` reference different `dedup_source` values (guards against the original 2026-04-12 duplicate-firm bug class).
- **test_bc_matrix_row_order_deterministic**: For two random permutations of the same firm list, `compute_bc_matrix` returns identical `gvkeys` lists and element-wise identical matrices (tests the internal sort).

### Regression against validated artifact

- **test_bc_matches_corrected_artifact**: For a 20-firm fixture (curated subset spanning all tiers), recompute BC matrix and assert `np.allclose(refit, stored, rtol=1e-9, atol=1e-12)` against the stored corrected BC matrix at K=15. The fixture covers: 5 single_gaussian, 5 small GMM, 5 mid GMM, 5 mega-firms.

### SG block caching

- **test_sg_block_matches_full_compute**: For a fixture with 10 SG and 5 GMM firms, compare:
  (a) `compute_bc_matrix()` without `sg_block` → full pairwise computation
  (b) `compute_bc_matrix()` with precomputed `sg_block` → cached lookup for SG-SG pairs
  Assert both produce identical matrices within float64 tolerance.
- **test_sg_block_K_invariant**: For the same SG firms fitted at different K_max values (in practice both yield K=1 with weights=[1.0]), the SG block should be identical regardless of K_max source.

### Round-trip serialization

- **test_save_load_round_trip**: Write a small BC matrix via `compute_bc_matrix()`, load via `load_bc_matrix()`, assert exact equality of gvkeys list, bc_matrix array, and metadata dict.
- **test_metadata_present**: Loaded metadata contains all canonical keys (k_max, formula, n_firms, computed_at_iso, sg_cache_hit_count, source_gmm_path, dedup_source).

### Test fixtures

- Synthetic GMMs hand-crafted with known means/variances/weights for property tests.
- Real 20-firm fixture: subset of `output/kmax_sweep/firm_gmm_parameters_k15.parquet` filtered to known-stable firms across tiers, plus the corresponding 20×20 sub-matrix extracted from the corrected stored BC matrix. Both files committed to `tests/fixtures/`.

---

## Week 3 Interface Contract

Week 3 (downstream hypothesis testing by Ananya/Arthur/Amie/Duncan) consumes the BC matrix via:

1. `BhattacharyyaComputer.load_bc_matrix("output/comparison/bc_matrix_k15.npz")` returns `(gvkeys, bc_matrix, metadata)`.
2. `gvkeys[i]` corresponds to row/column `i` of `bc_matrix`. Lookup table: `gvkey_to_idx = {gk: i for i, gk in enumerate(gvkeys)}`.
3. To get the BC between two firms: `bc_matrix[gvkey_to_idx[gv_a], gvkey_to_idx[gv_b]]`.
4. To get top-k partners for a firm: `np.argsort(bc_matrix[gvkey_to_idx[gv]])[::-1][:k]`.
5. The matrix is symmetric and the diagonal is 1.0 (self-comparison).
6. Firms not in `gvkeys` were either excluded (insufficient patents) or deduplicated; they should not be queried.

---

## Out of Scope

- **Directional complementarity** is its own module (`src/comparison/complementarity.py`) and spec (`complementarity_spec.md`), pending `adr_008_directional_complementarity_metric.md` formula decision.
- **Full covariance BC** — only diagonal covariance is supported, matching ADR-006. Adding full-covariance support would extend `compute_bc()` with a covariance_type branch.
- **BC under alternative mixture models** (Student-t, skew-t) — contingent on `gaussian_adequacy_audit.md` (Phase 3B); separate spec if needed.

---

## Implementation References

| Need | Source | Lines |
|---|---|---|
| Component-pair BC formula (diagonal) | `scripts/run_kmax_sweep.py` | 434-470 (`bc_component_matrix`) |
| Linear-weighted mixture BC | `scripts/recompute_bc_corrected.py` | 66-85 (`bc_mixture_linear`) |
| Vectorized SG block computation | `scripts/recompute_bc_corrected.py` | 88-148 (`compute_sg_block_linear`) |
| Full pairwise matrix loop | `scripts/recompute_bc_corrected.py` | 151-195 (`compute_bc_matrix_linear`) |
| Buggy formula (do NOT lift) | `scripts/run_kmax_sweep.py` | 473-489 (`bc_mixture` — √-weighted) |
