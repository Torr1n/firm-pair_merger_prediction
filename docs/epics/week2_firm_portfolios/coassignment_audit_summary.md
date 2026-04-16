# Co-assignment Audit Summary (Top-100 BC Pairs, K_max=15)

**Date**: 2026-04-15 (pre-handover)
**Source**: `output/kmax_sweep/corrected/output/kmax_sweep/bc_matrix_all_k15_dedup_linear.npz`
**Script**: `scripts/coassignment_audit.py`

## Aggregate Statistics

| Metric | Value |
|---|---|
| Median shared patent count | 0 |
| Mean Jaccard similarity | 0.0135 |
| Pairs with >10% overlap (min-normalized) | 2 / 100 |
| Pairs with >25% overlap (min-normalized) | 2 / 100 |
| Pairs with 0 shared patents | 98 / 100 |
| Pairs where one firm has no patents in gvkey_map | 0 / 100 |

## Interpretation

**The dedup rule worked well.** Of the top-100 BC pairs, **98 share zero patents** and only **2 exceed 10% overlap** (both clear parent-subsidiary dedup misses just below the 0.95 containment threshold: rank 37 `060888` + `PRIV_OBLONGINDUSTRIES` at 94% overlap, and rank 20 `063083` + `PRIV_ENDOLOGIX` at 75%). Mean Jaccard similarity is **0.0135** — essentially zero. BC is substantially independent of co-assignment structure for the top-tier pairs; the signal is genuinely distributional, not a rediscovery of existing joint ventures.

**Recommendation for downstream regressions.** Include a shared-patent count (`n_shared`) or Jaccard similarity (`jaccard`) from the audit parquet as a control covariate. The effect size should be small given the audit numbers, but it is defensive against the 2 outliers and against the long tail of <top-100 pairs we did not audit. This isolates the pure technological-similarity signal from already-existing structural ties.

## Full Results

See `output/kmax_sweep/coassignment_audit.parquet` (not committed; shipped in the handoff bundle).
Columns: `rank, gvkey_a, gvkey_b, bc, n_patents_a, n_patents_b, n_shared, jaccard, overlap_fraction`.
