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

Of the top-100 BC pairs, **2 share more than 10%** of their patents (min-normalized) and **2 share more than 25%**. Median shared-patent count is **0**; mean Jaccard similarity is **0.0135**. This indicates that BC is partially reflecting existing co-assignment structure (joint ventures, subsidiaries missed by the containment-based dedup, or long-running technical collaborations across firm boundaries).

Downstream M&A prediction regressions should consider including a shared-patent count (`n_shared`) or Jaccard similarity (`jaccard`) as a control covariate. This isolates the pure technological-similarity signal from already-existing structural ties — otherwise top-BC coefficients may be capturing a "firms that already collaborate" effect rather than a "firms whose technologies complement each other" effect.

## Full Results

See `output/kmax_sweep/coassignment_audit.parquet` (not committed; shipped in the handoff bundle).
Columns: `rank, gvkey_a, gvkey_b, bc, n_patents_a, n_patents_b, n_shared, jaccard, overlap_fraction`.
