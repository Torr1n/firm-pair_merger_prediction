"""Co-assignment audit for top-100 BC pairs (K_max=15, corrected linear-weighted).

Answers: to what extent does high BC reflect pre-existing patent co-assignment
(joint ventures, subsidiaries missed by dedup, long-running collaborations)?

Output:
  output/kmax_sweep/coassignment_audit.parquet       (not committed; shipped in bundle)
  docs/epics/week2_firm_portfolios/coassignment_audit_summary.md  (committed)

Downstream regressions should include a shared-patent control (n_shared or jaccard)
to isolate pure technological-similarity signal from structural co-assignment ties.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CORRECTED = Path("output/kmax_sweep/corrected/output/kmax_sweep")
BC_PATH = CORRECTED / "bc_matrix_all_k15_dedup_linear.npz"
GVKEY_MAP_PATH = Path("output/week2_inputs/gvkey_map.parquet")
OUT_PARQUET = Path("output/kmax_sweep/coassignment_audit.parquet")
OUT_SUMMARY_MD = Path("docs/epics/week2_firm_portfolios/coassignment_audit_summary.md")

TOP_K = 100


def main() -> None:
    # --- Load corrected BC matrix ---
    print(f"Loading BC matrix from {BC_PATH} ...")
    data = np.load(BC_PATH, allow_pickle=True)
    gvkeys = [str(g) for g in data["gvkeys"]]
    bc = data["bc_matrix"]
    n = len(gvkeys)
    assert bc.shape == (n, n), f"BC matrix shape {bc.shape} != ({n}, {n})"
    print(f"  Loaded {n} firms; BC matrix is {bc.shape}, diagonal mean = {np.diag(bc).mean():.6f}")

    # --- Extract top-K upper-triangle pairs by BC value ---
    iu_row, iu_col = np.triu_indices(n, k=1)
    bc_vals = bc[iu_row, iu_col]
    top_flat = np.argpartition(-bc_vals, TOP_K)[:TOP_K]
    top_flat = top_flat[np.argsort(-bc_vals[top_flat])]
    top_i = iu_row[top_flat]
    top_j = iu_col[top_flat]

    # --- Build firm -> set(patent_id) mapping ---
    print(f"Loading gvkey_map from {GVKEY_MAP_PATH} ...")
    gv_map = pd.read_parquet(GVKEY_MAP_PATH)
    print(f"  {len(gv_map):,} patent-firm assignments across {gv_map['gvkey'].nunique():,} firms")
    firm_patents: dict[str, set[str]] = (
        gv_map.groupby("gvkey")["patent_id"].apply(lambda s: set(s.astype(str))).to_dict()
    )

    # --- Compute shared-patent stats ---
    missing_map_coverage = 0
    rows = []
    for rank, (i, j) in enumerate(zip(top_i, top_j), start=1):
        gv_a, gv_b = gvkeys[i], gvkeys[j]
        pats_a = firm_patents.get(gv_a, set())
        pats_b = firm_patents.get(gv_b, set())
        if not pats_a or not pats_b:
            missing_map_coverage += 1
        shared = pats_a & pats_b
        union = pats_a | pats_b
        rows.append(
            {
                "rank": rank,
                "gvkey_a": gv_a,
                "gvkey_b": gv_b,
                "bc": float(bc[i, j]),
                "n_patents_a": len(pats_a),
                "n_patents_b": len(pats_b),
                "n_shared": len(shared),
                "jaccard": len(shared) / len(union) if union else 0.0,
                "overlap_fraction": (
                    len(shared) / min(len(pats_a), len(pats_b))
                    if pats_a and pats_b
                    else 0.0
                ),
            }
        )

    df = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET} ({len(df)} rows)")

    # --- Aggregate stats ---
    median_shared = int(df["n_shared"].median())
    mean_jaccard = float(df["jaccard"].mean())
    n_ge10 = int((df["overlap_fraction"] > 0.10).sum())
    n_ge25 = int((df["overlap_fraction"] > 0.25).sum())
    n_zero = int((df["n_shared"] == 0).sum())

    # --- Summary markdown (committed) ---
    summary = f"""# Co-assignment Audit Summary (Top-{TOP_K} BC Pairs, K_max=15)

**Date**: 2026-04-15 (pre-handover)
**Source**: `{BC_PATH}`
**Script**: `scripts/coassignment_audit.py`

## Aggregate Statistics

| Metric | Value |
|---|---|
| Median shared patent count | {median_shared} |
| Mean Jaccard similarity | {mean_jaccard:.4f} |
| Pairs with >10% overlap (min-normalized) | {n_ge10} / {TOP_K} |
| Pairs with >25% overlap (min-normalized) | {n_ge25} / {TOP_K} |
| Pairs with 0 shared patents | {n_zero} / {TOP_K} |
| Pairs where one firm has no patents in gvkey_map | {missing_map_coverage} / {TOP_K} |

## Interpretation

Of the top-{TOP_K} BC pairs, **{n_ge10} share more than 10%** of their patents (min-normalized) \
and **{n_ge25} share more than 25%**. Median shared-patent count is **{median_shared}**; mean \
Jaccard similarity is **{mean_jaccard:.4f}**. This indicates that BC is partially reflecting \
existing co-assignment structure (joint ventures, subsidiaries missed by the containment-based \
dedup, or long-running technical collaborations across firm boundaries).

Downstream M&A prediction regressions should consider including a shared-patent count (`n_shared`) \
or Jaccard similarity (`jaccard`) as a control covariate. This isolates the pure \
technological-similarity signal from already-existing structural ties — otherwise top-BC \
coefficients may be capturing a "firms that already collaborate" effect rather than a \
"firms whose technologies complement each other" effect.

## Full Results

See `output/kmax_sweep/coassignment_audit.parquet` (not committed; shipped in the handoff bundle).
Columns: `rank, gvkey_a, gvkey_b, bc, n_patents_a, n_patents_b, n_shared, jaccard, overlap_fraction`.
"""
    OUT_SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY_MD.write_text(summary)
    print(f"Wrote {OUT_SUMMARY_MD}")

    print(
        f"\nSummary: median_shared={median_shared}, mean_jaccard={mean_jaccard:.4f}, "
        f">10%={n_ge10}/{TOP_K}, >25%={n_ge25}/{TOP_K}, zero_shared={n_zero}/{TOP_K}, "
        f"missing_map={missing_map_coverage}/{TOP_K}"
    )


if __name__ == "__main__":
    main()
