"""Identify the actual firm pairs in the top-50 at each K_max.

This reveals WHO the pairs are and explains the near-tie phenomenon.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from scripts.run_kmax_sweep import load_gmm_results

OUTPUT_DIR = Path("output/kmax_sweep")


def top_pairs_with_info(k_max, n=30):
    """Get top-n pairs with their firm details."""
    data = np.load(OUTPUT_DIR / f"bc_matrix_all_k{k_max}.npz", allow_pickle=True)
    gvkeys = list(data["gvkeys"])
    bc = data["bc_matrix"]

    # Get results for firm metadata
    results = {r["gvkey"]: r for r in
               load_gmm_results(str(OUTPUT_DIR / f"firm_gmm_parameters_k{k_max}.parquet"))}

    idx = np.triu_indices_from(bc, k=1)
    vals = bc[idx]
    top_idx = np.argpartition(vals, -n)[-n:]
    top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]

    pairs = []
    for i in top_idx:
        a, b = idx[0][i], idx[1][i]
        ga, gb = gvkeys[a], gvkeys[b]
        ra = results.get(ga, {})
        rb = results.get(gb, {})
        pairs.append({
            "gvkey_a": ga,
            "gvkey_b": gb,
            "bc": float(vals[i]),
            "tier_a": ra.get("tier", "?"),
            "tier_b": rb.get("tier", "?"),
            "k_a": ra.get("n_components", 0),
            "k_b": rb.get("n_components", 0),
            "n_patents_a": ra.get("n_patents", 0),
            "n_patents_b": rb.get("n_patents", 0),
        })

    del bc, data
    return pairs


def main():
    # Load gvkey_map for patent counts
    gvkey_map = pd.read_parquet("output/week2_inputs/gvkey_map.parquet")
    patent_counts = gvkey_map.groupby("gvkey").size().to_dict()

    # Also check for shared patents between top-pair firms
    patent_by_firm = gvkey_map.groupby("gvkey")["patent_id"].apply(set).to_dict()

    for k_max in [10, 15, 20, 30]:
        print(f"\n{'='*70}")
        print(f"TOP-30 PAIRS at K_max={k_max}")
        print(f"{'='*70}")

        pairs = top_pairs_with_info(k_max, 30)

        for i, p in enumerate(pairs, 1):
            # Check for shared patents
            pa = patent_by_firm.get(p["gvkey_a"], set())
            pb = patent_by_firm.get(p["gvkey_b"], set())
            shared = len(pa & pb)
            total_a = len(pa)
            total_b = len(pb)

            print(f"\n  #{i:2d} BC={p['bc']:.6f}")
            print(f"      {p['gvkey_a']:>30s} ({p['tier_a']}, K={p['k_a']}, "
                  f"n={total_a})")
            print(f"      {p['gvkey_b']:>30s} ({p['tier_b']}, K={p['k_b']}, "
                  f"n={total_b})")
            if shared > 0:
                pct_a = shared / total_a * 100 if total_a > 0 else 0
                pct_b = shared / total_b * 100 if total_b > 0 else 0
                print(f"      *** SHARED PATENTS: {shared} "
                      f"({pct_a:.0f}% of A, {pct_b:.0f}% of B) ***")

    # Summary: how many top-30 pairs at K_max=10 involve shared patents?
    print(f"\n\n{'='*70}")
    print("SHARED PATENT ANALYSIS")
    print(f"{'='*70}")

    for k_max in [10, 20, 30]:
        pairs = top_pairs_with_info(k_max, 50)
        n_shared = 0
        total_shared = 0
        for p in pairs:
            pa = patent_by_firm.get(p["gvkey_a"], set())
            pb = patent_by_firm.get(p["gvkey_b"], set())
            shared = len(pa & pb)
            if shared > 0:
                n_shared += 1
                total_shared += shared
        print(f"\n  K_max={k_max}: {n_shared}/{len(pairs)} top-50 pairs share patents "
              f"(total shared: {total_shared})")


if __name__ == "__main__":
    main()
