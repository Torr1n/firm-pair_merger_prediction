"""Apply the unified containment-based deduplication rule.

Rule: Drop firm B if there exists another firm A in the dataset such that
|A| ≥ |B| and at least 95% of B's patents are also in A.

This catches three categories of "same legal entity" relationships:
1. Anagram aliases (containment 1.0 in both directions)
2. Subsidiaries (containment 1.0 from sub to parent)
3. Predecessor records (containment ≈1.0 from old to new)

Tiebreaker for equal-size cases: drop PRIV_-prefixed if mixed; otherwise
drop alphabetically larger.

Reports:
- Total firms removed
- Breakdown by category (alias clique / subsidiary / predecessor)
- Borderline cases (containment 0.95-0.97) for sanity checking
- Specific test cases Torrin asked about
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS_PATH = "output/kmax_sweep/duplicate_pairs.parquet"
GVKEY_MAP_PATH = "output/week2_inputs/gvkey_map.parquet"
OUTPUT_DIR = Path("output/kmax_sweep")


def main():
    print("Loading pairs and firm sizes...")
    pairs_df = pd.read_parquet(PAIRS_PATH)
    print(f"  {len(pairs_df):,} overlapping firm pairs loaded")

    df = pd.read_parquet(GVKEY_MAP_PATH)
    firm_sizes = df.groupby("gvkey").size().to_dict()
    non_excluded = {f for f, n in firm_sizes.items() if n >= 5}
    print(f"  {len(non_excluded):,} non-excluded firms in dataset")

    # ----- Apply the unified rule -----
    print("\nApplying unified rule: drop B if ∃ A with |A|≥|B| and containment(B→A) ≥ 0.95")

    candidates = pairs_df[pairs_df["containment_max"] >= 0.95].copy()
    print(f"  {len(candidates):,} pairs with containment_max ≥ 0.95")

    # For each pair, identify which firm to drop
    to_drop = set()
    drop_reason = {}  # firm -> reason
    drop_kept_with = {}  # firm -> the larger firm that kept

    for row in candidates.itertuples(index=False):
        # Determine smaller (drop) and larger (keep) firm
        if row.size_a > row.size_b:
            smaller, larger = row.firm_b, row.firm_a
        elif row.size_b > row.size_a:
            smaller, larger = row.firm_a, row.firm_b
        else:
            # Equal size: apply tiebreaker
            a_is_priv = row.firm_a.startswith("PRIV_")
            b_is_priv = row.firm_b.startswith("PRIV_")
            if a_is_priv and not b_is_priv:
                smaller, larger = row.firm_a, row.firm_b
            elif b_is_priv and not a_is_priv:
                smaller, larger = row.firm_b, row.firm_a
            else:
                # Both PRIV_ or both non-PRIV: alphabetically larger drops
                if row.firm_a > row.firm_b:
                    smaller, larger = row.firm_a, row.firm_b
                else:
                    smaller, larger = row.firm_b, row.firm_a

        # Categorize the relationship
        if row.containment_min >= 0.95:
            category = "alias"  # Bidirectional containment
        elif row.size_a == row.size_b:
            category = "alias"  # Same size, near-perfect overlap
        else:
            # One firm is much larger; smaller is nested but larger has additional patents
            ratio = max(row.size_a, row.size_b) / min(row.size_a, row.size_b)
            if ratio < 1.5:
                category = "predecessor"  # Similar size but nested
            else:
                category = "subsidiary"  # Much larger parent

        # Only mark to drop if not already marked with a better (larger) keeper
        if smaller not in to_drop or firm_sizes[larger] > firm_sizes.get(drop_kept_with.get(smaller, smaller), 0):
            to_drop.add(smaller)
            drop_reason[smaller] = category
            drop_kept_with[smaller] = larger

    print(f"\n  Firms marked for removal: {len(to_drop)}")
    print(f"  Remaining non-excluded firms: {len(non_excluded) - len(to_drop)}")

    # Breakdown by category
    print("\n  Breakdown by removal category:")
    cat_counts = pd.Series(list(drop_reason.values())).value_counts()
    for cat, count in cat_counts.items():
        print(f"    {cat:>12s}: {count:>5,}")

    # Save the dropped-firms list with details
    drop_records = []
    for firm in to_drop:
        keeper = drop_kept_with[firm]
        drop_records.append({
            "dropped": firm,
            "kept": keeper,
            "dropped_size": firm_sizes[firm],
            "kept_size": firm_sizes[keeper],
            "category": drop_reason[firm],
        })
    drop_df = pd.DataFrame(drop_records).sort_values("dropped_size", ascending=False)
    drop_df.to_parquet(OUTPUT_DIR / "deduplication_decisions.parquet")
    drop_df.to_csv(OUTPUT_DIR / "deduplication_decisions.csv", index=False)
    print(f"\n  Decision log saved to {OUTPUT_DIR}/deduplication_decisions.{{parquet,csv}}")

    # ----- Test cases Torrin asked about -----
    print("\n" + "=" * 70)
    print("SPECIFIC TEST CASES")
    print("=" * 70)
    test_pairs = [
        ("160329", "PRIV_WAYMO"),
        ("160329", "PRIV_VERILYLIFESCIENCES"),
        ("160329", "PRIV_GOOGLE"),
        ("024800", "PRIV_SNAPTRACK"),
        ("024800", "PRIV_QUALCOMMATHEROS"),
        ("006066", "PRIV_INTERNATIONALBUSINESS"),
        ("007585", "PRIV_MOTOROLAMOBILITY"),
        ("034873", "PRIV_LYFT"),
        ("003760", "PRIV_GENERALDATA"),
    ]
    print()
    for a, b in test_pairs:
        # Look up the pair
        pair = pairs_df[
            ((pairs_df["firm_a"] == a) & (pairs_df["firm_b"] == b)) |
            ((pairs_df["firm_a"] == b) & (pairs_df["firm_b"] == a))
        ]
        if len(pair) == 0:
            print(f"  {a:>15s} <-> {b:<35s}  (no shared patents)")
            continue
        row = pair.iloc[0]

        a_dropped = a in to_drop
        b_dropped = b in to_drop
        a_size = firm_sizes.get(a, 0)
        b_size = firm_sizes.get(b, 0)

        verdict = ""
        if a_dropped and not b_dropped:
            verdict = f"DROP {a}, KEEP {b}"
        elif b_dropped and not a_dropped:
            verdict = f"KEEP {a}, DROP {b}"
        elif a_dropped and b_dropped:
            verdict = "BOTH DROPPED (transitive)"
        else:
            verdict = "BOTH KEPT (rule not triggered)"

        print(f"  Jac={row['jaccard']:.4f}  C_max={row['containment_max']:.4f}  "
              f"|{a}|={a_size:>5d} |{b}|={b_size:>5d}  →  {verdict}")

    # ----- Borderline cases (containment 0.95 - 0.97) -----
    print("\n" + "=" * 70)
    print("BORDERLINE CASES (containment_max in [0.95, 0.97))")
    print("=" * 70)
    borderline = pairs_df[
        (pairs_df["containment_max"] >= 0.95) &
        (pairs_df["containment_max"] < 0.97)
    ].sort_values("size_a", ascending=False)
    print(f"\n  {len(borderline)} pairs in borderline range")
    print(f"  These are the closest calls — would NOT be removed if rule were tightened to 0.97")
    print()
    for row in borderline.head(20).itertuples(index=False):
        print(f"    Jac={row.jaccard:.3f}  C_max={row.containment_max:.3f}  "
              f"|A|={row.size_a:>5d} |B|={row.size_b:>5d} ∩={row.intersection:>5d}  "
              f"{row.firm_a:>30s} <-> {row.firm_b}")

    # ----- Cases caught newly (compared to old Jaccard ≥ 0.99 rule) -----
    print("\n" + "=" * 70)
    print("NEWLY CAUGHT BY UNIFIED RULE (containment ≥ 0.95 but Jaccard < 0.99)")
    print("=" * 70)
    newly_caught = pairs_df[
        (pairs_df["containment_max"] >= 0.95) &
        (pairs_df["jaccard"] < 0.99)
    ].copy()
    print(f"\n  {len(newly_caught)} pairs caught by containment that the Jaccard rule would miss")

    # Group by parent firm to see concentration
    parent_counts = {}
    for row in newly_caught.itertuples(index=False):
        if row.size_a >= row.size_b:
            parent = row.firm_a
        else:
            parent = row.firm_b
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
    parent_summary = pd.Series(parent_counts).sort_values(ascending=False)

    print(f"\n  Top 15 parent firms by number of subsidiary records caught:")
    for parent, count in parent_summary.head(15).items():
        print(f"    {parent:>15s} (size={firm_sizes[parent]:>6d}): {count} sub-records")

    print(f"\n  Sample of newly-caught pairs (largest parent first):")
    newly_caught["parent_size"] = newly_caught[["size_a", "size_b"]].max(axis=1)
    for row in newly_caught.nlargest(20, "parent_size").itertuples(index=False):
        print(f"    Jac={row.jaccard:.3f}  C_max={row.containment_max:.3f}  "
              f"|A|={row.size_a:>5d} |B|={row.size_b:>5d}  "
              f"{row.firm_a:>30s} <-> {row.firm_b}")

    # ----- Pairs NOT caught (just below threshold or low containment) -----
    print("\n" + "=" * 70)
    print("PAIRS NOT CAUGHT (containment_max ≥ 0.85 but < 0.95)")
    print("=" * 70)
    print("These are the next-most-likely candidates if we lowered the cutoff:")
    not_caught = pairs_df[
        (pairs_df["containment_max"] >= 0.85) &
        (pairs_df["containment_max"] < 0.95)
    ].sort_values("containment_max", ascending=False)
    print(f"  {len(not_caught)} pairs in [0.85, 0.95) containment range\n")
    for row in not_caught.head(15).itertuples(index=False):
        print(f"    Jac={row.jaccard:.3f}  C_max={row.containment_max:.3f}  "
              f"|A|={row.size_a:>5d} |B|={row.size_b:>5d}  "
              f"{row.firm_a:>30s} <-> {row.firm_b}")

    # ----- Impact on the K_max sweep dataset -----
    print("\n" + "=" * 70)
    print("IMPACT ON K_max SWEEP DATASET")
    print("=" * 70)

    # How many of the dropped firms are GMM-tier vs single-Gaussian?
    dropped_sizes = [firm_sizes[f] for f in to_drop]
    n_gmm_dropped = sum(1 for s in dropped_sizes if s >= 50)
    n_sg_dropped = sum(1 for s in dropped_sizes if 5 <= s < 50)
    n_excluded_dropped = sum(1 for s in dropped_sizes if s < 5)

    print(f"\n  Of {len(to_drop):,} dropped firms:")
    print(f"    GMM-tier (≥50 patents):       {n_gmm_dropped:>5,}")
    print(f"    Single-Gaussian (5-49):        {n_sg_dropped:>5,}")
    print(f"    Already-excluded (<5):         {n_excluded_dropped:>5,}")

    # Patent count distribution of dropped firms
    if dropped_sizes:
        sizes_arr = np.array(dropped_sizes)
        print(f"\n  Dropped-firm patent count distribution:")
        print(f"    min={sizes_arr.min()}, median={int(np.median(sizes_arr))}, "
              f"mean={sizes_arr.mean():.0f}, max={sizes_arr.max()}")
        print(f"    P25={int(np.percentile(sizes_arr, 25))}, "
              f"P75={int(np.percentile(sizes_arr, 75))}, "
              f"P90={int(np.percentile(sizes_arr, 90))}")

    # Effect on small-firm bin (where the K explosion lives)
    small_firms_kept = sum(1 for f, n in firm_sizes.items()
                           if 50 <= n <= 200 and f not in to_drop)
    small_firms_dropped = sum(1 for f in to_drop
                              if 50 <= firm_sizes[f] <= 200)
    print(f"\n  Effect on small GMM-tier firms (50-200 patents):")
    print(f"    Originally: {small_firms_kept + small_firms_dropped:,}")
    print(f"    Dropped:    {small_firms_dropped:,}")
    print(f"    Remaining:  {small_firms_kept:,}")
    print(f"    Reduction:  {small_firms_dropped / (small_firms_kept + small_firms_dropped) * 100:.1f}%")

    # ----- Save summary -----
    summary = {
        "rule": "drop firm B if ∃ A with |A|≥|B| and containment_max(pair) ≥ 0.95",
        "tiebreaker": "if sizes equal, drop PRIV_-prefixed; otherwise alphabetically larger",
        "n_pairs_with_containment_above_threshold": len(candidates),
        "n_firms_dropped": len(to_drop),
        "n_firms_remaining_non_excluded": len(non_excluded) - len(to_drop),
        "dropped_by_category": cat_counts.to_dict(),
        "n_dropped_gmm_tier": n_gmm_dropped,
        "n_dropped_single_gaussian": n_sg_dropped,
        "n_dropped_50_to_200_patents": small_firms_dropped,
        "small_firm_bin_reduction_pct": (
            small_firms_dropped / (small_firms_kept + small_firms_dropped) * 100
        ),
    }
    with open(OUTPUT_DIR / "deduplication_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {OUTPUT_DIR}/deduplication_summary.json")


if __name__ == "__main__":
    main()
