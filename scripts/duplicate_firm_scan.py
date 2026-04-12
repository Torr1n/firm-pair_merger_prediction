"""Project-wide duplicate firm detection.

For every firm pair that shares at least one patent, compute:
- Jaccard similarity = |A ∩ B| / |A ∪ B|
- Containment_max = max(|A∩B|/|A|, |A∩B|/|B|) — detects nested relationships

Reports distribution, threshold counts, and representative examples at each
overlap level so we can choose a deduplication rule based on actual data.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

GVKEY_MAP_PATH = "output/week2_inputs/gvkey_map.parquet"
OUTPUT_PATH = Path("output/kmax_sweep/duplicate_scan.json")


def main():
    print("Loading gvkey_map...")
    df = pd.read_parquet(GVKEY_MAP_PATH)
    print(f"  {len(df):,} (firm, patent) rows")
    print(f"  {df['gvkey'].nunique():,} unique firms")
    print(f"  {df['patent_id'].nunique():,} unique patents")

    # Build firm -> set of patents
    print("\nBuilding firm patent sets...")
    firm_patents = df.groupby("gvkey")["patent_id"].apply(set).to_dict()
    firm_sizes = {f: len(p) for f, p in firm_patents.items()}

    # Filter to non-excluded firms (≥5 patents) — these are the firms that
    # participate in the BC analysis. Including <5 firms would inflate the
    # noise in the duplicate scan.
    non_excluded = {f for f, n in firm_sizes.items() if n >= 5}
    print(f"  {len(non_excluded):,} firms with ≥5 patents (non-excluded)")

    # Build inverted index: patent -> list of firms (for non-excluded only)
    print("\nBuilding inverted index (patent -> firms)...")
    patent_to_firms = defaultdict(list)
    df_filtered = df[df["gvkey"].isin(non_excluded)]
    for row in df_filtered.itertuples(index=False):
        patent_to_firms[row.patent_id].append(row.gvkey)

    n_co_assigned = sum(1 for fs in patent_to_firms.values() if len(fs) > 1)
    print(f"  {n_co_assigned:,} patents are co-assigned (in 2+ firms)")

    # For each co-assigned patent, increment intersection counts for all firm pairs
    print("\nComputing pairwise intersections from co-assignments...")
    intersection_counts = defaultdict(int)
    n_processed = 0
    for patent_id, firm_list in patent_to_firms.items():
        if len(firm_list) > 1:
            # Sort to ensure canonical pair ordering
            firm_list = sorted(set(firm_list))
            for i, firm_a in enumerate(firm_list):
                for firm_b in firm_list[i + 1:]:
                    intersection_counts[(firm_a, firm_b)] += 1
            n_processed += 1
            if n_processed % 100000 == 0:
                print(f"    {n_processed:,} co-assigned patents processed, "
                      f"{len(intersection_counts):,} pairs so far")

    print(f"  {len(intersection_counts):,} firm pairs share at least one patent")

    # Compute Jaccard and containment for all overlapping pairs
    print("\nComputing Jaccard and containment ratios...")
    records = []
    for (firm_a, firm_b), intersection in intersection_counts.items():
        size_a = firm_sizes[firm_a]
        size_b = firm_sizes[firm_b]
        union = size_a + size_b - intersection
        jaccard = intersection / union
        containment_a = intersection / size_a  # fraction of A in B
        containment_b = intersection / size_b  # fraction of B in A
        containment_max = max(containment_a, containment_b)
        containment_min = min(containment_a, containment_b)
        records.append({
            "firm_a": firm_a,
            "firm_b": firm_b,
            "size_a": size_a,
            "size_b": size_b,
            "intersection": intersection,
            "jaccard": jaccard,
            "containment_max": containment_max,
            "containment_min": containment_min,
        })
    pairs_df = pd.DataFrame(records)
    print(f"  {len(pairs_df):,} overlapping pairs computed")

    # Save full pairs dataframe for downstream analysis
    pairs_df.to_parquet("output/kmax_sweep/duplicate_pairs.parquet")
    print(f"  Saved to output/kmax_sweep/duplicate_pairs.parquet")

    # ----- Distribution analysis -----
    print("\n" + "=" * 70)
    print("JACCARD DISTRIBUTION")
    print("=" * 70)

    jaccards = pairs_df["jaccard"].values
    print(f"\nTotal overlapping pairs: {len(jaccards):,}")
    print(f"Total non-excluded firms: {len(non_excluded):,}")
    print(f"Total possible pairs: {len(non_excluded) * (len(non_excluded) - 1) // 2:,}")
    print(f"Fraction of pairs with any overlap: "
          f"{len(jaccards) / (len(non_excluded) * (len(non_excluded) - 1) / 2) * 100:.4f}%")

    print(f"\nJaccard percentiles:")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9, 99.99]:
        val = np.percentile(jaccards, p)
        print(f"  P{p:>5.2f}: {val:.4f}")

    print(f"\nThreshold counts (Jaccard ≥ x):")
    thresholds = [1.0, 0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05, 0.01]
    threshold_counts = {}
    for t in thresholds:
        n = int(np.sum(jaccards >= t))
        n_firms = len(set(pairs_df[pairs_df["jaccard"] >= t]["firm_a"]) |
                      set(pairs_df[pairs_df["jaccard"] >= t]["firm_b"]))
        print(f"  Jaccard ≥ {t:.2f}: {n:>7,} pairs   ({n_firms:>5,} unique firms involved)")
        threshold_counts[str(t)] = {"n_pairs": n, "n_firms": n_firms}

    print(f"\nContainment_max (asymmetric overlap) distribution:")
    cmax = pairs_df["containment_max"].values
    for p in [50, 75, 90, 95, 99, 99.9, 99.99]:
        val = np.percentile(cmax, p)
        print(f"  P{p:>5.2f}: {val:.4f}")

    print(f"\nNested-relationship counts (containment_max ≥ x):")
    nested_counts = {}
    for t in [1.0, 0.99, 0.95, 0.90, 0.80]:
        n = int(np.sum(cmax >= t))
        print(f"  containment_max ≥ {t:.2f}: {n:>7,} pairs")
        nested_counts[str(t)] = n

    # ----- Strict aliases: 100% Jaccard (perfect duplicates) -----
    print("\n" + "=" * 70)
    print("STRICT ALIASES (Jaccard = 1.000)")
    print("=" * 70)
    aliases = pairs_df[pairs_df["jaccard"] == 1.0].sort_values("size_a", ascending=False)
    print(f"\nTotal: {len(aliases)} pairs")

    if len(aliases) > 0:
        # Identify connected components (cliques of mutually-aliased firms)
        adj = defaultdict(set)
        for row in aliases.itertuples(index=False):
            adj[row.firm_a].add(row.firm_b)
            adj[row.firm_b].add(row.firm_a)

        visited = set()
        cliques = []
        for f in adj:
            if f in visited:
                continue
            stack = [f]
            clique = set()
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                clique.add(cur)
                stack.extend(adj[cur] - visited)
            if len(clique) >= 2:
                cliques.append(sorted(clique))

        print(f"Connected to {len(cliques)} alias cliques (groups of mutually-identical firms)")
        clique_sizes = [len(c) for c in cliques]
        print(f"Clique size distribution: min={min(clique_sizes)}, max={max(clique_sizes)}, "
              f"mean={np.mean(clique_sizes):.1f}")
        size_counts = pd.Series(clique_sizes).value_counts().sort_index()
        for size, count in size_counts.items():
            print(f"  Cliques of size {size}: {count}")

        n_unique_aliased_firms = sum(clique_sizes)
        print(f"\nTotal firms involved in alias cliques: {n_unique_aliased_firms}")
        print(f"  → if we kept ONE per clique, we'd remove {n_unique_aliased_firms - len(cliques)} firms")

        # Top 20 largest cliques
        print("\nTop 20 largest alias cliques (by member firm patent count):")
        cliques_sorted = sorted(cliques, key=lambda c: -firm_sizes[c[0]])[:20]
        for i, clique in enumerate(cliques_sorted, 1):
            print(f"\n  Clique #{i}: {len(clique)} firms, {firm_sizes[clique[0]]} patents each")
            for f in clique[:8]:
                print(f"    {f}")
            if len(clique) > 8:
                print(f"    ... and {len(clique) - 8} more")
    else:
        cliques = []

    # ----- Near-aliases: 95-99% Jaccard -----
    print("\n" + "=" * 70)
    print("NEAR-ALIASES (0.95 ≤ Jaccard < 1.000)")
    print("=" * 70)
    near = pairs_df[(pairs_df["jaccard"] >= 0.95) & (pairs_df["jaccard"] < 1.0)]
    print(f"\nTotal: {len(near)} pairs")
    if len(near) > 0:
        print("\nTop 15 by patent count:")
        for row in near.nlargest(15, "size_a").itertuples(index=False):
            print(f"  Jac={row.jaccard:.4f}  C_max={row.containment_max:.4f}  "
                  f"|A|={row.size_a:>4d} |B|={row.size_b:>4d} ∩={row.intersection:>4d}  "
                  f"{row.firm_a:>30s} <-> {row.firm_b}")

    # ----- Nested relationships: high containment, low Jaccard -----
    print("\n" + "=" * 70)
    print("NESTED RELATIONSHIPS (containment_max ≥ 0.95 AND Jaccard < 0.80)")
    print("=" * 70)
    nested = pairs_df[(pairs_df["containment_max"] >= 0.95) & (pairs_df["jaccard"] < 0.80)]
    print(f"\nTotal: {len(nested)} pairs (one firm fully nested in a larger firm)")
    if len(nested) > 0:
        print("\nTop 15 (largest 'parent' first):")
        nested_sorted = nested.copy()
        nested_sorted["max_size"] = nested_sorted[["size_a", "size_b"]].max(axis=1)
        for row in nested_sorted.nlargest(15, "max_size").itertuples(index=False):
            print(f"  Jac={row.jaccard:.4f}  C_max={row.containment_max:.4f}  "
                  f"|A|={row.size_a:>5d} |B|={row.size_b:>5d} ∩={row.intersection:>5d}  "
                  f"{row.firm_a:>30s} <-> {row.firm_b}")

    # ----- Moderate overlap: 50-80% Jaccard -----
    print("\n" + "=" * 70)
    print("MODERATE OVERLAP (0.50 ≤ Jaccard < 0.95)")
    print("=" * 70)
    moderate = pairs_df[(pairs_df["jaccard"] >= 0.50) & (pairs_df["jaccard"] < 0.95)]
    print(f"\nTotal: {len(moderate)} pairs")
    if len(moderate) > 0:
        print("\nTop 15 by patent count:")
        for row in moderate.nlargest(15, "size_a").itertuples(index=False):
            print(f"  Jac={row.jaccard:.4f}  C_max={row.containment_max:.4f}  "
                  f"|A|={row.size_a:>5d} |B|={row.size_b:>5d} ∩={row.intersection:>5d}  "
                  f"{row.firm_a:>30s} <-> {row.firm_b}")

    # ----- Histogram for visualization -----
    print("\n" + "=" * 70)
    print("HISTOGRAM (log-scale)")
    print("=" * 70)
    bins = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0001]
    hist, _ = np.histogram(jaccards, bins=bins)
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        bar_len = int(np.log10(max(hist[i], 1)) * 5)
        bar = "█" * bar_len
        print(f"  [{low:.3f}, {high:.3f}): {hist[i]:>7,} {bar}")

    # ----- Save summary JSON -----
    summary = {
        "total_overlapping_pairs": len(pairs_df),
        "total_non_excluded_firms": len(non_excluded),
        "fraction_with_overlap_pct": len(pairs_df) / (len(non_excluded) * (len(non_excluded) - 1) / 2) * 100,
        "jaccard_percentiles": {
            f"P{p}": float(np.percentile(jaccards, p))
            for p in [50, 75, 90, 95, 99, 99.5, 99.9, 99.99]
        },
        "threshold_counts": threshold_counts,
        "containment_thresholds": nested_counts,
        "n_strict_aliases": int(len(aliases)) if len(aliases) > 0 else 0,
        "n_alias_cliques": len(cliques) if len(aliases) > 0 else 0,
        "n_near_aliases_95_to_99": int(len(near)),
        "n_nested": int(len(nested)),
        "n_moderate_overlap_50_to_95": int(len(moderate)),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
