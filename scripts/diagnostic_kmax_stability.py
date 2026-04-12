"""K_max Sweep Diagnostic Analysis: Near-Ties vs Misspecification.

This script executes the four-step diagnostic sequence prescribed by Codex
to explain the 0% top-50 overlap at K_max transitions 15→20, 20→25.

The central question: Is the top-tail instability caused by (a) near-tie
ranking noise in densely packed BC values, (b) genuine structural sensitivity
to K_max, or (c) both?

Outputs: diagnostic_results.json with all findings.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output/kmax_sweep")
K_MAX_VALUES = [10, 15, 20, 25, 30]
TOP_K_SIZES = [50, 100, 200, 500]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_bc_matrix(k_max: int) -> tuple[list[str], np.ndarray]:
    """Load a BC matrix and its gvkey ordering."""
    path = OUTPUT_DIR / f"bc_matrix_all_k{k_max}.npz"
    data = np.load(path, allow_pickle=True)
    gvkeys = list(data["gvkeys"])
    bc = data["bc_matrix"]
    return gvkeys, bc


def upper_triangle_values(bc: np.ndarray) -> np.ndarray:
    """Extract upper-triangle BC values (excluding diagonal)."""
    idx = np.triu_indices_from(bc, k=1)
    return bc[idx]


def top_k_pairs(gvkeys: list[str], bc: np.ndarray, k: int) -> set[tuple[str, str]]:
    """Return the top-k firm pairs by BC value as a set of (gvkey_a, gvkey_b) tuples."""
    idx = np.triu_indices_from(bc, k=1)
    vals = bc[idx]
    # Get indices of top-k values
    top_idx = np.argpartition(vals, -k)[-k:]
    top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]
    pairs = set()
    for i in top_idx:
        a, b = idx[0][i], idx[1][i]
        pairs.add((gvkeys[a], gvkeys[b]))
    return pairs


def top_k_pairs_with_values(gvkeys: list[str], bc: np.ndarray, k: int) -> list[tuple[str, str, float]]:
    """Return top-k pairs with their BC values, sorted descending."""
    idx = np.triu_indices_from(bc, k=1)
    vals = bc[idx]
    top_idx = np.argpartition(vals, -k)[-k:]
    top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]
    result = []
    for i in top_idx:
        a, b = idx[0][i], idx[1][i]
        result.append((gvkeys[a], gvkeys[b], float(vals[i])))
    return result


def pair_rank_map(gvkeys: list[str], bc: np.ndarray, max_rank: int = 500) -> dict[tuple[str, str], int]:
    """Map each top-max_rank pair to its rank (1-indexed)."""
    idx = np.triu_indices_from(bc, k=1)
    vals = bc[idx]
    top_idx = np.argpartition(vals, -max_rank)[-max_rank:]
    top_idx = top_idx[np.argsort(vals[top_idx])[::-1]]
    rank_map = {}
    for rank, i in enumerate(top_idx, 1):
        a, b = idx[0][i], idx[1][i]
        pair = (gvkeys[a], gvkeys[b])
        rank_map[pair] = rank
    return rank_map


# ---------------------------------------------------------------------------
# Diagnostic 1: Tail-Margin Analysis
# ---------------------------------------------------------------------------

def diagnostic_1_tail_margins():
    """Are the top-50 BC values densely packed with near-ties?"""
    print("=" * 70)
    print("DIAGNOSTIC 1: Tail-Margin Analysis")
    print("=" * 70)

    results = {}

    for k_max in K_MAX_VALUES:
        print(f"\n--- K_max = {k_max} ---")
        gvkeys, bc = load_bc_matrix(k_max)
        vals = upper_triangle_values(bc)

        # Sort descending
        sorted_vals = np.sort(vals)[::-1]
        n_pairs = len(sorted_vals)

        # BC values at key ranks
        rank_values = {}
        for rank in [1, 5, 10, 25, 50, 100, 200, 500, 1000]:
            if rank <= n_pairs:
                rank_values[rank] = float(sorted_vals[rank - 1])

        print(f"  BC at rank 1:    {rank_values.get(1, 'N/A'):.6f}")
        print(f"  BC at rank 10:   {rank_values.get(10, 'N/A'):.6f}")
        print(f"  BC at rank 25:   {rank_values.get(25, 'N/A'):.6f}")
        print(f"  BC at rank 50:   {rank_values.get(50, 'N/A'):.6f}")
        print(f"  BC at rank 100:  {rank_values.get(100, 'N/A'):.6f}")
        print(f"  BC at rank 200:  {rank_values.get(200, 'N/A'):.6f}")
        print(f"  BC at rank 500:  {rank_values.get(500, 'N/A'):.6f}")
        print(f"  BC at rank 1000: {rank_values.get(1000, 'N/A'):.6f}")

        # Gaps between adjacent ranks in the top-50
        top50 = sorted_vals[:50]
        gaps_top50 = np.diff(top50)  # negative (descending)
        abs_gaps_top50 = np.abs(gaps_top50)

        print(f"\n  Top-50 gap statistics:")
        print(f"    Mean gap:   {abs_gaps_top50.mean():.8f}")
        print(f"    Median gap: {np.median(abs_gaps_top50):.8f}")
        print(f"    Max gap:    {abs_gaps_top50.max():.8f}")
        print(f"    Min gap:    {abs_gaps_top50.min():.8f}")
        print(f"    Span (rank1 - rank50): {top50[0] - top50[-1]:.8f}")

        # Same for top-200
        top200 = sorted_vals[:200]
        gaps_top200 = np.abs(np.diff(top200))
        print(f"\n  Top-200 gap statistics:")
        print(f"    Mean gap:   {gaps_top200.mean():.8f}")
        print(f"    Median gap: {np.median(gaps_top200):.8f}")
        print(f"    Span (rank1 - rank200): {top200[0] - top200[-1]:.8f}")

        # How many pairs are within epsilon of the top-50 cutoff?
        cutoff_50 = sorted_vals[49]
        for eps in [1e-4, 1e-3, 5e-3, 1e-2]:
            n_within = int(np.sum(np.abs(sorted_vals - cutoff_50) < eps))
            print(f"    Pairs within {eps:.0e} of rank-50 cutoff ({cutoff_50:.6f}): {n_within}")

        # Threshold-based analysis: how many pairs have BC > various thresholds?
        for threshold in [0.9, 0.8, 0.7, 0.5, 0.3]:
            n_above = int(np.sum(sorted_vals > threshold))
            print(f"    Pairs with BC > {threshold}: {n_above}")

        results[str(k_max)] = {
            "rank_values": {str(k): v for k, v in rank_values.items()},
            "top50_span": float(top50[0] - top50[-1]),
            "top50_mean_gap": float(abs_gaps_top50.mean()),
            "top50_median_gap": float(np.median(abs_gaps_top50)),
            "top200_span": float(top200[0] - top200[-1]),
            "top200_mean_gap": float(gaps_top200.mean()),
            "n_within_1e4_of_cutoff50": int(np.sum(np.abs(sorted_vals - cutoff_50) < 1e-4)),
            "n_within_1e3_of_cutoff50": int(np.sum(np.abs(sorted_vals - cutoff_50) < 1e-3)),
            "n_within_1e2_of_cutoff50": int(np.sum(np.abs(sorted_vals - cutoff_50) < 1e-2)),
        }

        del bc, vals, sorted_vals  # free memory

    # Cross-K_max comparison: how much do BC values shift?
    print("\n--- Cross-K_max BC shift at fixed pairs ---")
    gvkeys_10, bc_10 = load_bc_matrix(10)
    gvkeys_15, bc_15 = load_bc_matrix(15)
    gvkeys_20, bc_20 = load_bc_matrix(20)

    # Verify gvkey ordering is consistent
    assert gvkeys_10 == gvkeys_15 == gvkeys_20, "Gvkey ordering differs!"

    vals_10 = upper_triangle_values(bc_10)
    vals_15 = upper_triangle_values(bc_15)
    vals_20 = upper_triangle_values(bc_20)

    # How much do individual pair BC values change?
    delta_10_15 = vals_15 - vals_10
    delta_15_20 = vals_20 - vals_15

    print(f"\n  BC shift K=10→15:")
    print(f"    Mean absolute shift: {np.abs(delta_10_15).mean():.8f}")
    print(f"    Median absolute shift: {np.median(np.abs(delta_10_15)):.8f}")
    print(f"    P99 absolute shift: {np.percentile(np.abs(delta_10_15), 99):.8f}")
    print(f"    Max absolute shift: {np.abs(delta_10_15).max():.8f}")

    print(f"\n  BC shift K=15→20:")
    print(f"    Mean absolute shift: {np.abs(delta_15_20).mean():.8f}")
    print(f"    Median absolute shift: {np.median(np.abs(delta_15_20)):.8f}")
    print(f"    P99 absolute shift: {np.percentile(np.abs(delta_15_20), 99):.8f}")
    print(f"    Max absolute shift: {np.abs(delta_15_20).max():.8f}")

    # Critical comparison: gap between ranks in top-50 vs. shift magnitude
    top50_10 = np.sort(vals_10)[::-1][:50]
    gap_in_top50 = np.abs(np.diff(top50_10)).mean()
    shift_magnitude = np.abs(delta_10_15).mean()

    print(f"\n  CRITICAL RATIO (K=10→15):")
    print(f"    Mean gap between adjacent top-50 ranks: {gap_in_top50:.8f}")
    print(f"    Mean BC shift across all pairs:         {shift_magnitude:.8f}")
    print(f"    Ratio (shift / gap):                    {shift_magnitude / gap_in_top50:.2f}")
    print(f"    → If ratio >> 1, shifts are much larger than gaps → near-tie explains instability")
    print(f"    → If ratio << 1, gaps are large, instability is structural")

    results["cross_kmax_shifts"] = {
        "k10_k15_mean_abs_shift": float(np.abs(delta_10_15).mean()),
        "k10_k15_median_abs_shift": float(np.median(np.abs(delta_10_15))),
        "k10_k15_p99_abs_shift": float(np.percentile(np.abs(delta_10_15), 99)),
        "k15_k20_mean_abs_shift": float(np.abs(delta_15_20).mean()),
        "k15_k20_median_abs_shift": float(np.median(np.abs(delta_15_20))),
        "k15_k20_p99_abs_shift": float(np.percentile(np.abs(delta_15_20), 99)),
        "top50_mean_gap_k10": float(gap_in_top50),
        "shift_to_gap_ratio_k10_k15": float(shift_magnitude / gap_in_top50),
    }

    # Threshold-based overlap: if we use BC > X instead of top-k, how stable?
    print("\n--- Threshold-based overlap (alternative to top-k) ---")
    for threshold in [0.99, 0.95, 0.90, 0.85, 0.80]:
        set_10 = set(np.where(vals_10 > threshold)[0])
        set_15 = set(np.where(vals_15 > threshold)[0])
        set_20 = set(np.where(vals_20 > threshold)[0])

        if len(set_10) > 0 and len(set_15) > 0:
            overlap_10_15 = len(set_10 & set_15) / max(len(set_10 | set_15), 1) * 100
        else:
            overlap_10_15 = 0.0
        if len(set_15) > 0 and len(set_20) > 0:
            overlap_15_20 = len(set_15 & set_20) / max(len(set_15 | set_20), 1) * 100
        else:
            overlap_15_20 = 0.0

        print(f"  BC > {threshold}: |set_10|={len(set_10)}, |set_15|={len(set_15)}, |set_20|={len(set_20)}")
        print(f"    Jaccard overlap K10↔K15: {overlap_10_15:.1f}%")
        print(f"    Jaccard overlap K15↔K20: {overlap_15_20:.1f}%")

    del bc_10, bc_15, bc_20, vals_10, vals_15, vals_20

    return results


# ---------------------------------------------------------------------------
# Diagnostic 2: Robust-Core vs Volatile-Fringe
# ---------------------------------------------------------------------------

def diagnostic_2_robust_core():
    """Is there a stable nucleus of firm pairs that persist across all K_max?"""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: Robust-Core vs Volatile-Fringe")
    print("=" * 70)

    results = {}

    # Compute top-k sets at each K_max
    top_k_sets = {}
    for k_max in K_MAX_VALUES:
        gvkeys, bc = load_bc_matrix(k_max)
        for k in TOP_K_SIZES:
            top_k_sets[(k_max, k)] = top_k_pairs(gvkeys, bc, k)
        del bc

    # Robust pairs: appear in top-k at ALL 5 K_max values
    for k in TOP_K_SIZES:
        sets_at_k = [top_k_sets[(km, k)] for km in K_MAX_VALUES]
        robust = sets_at_k[0].intersection(*sets_at_k[1:])
        union = sets_at_k[0].union(*sets_at_k[1:])

        # Pairs appearing at exactly N K_max values
        from collections import Counter
        pair_counts = Counter()
        for s in sets_at_k:
            for pair in s:
                pair_counts[pair] += 1

        count_dist = Counter(pair_counts.values())

        print(f"\n--- Top-{k} analysis ---")
        print(f"  Robust (all 5 K_max):    {len(robust)}")
        print(f"  Total unique pairs seen:  {len(union)}")
        print(f"  Robustness ratio:         {len(robust)/k*100:.1f}% of {k}")
        print(f"  Persistence distribution:")
        for n_appearances in sorted(count_dist.keys(), reverse=True):
            n_pairs = count_dist[n_appearances]
            print(f"    Appears at {n_appearances}/5 K_max: {n_pairs} pairs ({n_pairs/len(union)*100:.1f}%)")

        results[f"top_{k}"] = {
            "robust_count": len(robust),
            "total_unique": len(union),
            "robustness_ratio_pct": round(len(robust) / k * 100, 1),
            "persistence_distribution": {str(n): count_dist.get(n, 0) for n in range(1, 6)},
        }

        # Store robust pairs for downstream use
        if k == 200:
            results["robust_200_pairs"] = [list(p) for p in robust]

    # Pairwise overlap matrix for top-200
    print("\n--- Top-200 overlap matrix (Jaccard) ---")
    overlap_matrix = {}
    for i, km_i in enumerate(K_MAX_VALUES):
        for j, km_j in enumerate(K_MAX_VALUES):
            if i <= j:
                si = top_k_sets[(km_i, 200)]
                sj = top_k_sets[(km_j, 200)]
                jaccard = len(si & sj) / len(si | sj) * 100 if len(si | sj) > 0 else 0
                overlap_matrix[f"{km_i}_vs_{km_j}"] = round(jaccard, 1)

    print(f"  {'':>6}", end="")
    for km in K_MAX_VALUES:
        print(f"  K={km:>3}", end="")
    print()
    for km_i in K_MAX_VALUES:
        print(f"  K={km_i:>3}", end="")
        for km_j in K_MAX_VALUES:
            key = f"{min(km_i,km_j)}_vs_{max(km_i,km_j)}"
            print(f"  {overlap_matrix.get(key, 0):>5.1f}%", end="")
        print()

    results["top200_overlap_matrix"] = overlap_matrix

    return results


# ---------------------------------------------------------------------------
# Diagnostic 3: Link Instability to Firm Characteristics
# ---------------------------------------------------------------------------

def diagnostic_3_firm_characteristics():
    """Are the volatile firms systematically different?"""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: Linking Instability to Firm Characteristics")
    print("=" * 70)

    results = {}

    # Load GMM results to get firm characteristics
    sys.path.insert(0, str(Path.cwd()))
    from scripts.run_kmax_sweep import load_gmm_results

    # Load gvkey_map for patent counts
    gvkey_map = pd.read_parquet("output/week2_inputs/gvkey_map.parquet")
    patent_counts = gvkey_map.groupby("gvkey").size().to_dict()

    # Load GMM results at each K_max
    gmm_results = {}
    for k_max in K_MAX_VALUES:
        path = OUTPUT_DIR / f"firm_gmm_parameters_k{k_max}.parquet"
        gmm_results[k_max] = {r["gvkey"]: r for r in load_gmm_results(str(path))}

    # Identify firms involved in top-200 pairs at each K_max
    firm_in_top200 = {}  # gvkey -> set of K_max values where firm appears in a top-200 pair
    for k_max in K_MAX_VALUES:
        gvkeys, bc = load_bc_matrix(k_max)
        top200 = top_k_pairs(gvkeys, bc, 200)
        for (a, b) in top200:
            firm_in_top200.setdefault(a, set()).add(k_max)
            firm_in_top200.setdefault(b, set()).add(k_max)
        del bc

    # Classify firms by stability
    all_top200_firms = set(firm_in_top200.keys())
    robust_firms = {f for f, kms in firm_in_top200.items() if len(kms) == 5}
    volatile_firms = {f for f, kms in firm_in_top200.items() if len(kms) < 5}
    transient_firms = {f for f, kms in firm_in_top200.items() if len(kms) == 1}

    print(f"\nFirms appearing in any top-200 pair: {len(all_top200_firms)}")
    print(f"  Robust (all 5 K_max):   {len(robust_firms)}")
    print(f"  Volatile (<5 K_max):    {len(volatile_firms)}")
    print(f"  Transient (only 1):     {len(transient_firms)}")

    # Patent count distribution for each group
    def patent_stats(firm_set, label):
        counts = [patent_counts.get(f, 0) for f in firm_set if f in patent_counts]
        if not counts:
            return {}
        counts = np.array(counts)
        print(f"\n  {label} ({len(counts)} firms):")
        print(f"    Patent count — mean: {counts.mean():.0f}, median: {np.median(counts):.0f}, "
              f"min: {counts.min()}, max: {counts.max()}")
        print(f"    P25: {np.percentile(counts, 25):.0f}, P75: {np.percentile(counts, 75):.0f}, "
              f"P90: {np.percentile(counts, 90):.0f}")
        return {
            "n_firms": len(counts),
            "mean_patents": float(counts.mean()),
            "median_patents": float(np.median(counts)),
            "min_patents": int(counts.min()),
            "max_patents": int(counts.max()),
            "p25_patents": float(np.percentile(counts, 25)),
            "p75_patents": float(np.percentile(counts, 75)),
        }

    results["robust_firm_stats"] = patent_stats(robust_firms, "Robust firms")
    results["volatile_firm_stats"] = patent_stats(volatile_firms, "Volatile firms")
    results["transient_firm_stats"] = patent_stats(transient_firms, "Transient firms (1 K_max only)")

    # Effective K analysis: do volatile firms have rising K?
    print("\n--- Effective K progression for volatile vs robust firms ---")
    for label, firm_set in [("Robust", robust_firms), ("Volatile", volatile_firms)]:
        print(f"\n  {label} firms — effective K progression:")
        k_progressions = []
        for f in firm_set:
            progression = []
            for k_max in K_MAX_VALUES:
                if f in gmm_results[k_max]:
                    progression.append(gmm_results[k_max][f]["n_components"])
                else:
                    progression.append(None)
            k_progressions.append(progression)

        k_arr = np.array([p for p in k_progressions if all(x is not None for x in p)])
        if len(k_arr) > 0:
            for i, k_max in enumerate(K_MAX_VALUES):
                print(f"    K_max={k_max}: mean K = {k_arr[:, i].mean():.1f}, "
                      f"median = {np.median(k_arr[:, i]):.0f}, "
                      f"std = {k_arr[:, i].std():.1f}")

            # K growth: how much does effective K increase from K_max=10 to K_max=30?
            delta_k = k_arr[:, -1] - k_arr[:, 0]
            print(f"    K growth (K30 - K10): mean = {delta_k.mean():.1f}, "
                  f"median = {np.median(delta_k):.0f}, max = {delta_k.max()}")

            results[f"{label.lower()}_k_progression"] = {
                str(km): {
                    "mean_k": float(k_arr[:, i].mean()),
                    "median_k": float(np.median(k_arr[:, i])),
                }
                for i, km in enumerate(K_MAX_VALUES)
            }

    # Tier distribution: are volatile firms mostly GMM-tier or single-Gaussian?
    print("\n--- Tier distribution ---")
    for label, firm_set in [("Robust", robust_firms), ("Volatile", volatile_firms)]:
        tiers = {}
        for f in firm_set:
            if f in gmm_results[10]:
                tier = gmm_results[10][f]["tier"]
                tiers[tier] = tiers.get(tier, 0) + 1
        print(f"  {label}: {tiers}")
        results[f"{label.lower()}_tiers"] = tiers

    # Deep-dive: which specific firms drive the most churn?
    print("\n--- Top-20 most volatile firms (most K_max appearances in top-200 but not all) ---")
    # Sort by number of K_max appearances (fewer = more volatile)
    churn_firms = sorted(
        [(f, len(kms), patent_counts.get(f, 0)) for f, kms in firm_in_top200.items()],
        key=lambda x: (x[1], -x[2])  # fewer appearances first, then most patents
    )[:20]
    for f, n_kmax, n_pat in churn_firms:
        tier = gmm_results[10].get(f, {}).get("tier", "?")
        k_at_each = [gmm_results[km].get(f, {}).get("n_components", "?") for km in K_MAX_VALUES]
        appears_at = sorted(firm_in_top200[f])
        print(f"  {f}: {n_pat} patents, tier={tier}, "
              f"K=[{','.join(str(k) for k in k_at_each)}], "
              f"in top-200 at K_max={appears_at}")

    results["top_20_volatile_firms"] = [
        {
            "gvkey": f,
            "n_patents": n_pat,
            "n_kmax_appearances": n_kmax,
            "tier": gmm_results[10].get(f, {}).get("tier", "?"),
        }
        for f, n_kmax, n_pat in churn_firms
    ]

    # Concentration analysis: how many firms account for most of the top-200 churn?
    print("\n--- Churn concentration ---")
    # For K_max transitions where top-200 overlap is near 0%,
    # count how many unique firms are in top-200 across all K_max values
    all_pairs_union = set()
    all_firms_union = set()
    for k_max in K_MAX_VALUES:
        gvkeys, bc = load_bc_matrix(k_max)
        top200 = top_k_pairs(gvkeys, bc, 200)
        all_pairs_union.update(top200)
        for (a, b) in top200:
            all_firms_union.add(a)
            all_firms_union.add(b)
        del bc

    print(f"  Total unique pairs in any top-200: {len(all_pairs_union)}")
    print(f"  Total unique firms in any top-200: {len(all_firms_union)}")
    print(f"  If completely different pairs at each K_max: max = {200 * 5} = 1000")
    print(f"  Pair reuse ratio: {len(all_pairs_union) / (200 * 5) * 100:.1f}%")

    results["churn_concentration"] = {
        "total_unique_pairs": len(all_pairs_union),
        "total_unique_firms": len(all_firms_union),
        "pair_reuse_ratio_pct": round(len(all_pairs_union) / (200 * 5) * 100, 1),
    }

    return results


# ---------------------------------------------------------------------------
# Diagnostic 4: BC Distribution Characterization
# ---------------------------------------------------------------------------

def diagnostic_4_bc_distribution():
    """Characterize the overall BC distribution to understand the context."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 4: BC Distribution Characterization")
    print("=" * 70)

    results = {}

    for k_max in [10, 20, 30]:  # Sample 3 K_max values
        print(f"\n--- K_max = {k_max} ---")
        gvkeys, bc = load_bc_matrix(k_max)
        vals = upper_triangle_values(bc)

        print(f"  Total pairs: {len(vals):,}")
        print(f"  Mean BC: {vals.mean():.6f}")
        print(f"  Median BC: {np.median(vals):.6f}")
        print(f"  Std BC: {vals.std():.6f}")
        print(f"  Min BC: {vals.min():.6f}")
        print(f"  Max BC: {vals.max():.6f}")

        # Percentile profile
        percentiles = [50, 75, 90, 95, 99, 99.5, 99.9, 99.95, 99.99]
        pct_vals = np.percentile(vals, percentiles)
        print(f"\n  Percentile profile:")
        for p, v in zip(percentiles, pct_vals):
            n_above = int(np.sum(vals > v))
            print(f"    P{p:>5.2f}: {v:.6f}  ({n_above:>7,} pairs above)")

        results[str(k_max)] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "std": float(vals.std()),
            "max": float(vals.max()),
            "percentiles": {str(p): float(v) for p, v in zip(percentiles, pct_vals)},
        }

        del bc, vals

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("K_max Sweep Diagnostic Analysis")
    print(f"Working directory: {Path.cwd()}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    all_results = {}

    all_results["diagnostic_1_tail_margins"] = diagnostic_1_tail_margins()
    all_results["diagnostic_2_robust_core"] = diagnostic_2_robust_core()
    all_results["diagnostic_3_firm_characteristics"] = diagnostic_3_firm_characteristics()
    all_results["diagnostic_4_bc_distribution"] = diagnostic_4_bc_distribution()

    # Save results
    output_path = OUTPUT_DIR / "diagnostic_results.json"

    # Convert sets and numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n\nDiagnostic results saved to {output_path}")

    # Print executive summary
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)

    d1 = all_results["diagnostic_1_tail_margins"]
    d2 = all_results["diagnostic_2_robust_core"]

    print("\n1. NEAR-TIE HYPOTHESIS:")
    ratio = d1.get("cross_kmax_shifts", {}).get("shift_to_gap_ratio_k10_k15", 0)
    if ratio > 5:
        print(f"   STRONGLY SUPPORTED — shift/gap ratio = {ratio:.1f}")
        print(f"   BC shifts between K_max values are {ratio:.0f}x larger than gaps between adjacent top-50 ranks.")
        print(f"   The top-50 is effectively a random draw from a dense pack of near-identical BC values.")
    elif ratio > 1:
        print(f"   SUPPORTED — shift/gap ratio = {ratio:.1f}")
        print(f"   BC shifts are larger than inter-rank gaps, explaining most of the top-50 instability.")
    else:
        print(f"   NOT SUPPORTED — shift/gap ratio = {ratio:.1f}")
        print(f"   Gaps between ranks are larger than BC shifts. Instability is structural, not near-tie noise.")

    print(f"\n2. ROBUST CORE:")
    for k in [50, 100, 200]:
        key = f"top_{k}"
        if key in d2:
            robust = d2[key]["robust_count"]
            print(f"   Top-{k}: {robust} robust pairs (persistent across all 5 K_max)")

    print(f"\n3. RECOMMENDATION:")
    if ratio > 5:
        print("   The top-k ranking instability is primarily an artifact of using fixed-rank cutoffs")
        print("   on a near-flat BC distribution in the extreme tail. The underlying firm-similarity")
        print("   landscape is stable (ρ ≈ 0.99). Week 3 should use BC thresholds, not top-k ranks,")
        print("   to identify M&A candidates — or report candidates with their BC confidence interval.")
    elif ratio > 1:
        print("   Near-ties explain most instability, but some structural sensitivity remains.")
        print("   Use both: threshold-based candidate identification AND robustness classification.")
    else:
        print("   Structural sensitivity dominates. Model-sensitive classification is essential.")


if __name__ == "__main__":
    main()
