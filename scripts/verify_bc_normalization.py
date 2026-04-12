"""Verify BC normalization fix: recompute convergence with corrected metric.

The current bc_mixture formula uses √(πᵢ·πⱼ) weighting, which is an upper
bound on the true Bhattacharyya Coefficient. This produces values > 1 for
multi-component mixtures with overlapping components, inflating with K.

Fix: Normalize by self-similarity (cosine-style):
    BC_norm(A,B) = BC_raw(A,B) / √(BC_raw(A,A) · BC_raw(B,B))

This is guaranteed ∈ [0,1] by Cauchy-Schwarz and satisfies BC_norm(A,A) = 1.

This script:
1. Computes self-BC for each firm at each K_max
2. Normalizes existing BC matrices
3. Recomputes convergence metrics
4. Reports whether convergence emerges under the corrected metric
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path.cwd()))
from scripts.run_kmax_sweep import (
    load_gmm_results,
    bc_component_matrix,
)

OUTPUT_DIR = Path("output/kmax_sweep")
K_MAX_VALUES = [10, 15, 20, 25, 30]


def compute_self_bc(result: dict) -> float:
    """Compute BC_√(A,A) = Σᵢⱼ √(πᵢπⱼ) BC(Nᵢ, Nⱼ) for a single firm."""
    bc_grid = bc_component_matrix(
        result["means"], result["covariances"],
        result["means"], result["covariances"],
    )
    weight_grid = np.sqrt(result["weights"][:, None] * result["weights"][None, :])
    return float(np.sum(weight_grid * bc_grid))


def normalize_bc_matrix(bc_raw: np.ndarray, self_bcs: np.ndarray) -> np.ndarray:
    """Normalize BC matrix by self-similarities."""
    # Outer product of sqrt(self_bcs)
    norm_factor = np.sqrt(np.outer(self_bcs, self_bcs))
    # Avoid division by zero
    norm_factor = np.maximum(norm_factor, 1e-10)
    bc_norm = bc_raw / norm_factor
    # Clamp to [0, 1] for numerical safety
    np.clip(bc_norm, 0.0, 1.0, out=bc_norm)
    return bc_norm


def compute_convergence(gvkeys_a, bc_a, gvkeys_b, bc_b, k_max_a, k_max_b):
    """Simplified convergence metrics between two BC matrices."""
    assert gvkeys_a == gvkeys_b, "Gvkey ordering mismatch"
    n = len(gvkeys_a)
    idx = np.triu_indices(n, k=1)
    vals_a = bc_a[idx]
    vals_b = bc_b[idx]

    # Spearman
    rho, p = stats.spearmanr(vals_a, vals_b)

    # Top-k overlap
    overlaps = {}
    for k in [50, 100, 200]:
        top_a = set(np.argpartition(vals_a, -k)[-k:])
        top_b = set(np.argpartition(vals_b, -k)[-k:])
        overlaps[k] = len(top_a & top_b) / k * 100

    # Per-firm NN-5 stability
    nn_overlaps = []
    for i in range(n):
        row_a = bc_a[i].copy()
        row_b = bc_b[i].copy()
        row_a[i] = -1  # exclude self
        row_b[i] = -1
        nn_a = set(np.argpartition(row_a, -5)[-5:])
        nn_b = set(np.argpartition(row_b, -5)[-5:])
        nn_overlaps.append(len(nn_a & nn_b) / 5 * 100)
    nn_arr = np.array(nn_overlaps)

    return {
        "k_max_a": k_max_a,
        "k_max_b": k_max_b,
        "spearman_rho": round(float(rho), 4),
        "top_50_overlap_pct": round(overlaps[50], 1),
        "top_100_overlap_pct": round(overlaps[100], 1),
        "top_200_overlap_pct": round(overlaps[200], 1),
        "mean_nn5_overlap_pct": round(float(nn_arr.mean()), 1),
        "median_nn5_overlap_pct": round(float(np.median(nn_arr)), 1),
        "p10_nn5_overlap_pct": round(float(np.percentile(nn_arr, 10)), 1),
    }


def main():
    print("=" * 70)
    print("BC NORMALIZATION VERIFICATION")
    print("=" * 70)

    # Step 1: Compute self-BC for each firm at each K_max
    print("\nStep 1: Computing self-BC for each firm at each K_max...")
    self_bcs = {}
    gvkey_order = {}
    for k_max in K_MAX_VALUES:
        print(f"  Loading GMM results for K_max={k_max}...")
        results = load_gmm_results(str(OUTPUT_DIR / f"firm_gmm_parameters_k{k_max}.parquet"))
        results_by_gvkey = {r["gvkey"]: r for r in results}

        # Load BC matrix to get gvkey ordering
        data = np.load(OUTPUT_DIR / f"bc_matrix_all_k{k_max}.npz", allow_pickle=True)
        gvkeys = list(data["gvkeys"])
        gvkey_order[k_max] = gvkeys

        # Compute self-BC for each firm in the matrix ordering
        firm_self_bcs = np.ones(len(gvkeys))
        for i, gvkey in enumerate(gvkeys):
            if gvkey in results_by_gvkey:
                firm_self_bcs[i] = compute_self_bc(results_by_gvkey[gvkey])
            # else: firm not in GMM results (shouldn't happen), leave as 1.0

        self_bcs[k_max] = firm_self_bcs

        # Report self-BC distribution
        print(f"    K_max={k_max}: self-BC mean={firm_self_bcs.mean():.4f}, "
              f"median={np.median(firm_self_bcs):.4f}, "
              f"max={firm_self_bcs.max():.4f}, "
              f"min={firm_self_bcs.min():.4f}, "
              f"n_above_1={int(np.sum(firm_self_bcs > 1.01))}")

        del data

    # Step 2: Normalize BC matrices and recompute convergence
    print("\nStep 2: Normalizing BC matrices and computing convergence...")

    # Also compute raw vs normalized tail statistics
    for k_max in K_MAX_VALUES:
        data = np.load(OUTPUT_DIR / f"bc_matrix_all_k{k_max}.npz", allow_pickle=True)
        bc_raw = data["bc_matrix"]
        bc_norm = normalize_bc_matrix(bc_raw, self_bcs[k_max])

        idx = np.triu_indices_from(bc_raw, k=1)
        raw_vals = np.sort(bc_raw[idx])[::-1]
        norm_vals = np.sort(bc_norm[idx])[::-1]

        print(f"\n  K_max={k_max}:")
        print(f"    Raw:  rank1={raw_vals[0]:.4f}, rank50={raw_vals[49]:.4f}, "
              f"rank200={raw_vals[199]:.4f}, max={raw_vals[0]:.4f}")
        print(f"    Norm: rank1={norm_vals[0]:.6f}, rank50={norm_vals[49]:.6f}, "
              f"rank200={norm_vals[199]:.6f}, max={norm_vals[0]:.6f}")

        # How tightly packed is the normalized top-50?
        top50_norm = norm_vals[:50]
        gaps = np.abs(np.diff(top50_norm))
        print(f"    Norm top-50 span: {top50_norm[0] - top50_norm[-1]:.8f}")
        print(f"    Norm top-50 mean gap: {gaps.mean():.8f}")
        print(f"    Norm top-50 median gap: {np.median(gaps):.8f}")

        del bc_raw, data
        np.savez_compressed(
            OUTPUT_DIR / f"bc_matrix_norm_k{k_max}.npz",
            bc_matrix=bc_norm,
            gvkeys=np.array(gvkey_order[k_max]),
        )
        if k_max == K_MAX_VALUES[0]:
            bc_prev_norm = bc_norm
            gvkeys_prev = gvkey_order[k_max]
            k_prev = k_max
        else:
            # Convergence between adjacent K_max values
            metrics = compute_convergence(
                gvkeys_prev, bc_prev_norm,
                gvkey_order[k_max], bc_norm,
                k_prev, k_max,
            )
            print(f"\n  Convergence K={k_prev}→{k_max} (NORMALIZED):")
            print(f"    Spearman ρ:       {metrics['spearman_rho']}")
            print(f"    Top-50 overlap:   {metrics['top_50_overlap_pct']}%")
            print(f"    Top-100 overlap:  {metrics['top_100_overlap_pct']}%")
            print(f"    Top-200 overlap:  {metrics['top_200_overlap_pct']}%")
            print(f"    Mean NN-5:        {metrics['mean_nn5_overlap_pct']}%")
            print(f"    Median NN-5:      {metrics['median_nn5_overlap_pct']}%")
            print(f"    P10 NN-5:         {metrics['p10_nn5_overlap_pct']}%")

            # Check against thresholds
            passes = (metrics['spearman_rho'] > 0.95 and
                      metrics['top_50_overlap_pct'] > 80)
            print(f"    PASSES THRESHOLD: {'YES' if passes else 'NO'}")

            bc_prev_norm = bc_norm
            gvkeys_prev = gvkey_order[k_max]
            k_prev = k_max

    # Step 3: Non-adjacent comparison (K=10 vs K=30)
    print("\n\nStep 3: Non-adjacent comparison K=10 vs K=30 (NORMALIZED)...")
    data_10 = np.load(OUTPUT_DIR / "bc_matrix_norm_k10.npz", allow_pickle=True)
    data_30 = np.load(OUTPUT_DIR / "bc_matrix_norm_k30.npz", allow_pickle=True)
    metrics_10_30 = compute_convergence(
        list(data_10["gvkeys"]), data_10["bc_matrix"],
        list(data_30["gvkeys"]), data_30["bc_matrix"],
        10, 30,
    )
    print(f"  Spearman ρ:       {metrics_10_30['spearman_rho']}")
    print(f"  Top-50 overlap:   {metrics_10_30['top_50_overlap_pct']}%")
    print(f"  Top-100 overlap:  {metrics_10_30['top_100_overlap_pct']}%")
    print(f"  Top-200 overlap:  {metrics_10_30['top_200_overlap_pct']}%")
    print(f"  Mean NN-5:        {metrics_10_30['mean_nn5_overlap_pct']}%")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print("\nIf normalized convergence passes at all transitions → the instability")
    print("was entirely caused by metric inflation, and the corrected metric is")
    print("suitable for production use.")
    print("\nIf it still fails → there is genuine structural sensitivity beyond")
    print("the metric issue, and both findings must be reported.")


if __name__ == "__main__":
    main()
