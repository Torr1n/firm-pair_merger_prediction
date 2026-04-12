"""Recompute BC matrices with deduplication and corrected linear-weight formula.

This is a self-contained recomputation script that:
1. Loads the saved GMM parameters from the original sweep (no re-fitting)
2. Filters to deduplicated firms (per `output/kmax_sweep/deduplication_decisions.csv`)
3. Computes BC matrices using LINEAR weights (πᵢπⱼ instead of √(πᵢπⱼ))
4. Recomputes convergence metrics (Spearman ρ, top-k overlap, NN-5 stability)
5. Saves new BC matrices and convergence summary

Optimizations:
- SG block caching: single-Gaussian (K=1) pairs are K_max-invariant. Compute once, reuse.
- Vectorized component-pair BC via bc_component_matrix.

The fix to the BC formula is in `bc_mixture_linear` below. The original
`bc_mixture` in `run_kmax_sweep.py` uses √(πᵢπⱼ) which is mathematically
an upper bound on the true Bhattacharyya Coefficient and exceeds 1.0 for
multi-component mixtures. The linear-weighted version is bounded in [0,1]
and aligns with Arthur's methodology ("aggregate using GMM weights").

Designed to be runnable on AWS VM (c5.4xlarge) or locally.
Estimated runtime: 60-120 minutes on c5.4xlarge for the full sweep.

Usage:
    # Quick test on 200-firm sample (~2 minutes)
    python scripts/recompute_bc_corrected.py --test-mode

    # Full recomputation
    python scripts/recompute_bc_corrected.py

    # Single K_max only
    python scripts/recompute_bc_corrected.py --k-max-values 10

    # Recompute convergence from existing BC matrices
    python scripts/recompute_bc_corrected.py --skip-bc
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Reuse the (correct) component-pair BC and (correct) GMM result loader
sys.path.insert(0, str(Path.cwd()))
from scripts.run_kmax_sweep import load_gmm_results, bc_component_matrix

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output/kmax_sweep")
K_MAX_VALUES = [10, 15, 20, 25, 30]
DEDUP_DECISIONS_PATH = OUTPUT_DIR / "deduplication_decisions.parquet"

# Output filename suffix to distinguish from original (buggy) BC matrices
SUFFIX = "_dedup_linear"

# ---------------------------------------------------------------------------
# Corrected BC formula (linear weights)
# ---------------------------------------------------------------------------

def bc_mixture_linear(gmm_a: dict, gmm_b: dict) -> float:
    """BC between two GMMs using LINEAR weights (πᵢ·πⱼ).

    This is bounded in [0, 1] and represents the expected Bhattacharyya
    coefficient between a randomly chosen technology area from each firm.

    Formula:
        BC(A, B) = Σᵢ Σⱼ πᵢᴬ · πⱼᴮ · BC(Nᵢᴬ, Nⱼᴮ)

    Properties:
    - BC(A, A) = Σᵢⱼ πᵢπⱼ · BC(Nᵢ, Nⱼ) ≤ 1 (Cauchy-Schwarz)
    - K-invariant: doesn't inflate as K increases
    - Symmetric: BC(A, B) = BC(B, A)
    """
    bc_grid = bc_component_matrix(
        gmm_a["means"], gmm_a["covariances"],
        gmm_b["means"], gmm_b["covariances"],
    )
    weight_grid = gmm_a["weights"][:, None] * gmm_b["weights"][None, :]
    return float(np.sum(weight_grid * bc_grid))


def compute_sg_block_linear(results: list[dict]) -> tuple[list[str], np.ndarray]:
    """Precompute the SG-vs-SG BC block (K_max-invariant) using linear weights.

    For K=1 firms, weights are [1.0], so:
    BC_linear(SG, SG) = 1.0 * 1.0 * BC(N_a, N_b) = BC(N_a, N_b)

    The bc_component_matrix already computes this correctly. For SG firms with
    one component each, we can vectorize across all pairs by stacking their
    means and variances and using broadcasting on the chunked computation.
    """
    sg_firms = [r for r in results if r["tier"] == "single_gaussian"]
    n = len(sg_firms)
    gvkeys = [r["gvkey"] for r in sg_firms]
    block = np.zeros((n, n), dtype=np.float64)

    if n == 0:
        return gvkeys, block

    # Stack all SG means and variances: (n, D)
    means = np.vstack([r["means"][0] for r in sg_firms])  # (n, D)
    vars_ = np.vstack([r["covariances"][0] for r in sg_firms])  # (n, D)

    print(f"    SG block: {n} firms, {n*(n-1)//2:,} pairs")
    t0 = time.time()

    # Process in chunks to manage memory
    chunk_size = 200
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        chunk_means = means[i_start:i_end]  # (chunk, D)
        chunk_vars = vars_[i_start:i_end]

        # Compute BC for chunk-vs-all using vectorized component_matrix
        # bc_component_matrix takes (K_a, D), (K_a, D), (K_b, D), (K_b, D)
        # We treat each SG firm as K=1
        # Reshape chunk_means to (chunk, 1, D), means to (1, n, D) won't work
        # because bc_component_matrix expects K_a × D and K_b × D matrices.
        # Instead, compute pairwise via broadcasting manually:
        # diff: (chunk, n, D)
        sigma_avg = (chunk_vars[:, None, :] + vars_[None, :, :]) / 2.0
        diff = chunk_means[:, None, :] - means[None, :, :]
        mahal = 0.125 * np.sum(diff**2 / sigma_avg, axis=2)  # (chunk, n)
        log_det_avg = np.sum(np.log(sigma_avg), axis=2)  # (chunk, n)
        log_det_a = np.sum(np.log(chunk_vars), axis=1)  # (chunk,)
        log_det_b = np.sum(np.log(vars_), axis=1)  # (n,)
        det_term = 0.5 * (log_det_avg - 0.5 * (log_det_a[:, None] + log_det_b[None, :]))
        bc_chunk = np.exp(-(mahal + det_term))  # (chunk, n)

        block[i_start:i_end, :] = bc_chunk

        if (i_start // chunk_size) % 10 == 0:
            elapsed = time.time() - t0
            pct = (i_end / n) * 100
            print(f"      SG progress: {pct:>5.1f}% ({i_end}/{n}) elapsed={elapsed:.0f}s")

    # Diagonal should be 1.0
    np.fill_diagonal(block, 1.0)

    elapsed = time.time() - t0
    print(f"    SG block done in {elapsed:.0f}s")
    return gvkeys, block


def compute_bc_matrix_linear(
    results: list[dict],
    label: str = "all",
    sg_block: np.ndarray | None = None,
    sg_gvkeys: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Compute pairwise BC matrix using linear weights, with SG block caching."""
    n = len(results)
    gvkeys = [r["gvkey"] for r in results]
    bc_matrix = np.zeros((n, n), dtype=np.float64)

    sg_idx_map = {}
    if sg_block is not None and sg_gvkeys is not None:
        sg_idx_map = {gk: idx for idx, gk in enumerate(sg_gvkeys)}

    is_sg = [r["tier"] == "single_gaussian" for r in results]

    total_pairs = n * (n - 1) // 2
    computed = 0
    skipped = 0
    t0 = time.time()

    for i in range(n):
        bc_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            if (is_sg[i] and is_sg[j] and
                gvkeys[i] in sg_idx_map and gvkeys[j] in sg_idx_map):
                bc_val = sg_block[sg_idx_map[gvkeys[i]], sg_idx_map[gvkeys[j]]]
                skipped += 1
            else:
                bc_val = bc_mixture_linear(results[i], results[j])
                computed += 1

            bc_matrix[i, j] = bc_val
            bc_matrix[j, i] = bc_val

        if (i + 1) % 250 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            done = computed + skipped
            pct = done / total_pairs * 100
            rate = computed / elapsed if elapsed > 0 else 0
            print(f"    [{label}] row {i+1}/{n}  computed={computed:,} cached={skipped:,} "
                  f"({pct:.1f}%)  {elapsed:.0f}s  ({rate:.0f} pairs/s computed)")

    return gvkeys, bc_matrix


# ---------------------------------------------------------------------------
# Convergence metrics
# ---------------------------------------------------------------------------

def compute_convergence_metrics(
    gvkeys_a: list[str], bc_a: np.ndarray,
    gvkeys_b: list[str], bc_b: np.ndarray,
    k_max_a: int, k_max_b: int,
) -> dict:
    """Compare BC ranking stability between two K_max settings."""
    assert gvkeys_a == gvkeys_b, "Gvkey ordering must match between matrices"
    n = len(gvkeys_a)
    idx = np.triu_indices(n, k=1)
    vals_a = bc_a[idx]
    vals_b = bc_b[idx]

    rho, p_rho = stats.spearmanr(vals_a, vals_b)

    overlaps = {}
    for k in [50, 100, 200]:
        if k > len(vals_a):
            continue
        top_a = set(np.argpartition(vals_a, -k)[-k:])
        top_b = set(np.argpartition(vals_b, -k)[-k:])
        overlaps[k] = len(top_a & top_b) / k * 100

    nn_overlaps = []
    for i in range(n):
        row_a = bc_a[i].copy()
        row_b = bc_b[i].copy()
        row_a[i] = -1
        row_b[i] = -1
        nn_a = set(np.argpartition(row_a, -5)[-5:])
        nn_b = set(np.argpartition(row_b, -5)[-5:])
        nn_overlaps.append(len(nn_a & nn_b) / 5 * 100)
    nn_arr = np.array(nn_overlaps)

    return {
        "k_max_a": k_max_a,
        "k_max_b": k_max_b,
        "n_firms": n,
        "n_pairs": int(n * (n - 1) / 2),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(p_rho),
        "top_50_overlap_pct": round(overlaps.get(50, 0), 1),
        "top_100_overlap_pct": round(overlaps.get(100, 0), 1),
        "top_200_overlap_pct": round(overlaps.get(200, 0), 1),
        "mean_nn5_overlap_pct": round(float(nn_arr.mean()), 1),
        "median_nn5_overlap_pct": round(float(np.median(nn_arr)), 1),
        "p10_nn5_overlap_pct": round(float(np.percentile(nn_arr, 10)), 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-max-values", nargs="+", type=int, default=K_MAX_VALUES)
    parser.add_argument("--skip-bc", action="store_true",
                        help="Skip BC matrix computation, use existing files")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run on a 200-firm sample for validation")
    parser.add_argument("--test-size", type=int, default=100,
                        help="Per-tier sample size in test mode (default 100)")
    args = parser.parse_args()

    print("=" * 70)
    print("BC RECOMPUTATION (deduplicated + linear-weighted)")
    print("=" * 70)
    print(f"K_max values: {args.k_max_values}")
    print(f"Output suffix: {SUFFIX}{'_test' if args.test_mode else ''}")

    test_suffix = "_test" if args.test_mode else ""
    final_suffix = SUFFIX + test_suffix

    # Step 1: Load deduplication decisions
    print("\nStep 1: Loading deduplication decisions...")
    if not DEDUP_DECISIONS_PATH.exists():
        print(f"ERROR: {DEDUP_DECISIONS_PATH} not found.")
        sys.exit(1)
    dedup_df = pd.read_parquet(DEDUP_DECISIONS_PATH)
    dropped_firms = set(dedup_df["dropped"].tolist())
    print(f"  {len(dropped_firms)} firms to drop")

    # Step 2: Recompute BC for each K_max
    bc_matrices = {}
    sg_block_cache = None
    sg_gvkeys_cache = None

    for k_max in args.k_max_values:
        print(f"\n--- K_max = {k_max} ---")
        results = load_gmm_results(str(OUTPUT_DIR / f"firm_gmm_parameters_k{k_max}.parquet"))
        print(f"  Loaded {len(results)} firm GMM results")

        results_filtered = [r for r in results if r["gvkey"] not in dropped_firms]
        results_filtered = sorted(results_filtered, key=lambda r: r["gvkey"])
        print(f"  After deduplication: {len(results_filtered)} firms")

        # Test mode: subsample (use deterministic stratified sample)
        if args.test_mode:
            sg = [r for r in results_filtered if r["tier"] == "single_gaussian"][:args.test_size]
            gm = [r for r in results_filtered if r["tier"] == "gmm"][:args.test_size]
            results_filtered = sorted(sg + gm, key=lambda r: r["gvkey"])
            print(f"  TEST MODE: subsampled to {len(results_filtered)} firms "
                  f"({len(sg)} SG + {len(gm)} GMM)")
            sg_block_cache = None  # Recompute SG block for new sample

        # Skip mode: load existing BC matrix
        if args.skip_bc:
            bc_path = OUTPUT_DIR / f"bc_matrix_all_k{k_max}{final_suffix}.npz"
            if bc_path.exists():
                print(f"  Loading existing BC matrix from {bc_path}")
                data = np.load(bc_path, allow_pickle=True)
                bc_matrices[k_max] = (list(data["gvkeys"]), data["bc_matrix"])
                continue

        # Compute SG block once (K_max-invariant)
        if sg_block_cache is None:
            sg_only = [r for r in results_filtered if r["tier"] == "single_gaussian"]
            print(f"  Computing SG block ({len(sg_only)} firms, K_max-invariant)...")
            sg_gvkeys_cache, sg_block_cache = compute_sg_block_linear(sg_only)
            # Save SG block separately
            sg_path = OUTPUT_DIR / f"bc_block_sg_vs_sg{final_suffix}.npz"
            np.savez_compressed(sg_path, bc_matrix=sg_block_cache, gvkeys=np.array(sg_gvkeys_cache))
            print(f"  SG block saved to {sg_path}")

        # Compute full BC matrix (with SG cache reuse)
        print(f"  Computing full BC matrix...")
        gvkeys, bc = compute_bc_matrix_linear(
            results_filtered,
            label=f"K{k_max}",
            sg_block=sg_block_cache,
            sg_gvkeys=sg_gvkeys_cache,
        )

        out_path = OUTPUT_DIR / f"bc_matrix_all_k{k_max}{final_suffix}.npz"
        np.savez_compressed(out_path, bc_matrix=bc, gvkeys=np.array(gvkeys))
        print(f"  Saved to {out_path}")

        bc_matrices[k_max] = (gvkeys, bc)

        # Sanity check
        upper_vals = bc[np.triu_indices_from(bc, k=1)]
        max_bc = upper_vals.max()
        n_above_1 = int(np.sum(upper_vals > 1.0001))
        n_above_05 = int(np.sum(upper_vals > 0.5))
        print(f"  Sanity: max BC={max_bc:.6f}, n>1.0001={n_above_1}, n>0.5={n_above_05}")

    # Step 3: Convergence metrics
    print("\n" + "=" * 70)
    print("CONVERGENCE METRICS (deduplicated + linear-weighted)")
    print("=" * 70)

    if len(bc_matrices) < 2:
        return

    sorted_kmax = sorted(bc_matrices.keys())
    adjacent_metrics = []
    non_adjacent_metrics = []

    print("\nAdjacent comparisons:")
    for i in range(len(sorted_kmax) - 1):
        k_a, k_b = sorted_kmax[i], sorted_kmax[i + 1]
        gvkeys_a, bc_a = bc_matrices[k_a]
        gvkeys_b, bc_b = bc_matrices[k_b]
        m = compute_convergence_metrics(gvkeys_a, bc_a, gvkeys_b, bc_b, k_a, k_b)
        adjacent_metrics.append(m)
        print(f"\n  K={k_a}→K={k_b}:")
        print(f"    Spearman ρ:       {m['spearman_rho']}")
        print(f"    Top-50 overlap:   {m['top_50_overlap_pct']}%")
        print(f"    Top-100 overlap:  {m['top_100_overlap_pct']}%")
        print(f"    Top-200 overlap:  {m['top_200_overlap_pct']}%")
        print(f"    Mean NN-5:        {m['mean_nn5_overlap_pct']}%")
        print(f"    Median NN-5:      {m['median_nn5_overlap_pct']}%")
        print(f"    P10 NN-5:         {m['p10_nn5_overlap_pct']}%")
        passes = (m["spearman_rho"] > 0.95 and m["top_50_overlap_pct"] > 80)
        print(f"    PASSES THRESHOLD: {'YES' if passes else 'NO'}")

    if 10 in bc_matrices and 30 in bc_matrices:
        gvkeys_10, bc_10 = bc_matrices[10]
        gvkeys_30, bc_30 = bc_matrices[30]
        m = compute_convergence_metrics(gvkeys_10, bc_10, gvkeys_30, bc_30, 10, 30)
        non_adjacent_metrics.append(m)
        print(f"\n  Non-adjacent K=10→K=30:")
        print(f"    Spearman ρ:       {m['spearman_rho']}")
        print(f"    Top-50 overlap:   {m['top_50_overlap_pct']}%")
        print(f"    Mean NN-5:        {m['mean_nn5_overlap_pct']}%")

    converged_at = None
    for i, m in enumerate(adjacent_metrics):
        if all(mm["spearman_rho"] > 0.95 and mm["top_50_overlap_pct"] > 80
               for mm in adjacent_metrics[i:]):
            converged_at = m["k_max_a"]
            break

    print("\n" + "=" * 70)
    print(f"VERDICT: {'CONVERGED at K_max=' + str(converged_at) if converged_at else 'NOT CONVERGED'}")
    print("=" * 70)

    summary = {
        "method": "deduplicated + linear-weighted BC",
        "k_max_values": args.k_max_values,
        "n_firms_after_dedup": len(bc_matrices[sorted_kmax[0]][0]),
        "convergence_verdict": "converged" if converged_at else "not_converged",
        "converged_at_kmax": converged_at,
        "decision_rule": {
            "spearman_threshold": 0.95,
            "top_50_overlap_threshold_pct": 80,
            "method": "persistent_stability",
        },
        "adjacent_comparisons": adjacent_metrics,
        "non_adjacent_comparisons": non_adjacent_metrics,
    }
    summary_path = OUTPUT_DIR / f"convergence_summary{final_suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
