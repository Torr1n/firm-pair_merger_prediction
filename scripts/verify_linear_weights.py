"""Quick verification: linear-weighted BC (πᵢπⱼ instead of √(πᵢπⱼ)).

This is a fast check that only computes on a sample of firms to compare
the two weighting schemes' behavior.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from scripts.run_kmax_sweep import load_gmm_results, bc_component_matrix

OUTPUT_DIR = Path("output/kmax_sweep")


def bc_mixture_linear(gmm_a, gmm_b):
    """BC with linear weights: πᵢ·πⱼ (bounded in [0,1])."""
    bc_grid = bc_component_matrix(
        gmm_a["means"], gmm_a["covariances"],
        gmm_b["means"], gmm_b["covariances"],
    )
    weight_grid = gmm_a["weights"][:, None] * gmm_b["weights"][None, :]
    return float(np.sum(weight_grid * bc_grid))


def bc_mixture_sqrt(gmm_a, gmm_b):
    """BC with sqrt weights: √(πᵢ·πⱼ) (current formula, unbounded)."""
    bc_grid = bc_component_matrix(
        gmm_a["means"], gmm_a["covariances"],
        gmm_b["means"], gmm_b["covariances"],
    )
    weight_grid = np.sqrt(gmm_a["weights"][:, None] * gmm_b["weights"][None, :])
    return float(np.sum(weight_grid * bc_grid))


def main():
    # Load results at two K_max values
    for k_max in [10, 20, 30]:
        print(f"\n{'='*60}")
        print(f"K_max = {k_max}")
        print(f"{'='*60}")
        results = load_gmm_results(str(OUTPUT_DIR / f"firm_gmm_parameters_k{k_max}.parquet"))

        # Separate by tier
        sg_firms = [r for r in results if r["tier"] == "single_gaussian"]
        gmm_firms = [r for r in results if r["tier"] == "gmm"]

        print(f"  SG firms: {len(sg_firms)}, GMM firms: {len(gmm_firms)}")

        # Sample pairs: 50 SG-SG, 50 GMM-GMM, 50 SG-GMM
        rng = np.random.RandomState(42)

        def sample_pairs(list_a, list_b, n=50):
            pairs = []
            for _ in range(n):
                a = list_a[rng.randint(len(list_a))]
                b = list_b[rng.randint(len(list_b))]
                pairs.append((a, b))
            return pairs

        for label, list_a, list_b in [
            ("SG-SG", sg_firms, sg_firms),
            ("GMM-GMM", gmm_firms, gmm_firms),
            ("SG-GMM", sg_firms, gmm_firms),
        ]:
            pairs = sample_pairs(list_a, list_b, 200)
            sqrt_vals = [bc_mixture_sqrt(a, b) for a, b in pairs]
            linear_vals = [bc_mixture_linear(a, b) for a, b in pairs]

            sqrt_arr = np.array(sqrt_vals)
            linear_arr = np.array(linear_vals)

            print(f"\n  {label} ({len(pairs)} random pairs):")
            print(f"    √-weighted: mean={sqrt_arr.mean():.6f}, max={sqrt_arr.max():.6f}, "
                  f">1.0: {np.sum(sqrt_arr > 1.0)}")
            print(f"    Linear:     mean={linear_arr.mean():.6f}, max={linear_arr.max():.6f}, "
                  f">1.0: {np.sum(linear_arr > 1.0)}")

            # For the highest-BC pairs, compare the two
            top_sqrt = np.argsort(sqrt_arr)[-5:]
            print(f"    Top-5 by √-weight:")
            for idx in reversed(top_sqrt):
                a, b = pairs[idx]
                print(f"      {a['gvkey']}(K={a['n_components']}) vs "
                      f"{b['gvkey']}(K={b['n_components']}): "
                      f"√={sqrt_vals[idx]:.4f}, L={linear_vals[idx]:.4f}")

        # Check: how many SG-SG pairs have BC within 1e-4 of 1.0?
        sg_sample = sample_pairs(sg_firms, sg_firms, 2000)
        sg_bcs = np.array([bc_mixture_sqrt(a, b) for a, b in sg_sample])
        n_near_1 = np.sum(sg_bcs > 0.9999)
        n_above_09 = np.sum(sg_bcs > 0.9)
        n_above_05 = np.sum(sg_bcs > 0.5)
        print(f"\n  SG-SG (2000 random): {n_near_1} at BC>0.9999, "
              f"{n_above_09} at BC>0.9, {n_above_05} at BC>0.5")


if __name__ == "__main__":
    main()
