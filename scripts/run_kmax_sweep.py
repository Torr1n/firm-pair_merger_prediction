"""K_max convergence sweep for firm GMM portfolio fitting.

Fits Bayesian GMMs for all GMM-tier firms (50+ patents) at K_max ∈ {10, 15, 20, 25, 30},
computes pairwise Bhattacharyya Coefficients, and measures BC ranking convergence
across adjacent K_max values.

Why this matters:
    The design phase (ADR-004) found BC rankings moderately stable (Spearman ρ ≈ 0.80)
    but top-tail unstable (top-50 overlap 22-48%). This sweep determines whether rankings
    converge by K_max=25-30, resolving whether K_max is a tuning parameter or a
    methodological limitation requiring model-family escalation.

Decision rule (from Codex review):
    - If Spearman ρ > 0.95 AND top-50 overlap > 80% between adjacent K_max → converged
    - If convergence does not emerge by K_max=30 → escalate (reopen ADR-004)

Usage (on AWS c5.4xlarge or equivalent CPU instance):
    source venv/bin/activate
    python scripts/run_kmax_sweep.py 2>&1 | tee output/kmax_sweep/sweep.log

    # Optional: use local data instead of S3
    python scripts/run_kmax_sweep.py --local 2>&1 | tee output/kmax_sweep/sweep.log

After completion:
    aws s3 sync output/kmax_sweep/ \\
        s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/ --profile torrin

Checkpoint behavior:
    - GMM results are saved per K_max value after all firms are fitted at that K_max
    - If interrupted, re-run to resume from the last completed K_max
    - BC matrices and convergence metrics are computed after all K_max values are fitted
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kendalltau
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output/kmax_sweep"
WEIGHT_PRUNE_THRESHOLD = 0.01  # Components below this weight are pruned


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_inputs(config: dict, local: bool = False) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Load 50D patent vectors and gvkey mapping.

    Uses local files in output/week2_inputs/ (downloaded from S3 by Codex
    during instance setup, or already present from Week 1).

    Returns:
        vectors: (N, 50) float32 array of patent vectors
        gvkey_map: DataFrame with columns [patent_id, gvkey]
    """
    if local:
        vec_path = "output/week2_inputs/patent_vectors_50d.parquet"
        map_path = "output/week2_inputs/gvkey_map.parquet"
    else:
        # On AWS: Codex copies S3 data to these paths during bootstrap
        vec_path = config["output"]["patent_vectors_50d"]
        map_path = config["output"]["gvkey_map"]

    print(f"  Loading vectors from {vec_path}")
    vec_table = pq.read_table(vec_path)
    patent_ids_vec = vec_table["patent_id"].to_pylist()
    vectors = np.array(
        [np.frombuffer(b, dtype=np.float32) for b in vec_table["embedding"].to_pylist()]
    )
    print(f"  Vectors: {vectors.shape} ({vectors.nbytes / 1e6:.0f} MB)")

    print(f"  Loading gvkey map from {map_path}")
    gvkey_map = pd.read_parquet(map_path)
    print(f"  Gvkey map: {len(gvkey_map):,} rows, "
          f"{gvkey_map['gvkey'].nunique():,} firms")

    # Build patent_id → row index for fast vector lookup
    pid_to_idx = {pid: i for i, pid in enumerate(patent_ids_vec)}

    return vectors, gvkey_map, pid_to_idx


# ---------------------------------------------------------------------------
# Firm grouping and tier classification (per ADR-005)
# ---------------------------------------------------------------------------

def group_and_classify(
    vectors: np.ndarray,
    gvkey_map: pd.DataFrame,
    pid_to_idx: dict,
    min_patents: int = 5,
    single_gaussian_max: int = 49,
) -> tuple[dict, dict, list[tuple[str, int]]]:
    """Group patent vectors by firm and classify into tiers.

    Tiers (ADR-005):
        - exclude: n < min_patents (dropped)
        - single_gaussian: min_patents <= n <= single_gaussian_max
        - gmm: n > single_gaussian_max

    Returns:
        firm_vectors: dict[gvkey -> (n_patents, 50) array]
        tier_assignments: dict[gvkey -> tier_str] (only non-excluded firms)
        excluded_firms: list of (gvkey, n_patents) tuples for excluded firms
    """
    # Group patent_ids by gvkey (co-assigned patents appear in each firm)
    grouped = gvkey_map.groupby("gvkey")["patent_id"].apply(list)

    firm_vectors = {}
    tier_assignments = {}
    excluded_firms = []

    for gvkey, patent_ids in grouped.items():
        # Look up vector indices; skip patents not in the vector set
        indices = [pid_to_idx[pid] for pid in patent_ids if pid in pid_to_idx]
        n = len(indices)

        if n < min_patents:
            excluded_firms.append((gvkey, n))
            continue

        firm_vectors[gvkey] = vectors[indices]

        if n <= single_gaussian_max:
            tier_assignments[gvkey] = "single_gaussian"
        else:
            tier_assignments[gvkey] = "gmm"

    return firm_vectors, tier_assignments, excluded_firms


# ---------------------------------------------------------------------------
# Global empirical Bayes priors (from unique patent matrix — not grouped)
# ---------------------------------------------------------------------------

def compute_global_priors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute global mean and variance from the full unique patent matrix.

    Why from unique patents, not grouped firm vectors:
        Co-assigned patents (80,687 in production) would be duplicated if we
        concatenated per-firm arrays. Computing from the unique patent matrix
        avoids double-counting. (Codex Major finding #3, design review.)

    Returns:
        global_mean: shape (50,)
        global_var: shape (50,)
    """
    global_mean = np.mean(vectors, axis=0)  # shape (50,)
    global_var = np.var(vectors, axis=0)    # shape (50,)
    return global_mean, global_var


# ---------------------------------------------------------------------------
# GMM fitting
# ---------------------------------------------------------------------------

def fit_single_gaussian(gvkey: str, X: np.ndarray) -> dict:
    """Fit K=1 Gaussian for single-Gaussian tier firms (5-49 patents)."""
    gm = GaussianMixture(
        n_components=1,
        covariance_type="diag",
        max_iter=200,
        random_state=42,
        reg_covar=1e-6,
    )
    gm.fit(X)
    return {
        "gvkey": gvkey,
        "n_patents": len(X),
        "n_components": 1,
        "tier": "single_gaussian",
        "covariance_type": "diagonal",
        "means": gm.means_.astype(np.float64),              # (1, 50)
        "covariances": gm.covariances_.astype(np.float64),   # (1, 50)
        "weights": gm.weights_.astype(np.float64),           # (1,)
        "converged": gm.converged_,
        "lower_bound": float(gm.lower_bound_),
        "n_iter": gm.n_iter_,
    }


def fit_bayesian_gmm(
    gvkey: str,
    X: np.ndarray,
    k_max: int,
    global_mean: np.ndarray,
    global_var: np.ndarray,
    config: dict,
) -> dict:
    """Fit Bayesian GMM with Dirichlet Process prior for GMM-tier firms.

    Priors (ADR-004, post-STAT-405 audit):
        - γ = 1.0 (DP concentration; E[K] ≈ γ·log(n))
        - κ₀ = 1.0 (weakly informative mean shrinkage)
        - ν₀ = 52 (= d+2, finite posterior mean for covariance)
        - mean_prior = global_mean (empirical Bayes from pooled patents)
        - covariance_prior = global_var (empirical Bayes from pooled patents)

    Components with weight < 0.01 are pruned; remaining weights renormalized.
    """
    portfolio_cfg = config["portfolio"]

    # Guard: K_max cannot exceed n_samples - 1 (sklearn constraint)
    actual_kmax = min(k_max, len(X) - 1)

    bgm = BayesianGaussianMixture(
        n_components=actual_kmax,
        covariance_type="diag",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=portfolio_cfg["weight_concentration_prior"],
        mean_prior=global_mean,
        mean_precision_prior=portfolio_cfg["mean_precision_prior"],
        degrees_of_freedom_prior=portfolio_cfg["degrees_of_freedom_prior"],
        covariance_prior=global_var,
        max_iter=portfolio_cfg["max_iter"],
        n_init=portfolio_cfg["n_init"],
        random_state=portfolio_cfg["random_state"],
        reg_covar=1e-6,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress convergence warnings
        bgm.fit(X)

    # Prune components with weight below threshold
    weights = bgm.weights_
    mask = weights >= WEIGHT_PRUNE_THRESHOLD
    if not mask.any():
        # Degenerate case: all pruned → keep heaviest component
        mask[np.argmax(weights)] = True

    pruned_means = bgm.means_[mask].astype(np.float64)
    pruned_covs = bgm.covariances_[mask].astype(np.float64)
    pruned_weights = weights[mask].astype(np.float64)
    pruned_weights /= pruned_weights.sum()  # Renormalize

    return {
        "gvkey": gvkey,
        "n_patents": len(X),
        "n_components": int(mask.sum()),
        "tier": "gmm",
        "covariance_type": "diagonal",
        "means": pruned_means,           # (K_eff, 50)
        "covariances": pruned_covs,       # (K_eff, 50)
        "weights": pruned_weights,        # (K_eff,)
        "converged": bgm.converged_,
        "lower_bound": float(bgm.lower_bound_),
        "n_iter": bgm.n_iter_,
    }


def fit_all_firms(
    firm_vectors: dict,
    tier_assignments: dict,
    k_max: int,
    global_mean: np.ndarray,
    global_var: np.ndarray,
    config: dict,
) -> list[dict]:
    """Fit GMMs for all non-excluded firms at a given K_max.

    Processing order: sorted gvkey for determinism.
    """
    results = []
    sorted_gvkeys = sorted(tier_assignments.keys())
    n_total = len(sorted_gvkeys)
    n_gmm = sum(1 for t in tier_assignments.values() if t == "gmm")
    n_sg = n_total - n_gmm

    print(f"    Fitting {n_total} firms (GMM: {n_gmm}, single-Gaussian: {n_sg})")

    t0 = time.time()
    for i, gvkey in enumerate(sorted_gvkeys):
        tier = tier_assignments[gvkey]
        X = firm_vectors[gvkey]

        if tier == "single_gaussian":
            result = fit_single_gaussian(gvkey, X)
        else:
            result = fit_bayesian_gmm(gvkey, X, k_max, global_mean, global_var, config)

        results.append(result)

        # Progress reporting every 500 firms
        if (i + 1) % 500 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            print(f"    [{i+1}/{n_total}] {elapsed:.0f}s elapsed, "
                  f"{rate:.1f} firms/s, ETA {eta:.0f}s")

    return results


# ---------------------------------------------------------------------------
# Serialization (follows firm_portfolio_spec.md contract)
# ---------------------------------------------------------------------------

def serialize_gmm_results(results: list[dict], output_path: str, k_max: int, config: dict) -> Path:
    """Serialize GMM results to parquet following the spec contract.

    Schema:
        gvkey: string, n_patents: int32, n_components: int32, tier: string,
        covariance_type: string, means: binary, covariances: binary,
        weights: binary, converged: bool, lower_bound: float64, n_iter: int32
    """
    rows = []
    for r in results:
        rows.append({
            "gvkey": r["gvkey"],
            "n_patents": r["n_patents"],
            "n_components": r["n_components"],
            "tier": r["tier"],
            "covariance_type": r["covariance_type"],
            "means": r["means"].tobytes(),
            "covariances": r["covariances"].tobytes(),
            "weights": r["weights"].tobytes(),
            "converged": r["converged"],
            "lower_bound": r["lower_bound"],
            "n_iter": r["n_iter"],
        })

    table = pa.table({
        "gvkey": pa.array([r["gvkey"] for r in rows], type=pa.string()),
        "n_patents": pa.array([r["n_patents"] for r in rows], type=pa.int32()),
        "n_components": pa.array([r["n_components"] for r in rows], type=pa.int32()),
        "tier": pa.array([r["tier"] for r in rows], type=pa.string()),
        "covariance_type": pa.array([r["covariance_type"] for r in rows], type=pa.string()),
        "means": pa.array([r["means"] for r in rows], type=pa.binary()),
        "covariances": pa.array([r["covariances"] for r in rows], type=pa.binary()),
        "weights": pa.array([r["weights"] for r in rows], type=pa.binary()),
        "converged": pa.array([r["converged"] for r in rows], type=pa.bool_()),
        "lower_bound": pa.array([r["lower_bound"] for r in rows], type=pa.float64()),
        "n_iter": pa.array([r["n_iter"] for r in rows], type=pa.int32()),
    })

    # File-level metadata per spec
    portfolio_cfg = config["portfolio"]
    meta = {
        b"gmm_method": b"bayesian",
        b"k_max": str(k_max).encode(),
        b"covariance_type": b"diag",
        b"weight_pruning_threshold": str(WEIGHT_PRUNE_THRESHOLD).encode(),
        b"weight_concentration_prior": str(portfolio_cfg["weight_concentration_prior"]).encode(),
        b"normalization": portfolio_cfg["normalization"].encode(),
        b"total_firms": str(len(results)).encode(),
        b"created_at": datetime.now(timezone.utc).isoformat().encode(),
    }
    existing = table.schema.metadata or {}
    existing.update(meta)
    table = table.replace_schema_metadata(existing)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    return path


def load_gmm_results(path: str) -> list[dict]:
    """Load serialized GMM results back into list of dicts."""
    table = pq.read_table(path)
    results = []
    for i in range(len(table)):
        n_comp = table["n_components"][i].as_py()
        d = 50  # UMAP output dimensionality
        results.append({
            "gvkey": table["gvkey"][i].as_py(),
            "n_patents": table["n_patents"][i].as_py(),
            "n_components": n_comp,
            "tier": table["tier"][i].as_py(),
            "covariance_type": table["covariance_type"][i].as_py(),
            "means": np.frombuffer(table["means"][i].as_py(), dtype=np.float64).reshape(n_comp, d),
            "covariances": np.frombuffer(table["covariances"][i].as_py(), dtype=np.float64).reshape(n_comp, d),
            "weights": np.frombuffer(table["weights"][i].as_py(), dtype=np.float64),
            "converged": table["converged"][i].as_py(),
            "lower_bound": table["lower_bound"][i].as_py(),
            "n_iter": table["n_iter"][i].as_py(),
        })
    return results


# ---------------------------------------------------------------------------
# Bhattacharyya Coefficient (diagonal case)
# ---------------------------------------------------------------------------

def bc_component_matrix(
    mu_a: np.ndarray, var_a: np.ndarray,
    mu_b: np.ndarray, var_b: np.ndarray,
) -> np.ndarray:
    """BC between all component pairs of two GMMs (vectorized).

    Computes the full K_A × K_B grid of component-pair BCs in one numpy
    operation using broadcasting, eliminating Python loops.

    Formula (ADR-006):
        D_B = (1/8)(μ₁-μ₂)ᵀ Σ⁻¹ (μ₁-μ₂) + (1/2) ln(det(Σ) / √(det(Σ₁)·det(Σ₂)))
        BC = exp(-D_B)
    where Σ = (Σ₁ + Σ₂) / 2, all covariances diagonal.

    Args:
        mu_a: (K_A, D) means of GMM A
        var_a: (K_A, D) diagonal covariances of GMM A
        mu_b: (K_B, D) means of GMM B
        var_b: (K_B, D) diagonal covariances of GMM B

    Returns:
        (K_A, K_B) matrix of component-pair BC values
    """
    # Broadcast to (K_A, K_B, D)
    sigma_avg = (var_a[:, None, :] + var_b[None, :, :]) / 2.0
    diff = mu_a[:, None, :] - mu_b[None, :, :]

    # Mahalanobis term: (K_A, K_B)
    mahal = 0.125 * np.sum(diff**2 / sigma_avg, axis=2)

    # Log-determinant terms
    log_det_avg = np.sum(np.log(sigma_avg), axis=2)   # (K_A, K_B)
    log_det_a = np.sum(np.log(var_a), axis=1)          # (K_A,)
    log_det_b = np.sum(np.log(var_b), axis=1)          # (K_B,)
    det_term = 0.5 * (log_det_avg - 0.5 * (log_det_a[:, None] + log_det_b[None, :]))

    return np.exp(-(mahal + det_term))


def bc_mixture(gmm_a: dict, gmm_b: dict) -> float:
    """Bhattacharyya coefficient between two GMMs (vectorized).

    Formula (methodology.md):
        BC(A, B) = Σᵢ Σⱼ √(πᵢᴬ · πⱼᴮ) · BC(Nᵢᴬ, Nⱼᴮ)

    All K_A × K_B component-pair BCs computed in one numpy operation.
    """
    # Component-pair BC matrix: (K_A, K_B)
    bc_grid = bc_component_matrix(
        gmm_a["means"], gmm_a["covariances"],
        gmm_b["means"], gmm_b["covariances"],
    )
    # Weight matrix: (K_A, K_B)
    weight_grid = np.sqrt(gmm_a["weights"][:, None] * gmm_b["weights"][None, :])

    return float(np.sum(weight_grid * bc_grid))


def compute_bc_matrix(
    results: list[dict],
    label: str = "all",
    sg_block: np.ndarray | None = None,
    sg_gvkeys: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Compute pairwise BC matrix for all non-excluded firms.

    Optimization: single-Gaussian (K=1) firms are K_max-invariant. The BC
    between two SG firms never changes across K_max values. If sg_block and
    sg_gvkeys are provided, the SG-vs-SG block is copied from the precomputed
    cache rather than recomputed.

    Args:
        results: list of GMM result dicts (all non-excluded firms)
        label: label for progress messages
        sg_block: precomputed (N_sg, N_sg) BC matrix for SG-vs-SG pairs
        sg_gvkeys: ordered gvkeys corresponding to sg_block rows/cols

    Returns:
        gvkeys: ordered list of firm identifiers
        bc_matrix: (N, N) symmetric matrix of BC values
    """
    n = len(results)
    gvkeys = [r["gvkey"] for r in results]
    bc_matrix = np.zeros((n, n), dtype=np.float64)

    # Build index for sg_block reuse
    sg_idx_map = {}
    if sg_block is not None and sg_gvkeys is not None:
        sg_idx_map = {gk: idx for idx, gk in enumerate(sg_gvkeys)}

    # Classify each result by tier for skip logic
    is_sg = [r["tier"] == "single_gaussian" for r in results]

    total_pairs = n * (n - 1) // 2
    computed = 0
    skipped = 0
    t0 = time.time()

    for i in range(n):
        bc_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            # Reuse precomputed SG-vs-SG block if both firms are single-Gaussian
            if is_sg[i] and is_sg[j] and gvkeys[i] in sg_idx_map and gvkeys[j] in sg_idx_map:
                bc_val = sg_block[sg_idx_map[gvkeys[i]], sg_idx_map[gvkeys[j]]]
                skipped += 1
            else:
                bc_val = bc_mixture(results[i], results[j])
                computed += 1

            bc_matrix[i, j] = bc_val
            bc_matrix[j, i] = bc_val

        # Progress every 250 firms
        done = computed + skipped
        if (i + 1) % 250 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            pct = done / total_pairs * 100 if total_pairs > 0 else 100
            rate = computed / elapsed if elapsed > 0 else 0
            eta_computed = total_pairs - done
            # Rough ETA: remaining pairs that need computation (not skipped)
            print(f"    BC {label} [{i+1}/{n} rows] computed={computed:,} "
                  f"cached={skipped:,} ({pct:.1f}%) {elapsed:.0f}s elapsed")

    return gvkeys, bc_matrix


def compute_sg_block(results: list[dict]) -> tuple[list[str], np.ndarray]:
    """Precompute the SG-vs-SG BC block (K_max-invariant).

    Single-Gaussian firms have K=1, so their BC values never change
    across K_max settings. Computing this block once and reusing it
    across all K_max values avoids redundant work.
    """
    sg_firms = [r for r in results if r["tier"] == "single_gaussian"]
    n = len(sg_firms)
    gvkeys = [r["gvkey"] for r in sg_firms]
    block = np.zeros((n, n), dtype=np.float64)

    total_pairs = n * (n - 1) // 2
    computed = 0
    t0 = time.time()

    for i in range(n):
        block[i, i] = 1.0
        for j in range(i + 1, n):
            bc_val = bc_mixture(sg_firms[i], sg_firms[j])
            block[i, j] = bc_val
            block[j, i] = bc_val
            computed += 1

        if (i + 1) % 500 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            pct = computed / total_pairs * 100 if total_pairs > 0 else 100
            print(f"    SG block [{i+1}/{n}] {computed:,}/{total_pairs:,} pairs "
                  f"({pct:.1f}%) {elapsed:.0f}s")

    return gvkeys, block


# ---------------------------------------------------------------------------
# Convergence metrics
# ---------------------------------------------------------------------------

def compute_convergence_metrics(
    gvkeys_a: list[str], bc_a: np.ndarray,
    gvkeys_b: list[str], bc_b: np.ndarray,
    k_max_a: int, k_max_b: int,
) -> dict:
    """Compare BC ranking stability between two K_max settings.

    Metrics:
        - Spearman ρ on all pairwise BC values
        - Kendall τ on all pairwise BC values
        - Top-k pair overlap (k=50, 100, 200)
        - Per-firm top-5 nearest neighbor stability
    """
    # Ensure same firms in same order
    assert gvkeys_a == gvkeys_b, "Firm sets must match between K_max values"
    n = len(gvkeys_a)

    # Extract upper-triangle BC values (all unique pairs)
    idx_upper = np.triu_indices(n, k=1)
    bc_flat_a = bc_a[idx_upper]
    bc_flat_b = bc_b[idx_upper]

    # Rank correlations
    rho, rho_p = spearmanr(bc_flat_a, bc_flat_b)
    tau, tau_p = kendalltau(bc_flat_a, bc_flat_b)

    # Top-k pair overlap
    # Pairs are indexed by their position in the flat upper-triangle array
    rank_a = np.argsort(-bc_flat_a)  # Descending
    rank_b = np.argsort(-bc_flat_b)

    top_k_overlaps = {}
    for k in [50, 100, 200]:
        k_actual = min(k, len(rank_a))
        top_a = set(rank_a[:k_actual])
        top_b = set(rank_b[:k_actual])
        overlap = len(top_a & top_b) / k_actual * 100 if k_actual > 0 else 0.0
        top_k_overlaps[f"top_{k}_overlap_pct"] = round(overlap, 1)

    # Per-firm top-5 nearest neighbor stability
    nn_overlaps = []
    for i in range(n):
        # BC values for firm i (excluding self)
        row_a = bc_a[i].copy()
        row_b = bc_b[i].copy()
        row_a[i] = -1  # Exclude self
        row_b[i] = -1

        top5_a = set(np.argsort(-row_a)[:5])
        top5_b = set(np.argsort(-row_b)[:5])
        nn_overlaps.append(len(top5_a & top5_b) / 5 * 100)

    mean_nn_overlap = np.mean(nn_overlaps)
    median_nn_overlap = np.median(nn_overlaps)
    p10_nn_overlap = np.percentile(nn_overlaps, 10)

    return {
        "k_max_a": k_max_a,
        "k_max_b": k_max_b,
        "n_firms": n,
        "n_pairs": len(bc_flat_a),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(rho_p),
        "kendall_tau": round(float(tau), 4),
        "kendall_p": float(tau_p),
        **top_k_overlaps,
        "mean_nn5_overlap_pct": round(float(mean_nn_overlap), 1),
        "median_nn5_overlap_pct": round(float(median_nn_overlap), 1),
        "p10_nn5_overlap_pct": round(float(p10_nn_overlap), 1),
    }


def compute_effective_k_summary(results: list[dict], k_max: int = 0) -> dict:
    """Summary statistics on effective K across firms.

    Args:
        results: list of GMM result dicts
        k_max: the K_max used for this sweep (for ceiling rate calculation)
    """
    gmm_results = [r for r in results if r["tier"] == "gmm"]
    ks = [r["n_components"] for r in gmm_results]
    if not ks:
        return {}
    return {
        "n_gmm_firms": len(ks),
        "mean_k": round(np.mean(ks), 2),
        "median_k": int(np.median(ks)),
        "std_k": round(np.std(ks), 2),
        "min_k": int(np.min(ks)),
        "max_k": int(np.max(ks)),
        "p25_k": int(np.percentile(ks, 25)),
        "p75_k": int(np.percentile(ks, 75)),
        "p90_k": int(np.percentile(ks, 90)),
        "pct_at_ceiling": round(sum(1 for r in gmm_results
                                     if r["n_components"] >= min(r["n_patents"] - 1,
                                                                  k_max)) / len(ks) * 100, 1)
                          if k_max > 0 else 0.0,
        "converged_pct": round(sum(1 for r in gmm_results
                                    if r["converged"]) / len(ks) * 100, 1),
    }


# ---------------------------------------------------------------------------
# Status and logging helpers
# ---------------------------------------------------------------------------

def write_status(status: str, run_id: str, extra: dict | None = None):
    """Write status JSON for monitoring (matches Week 1 pattern)."""
    status_dir = Path(OUTPUT_DIR) / "status"
    status_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "status": status,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        payload.update(extra)

    with open(status_dir / "sweep_status.json", "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="K_max convergence sweep")
    parser.add_argument("--local", action="store_true",
                        help="Use local output/week2_inputs/ instead of config paths")
    args = parser.parse_args()

    t_start = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    config = load_config()
    portfolio_cfg = config["portfolio"]

    # Read K_max sweep values from config (Minor #1: no hardcoded values)
    k_max_values = sorted(portfolio_cfg["k_max_sweep"])

    print("=" * 70)
    print(f"K_max Convergence Sweep — Run {run_id}")
    print(f"K_max values: {k_max_values}")
    print("=" * 70)

    write_status("running", run_id, {"k_max_values": k_max_values})

    # ---- Stage 1: Load data ----
    print("\n[Stage 1] Loading data...")
    vectors, gvkey_map, pid_to_idx = load_inputs(config, local=args.local)

    # ---- Stage 2: Group firms and classify tiers ----
    print("\n[Stage 2] Grouping firms and classifying tiers (ADR-005)...")
    firm_vectors, tier_assignments, excluded_firms = group_and_classify(
        vectors, gvkey_map, pid_to_idx,
        min_patents=portfolio_cfg["min_patents"],
        single_gaussian_max=portfolio_cfg["single_gaussian_max"],
    )

    n_gmm = sum(1 for t in tier_assignments.values() if t == "gmm")
    n_sg = sum(1 for t in tier_assignments.values() if t == "single_gaussian")
    n_gmm_patents = sum(len(firm_vectors[gk]) for gk, t in tier_assignments.items() if t == "gmm")
    n_all = n_gmm + n_sg
    print(f"  Excluded: {len(excluded_firms):,} firms")
    print(f"  Single-Gaussian: {n_sg:,} firms")
    print(f"  GMM-tier: {n_gmm:,} firms ({n_gmm_patents:,} patents)")
    print(f"  Total non-excluded: {n_all:,} firms (all included in BC analysis)")

    # ---- Stage 3: Compute global priors ----
    print("\n[Stage 3] Computing global empirical Bayes priors...")
    # From unique patent matrix — NOT from grouped firm vectors (Codex Major #3)
    global_mean, global_var = compute_global_priors(vectors)
    print(f"  Global mean range: [{global_mean.min():.3f}, {global_mean.max():.3f}]")
    print(f"  Global var range: [{global_var.min():.4f}, {global_var.max():.4f}]")

    # Free the large vector array after extracting firm vectors and priors
    del vectors, gvkey_map, pid_to_idx

    # ---- Stage 4: Fit GMMs at each K_max ----
    all_results = {}  # k_max -> list[dict]

    for k_idx, k_max in enumerate(k_max_values):
        print(f"\n[Stage 4.{k_idx+1}] Fitting GMMs at K_max={k_max}...")

        # Check for checkpoint
        ckpt_path = f"{OUTPUT_DIR}/firm_gmm_parameters_k{k_max}.parquet"
        if Path(ckpt_path).exists():
            print(f"    Checkpoint found at {ckpt_path}, loading...")
            all_results[k_max] = load_gmm_results(ckpt_path)
            k_summary = compute_effective_k_summary(all_results[k_max], k_max)
            print(f"    Loaded {len(all_results[k_max])} firms, "
                  f"mean K={k_summary.get('mean_k', 'N/A')}")
            continue

        t_k = time.time()
        results = fit_all_firms(
            firm_vectors, tier_assignments, k_max,
            global_mean, global_var, config,
        )
        elapsed_k = time.time() - t_k

        # Summary stats for this K_max
        k_summary = compute_effective_k_summary(results, k_max)
        print(f"    K_max={k_max} complete in {elapsed_k:.0f}s ({elapsed_k/60:.1f} min)")
        print(f"    Effective K: mean={k_summary.get('mean_k', 'N/A')}, "
              f"median={k_summary.get('median_k', 'N/A')}, "
              f"ceiling={k_summary.get('pct_at_ceiling', 'N/A')}%, "
              f"converged={k_summary.get('converged_pct', 'N/A')}%")

        # Save checkpoint
        serialize_gmm_results(results, ckpt_path, k_max, config)
        print(f"    Saved to {ckpt_path}")

        all_results[k_max] = results
        write_status("running", run_id, {
            "k_max_values": k_max_values,
            "completed_k_max": k_max_values[:k_idx+1],
            "current_k_summary": k_summary,
        })

    # ---- Stage 5: Compute pairwise BC at each K_max ----
    # Primary analysis: ALL non-excluded firms (Codex Major #1)
    # Single-Gaussian firms are K_max-invariant, so we precompute their
    # mutual BC block once and reuse it across all K_max values.
    print("\n[Stage 5] Computing pairwise Bhattacharyya Coefficients...")
    print("  Scope: ALL non-excluded firms (single-Gaussian + GMM-tier)")

    # 5a: Precompute SG-vs-SG block (K_max-invariant)
    sg_ckpt = f"{OUTPUT_DIR}/bc_block_sg_vs_sg.npz"
    if Path(sg_ckpt).exists():
        print("\n  Loading cached SG-vs-SG block...")
        sg_data = np.load(sg_ckpt, allow_pickle=True)
        sg_gvkeys = sg_data["gvkeys"].tolist()
        sg_block = sg_data["bc_matrix"]
        print(f"    SG block: {sg_block.shape} ({len(sg_gvkeys)} firms)")
    else:
        # Use results from any K_max (SG firms are identical across K_max)
        first_k = k_max_values[0]
        print(f"\n  Computing SG-vs-SG block ({n_sg} firms, K_max-invariant)...")
        t_sg = time.time()
        sg_gvkeys, sg_block = compute_sg_block(all_results[first_k])
        elapsed_sg = time.time() - t_sg
        print(f"    SG block: {sg_block.shape}, {elapsed_sg:.0f}s ({elapsed_sg/60:.1f} min)")
        np.savez_compressed(sg_ckpt, gvkeys=np.array(sg_gvkeys), bc_matrix=sg_block)
        print(f"    Cached to {sg_ckpt}")

    # 5b: Full BC matrices per K_max (reusing SG block)
    bc_matrices = {}  # k_max -> (gvkeys, bc_matrix)

    for k_max in k_max_values:
        print(f"\n  [K_max={k_max}] Computing BC matrix (all non-excluded firms)...")

        bc_ckpt = f"{OUTPUT_DIR}/bc_matrix_all_k{k_max}.npz"
        if Path(bc_ckpt).exists():
            print(f"    Checkpoint found at {bc_ckpt}, loading...")
            data = np.load(bc_ckpt, allow_pickle=True)
            bc_matrices[k_max] = (data["gvkeys"].tolist(), data["bc_matrix"])
            print(f"    Loaded {len(bc_matrices[k_max][0])} firms")
            continue

        t_bc = time.time()
        gvkeys, bc_matrix = compute_bc_matrix(
            all_results[k_max],
            label=f"k{k_max}",
            sg_block=sg_block,
            sg_gvkeys=sg_gvkeys,
        )
        elapsed_bc = time.time() - t_bc
        print(f"    BC matrix: {bc_matrix.shape}, {elapsed_bc:.0f}s ({elapsed_bc/60:.1f} min)")

        np.savez_compressed(bc_ckpt, gvkeys=np.array(gvkeys), bc_matrix=bc_matrix)
        print(f"    Saved to {bc_ckpt}")
        bc_matrices[k_max] = (gvkeys, bc_matrix)

    # ---- Stage 6: Convergence analysis ----
    print("\n[Stage 6] Computing convergence metrics...")

    convergence_results = []
    sorted_kmax = sorted(bc_matrices.keys())
    step = sorted_kmax[1] - sorted_kmax[0] if len(sorted_kmax) > 1 else 5

    # Adjacent comparisons
    for i in range(len(sorted_kmax) - 1):
        k_a = sorted_kmax[i]
        k_b = sorted_kmax[i + 1]
        gvkeys_a, bc_a = bc_matrices[k_a]
        gvkeys_b, bc_b = bc_matrices[k_b]

        print(f"\n  Comparing K_max={k_a} vs K_max={k_b}...")
        metrics = compute_convergence_metrics(gvkeys_a, bc_a, gvkeys_b, bc_b, k_a, k_b)
        convergence_results.append(metrics)

        print(f"    Spearman ρ = {metrics['spearman_rho']:.4f}")
        print(f"    Kendall τ  = {metrics['kendall_tau']:.4f}")
        for k_val in [50, 100, 200]:
            key = f"top_{k_val}_overlap_pct"
            if key in metrics:
                print(f"    Top-{k_val} overlap = {metrics[key]:.1f}%")
        print(f"    Mean NN-5 overlap = {metrics['mean_nn5_overlap_pct']:.1f}%")
        print(f"    P10 NN-5 overlap  = {metrics['p10_nn5_overlap_pct']:.1f}%")

    # Non-adjacent comparisons for full picture
    print("\n  Non-adjacent comparisons...")
    non_adjacent_pairs = []
    if len(sorted_kmax) >= 3:
        non_adjacent_pairs.append((sorted_kmax[0], sorted_kmax[-1]))   # min vs max
        non_adjacent_pairs.append((sorted_kmax[0], sorted_kmax[2]))    # e.g. K=10 vs K=20
    if len(sorted_kmax) >= 4:
        non_adjacent_pairs.append((sorted_kmax[1], sorted_kmax[-1]))   # e.g. K=15 vs K=30

    for k_a, k_b in non_adjacent_pairs:
        gvkeys_a, bc_a = bc_matrices[k_a]
        gvkeys_b, bc_b = bc_matrices[k_b]
        metrics = compute_convergence_metrics(gvkeys_a, bc_a, gvkeys_b, bc_b, k_a, k_b)
        convergence_results.append(metrics)
        print(f"    K_max={k_a} vs {k_b}: ρ={metrics['spearman_rho']:.4f}, "
              f"top-50={metrics.get('top_50_overlap_pct', 'N/A')}%")

    # ---- Stage 7: Effective K summary across K_max values ----
    print("\n[Stage 7] Effective K summary across K_max values...")

    k_summaries = {}
    for k_max in k_max_values:
        k_summaries[k_max] = compute_effective_k_summary(all_results[k_max], k_max)
        s = k_summaries[k_max]
        print(f"  K_max={k_max}: mean_K={s['mean_k']}, median_K={s['median_k']}, "
              f"p90_K={s['p90_k']}, ceiling={s['pct_at_ceiling']}%")

    # ---- Stage 8: Save convergence summary ----
    print("\n[Stage 8] Saving convergence summary...")

    # Determine convergence verdict (Major #2: persistent stability)
    # K* = smallest K_max such that ALL subsequent adjacent comparisons pass.
    # "Converged at K*" means: the transition INTO K* passes, and all transitions
    # AFTER K* also pass. This ensures "raising K_max further stops mattering."
    adjacent_results = [m for m in convergence_results
                        if m["k_max_b"] - m["k_max_a"] == step]
    # Sort by k_max_a to ensure correct order
    adjacent_results.sort(key=lambda m: m["k_max_a"])

    def passes_threshold(m: dict) -> bool:
        return (m["spearman_rho"] > 0.95
                and m.get("top_50_overlap_pct", 0) > 80)

    converged = False
    converged_at = None

    # Walk from the earliest adjacent pair forward.
    # For each candidate K*, check that this pair AND all subsequent pairs pass.
    for start_idx in range(len(adjacent_results)):
        # K* would be adjacent_results[start_idx]["k_max_b"] (the "to" value)
        all_pass = all(passes_threshold(adjacent_results[j])
                       for j in range(start_idx, len(adjacent_results)))
        if all_pass:
            converged = True
            converged_at = adjacent_results[start_idx]["k_max_b"]
            break

    summary = {
        "run_id": run_id,
        "k_max_values": k_max_values,
        "bc_scope": "all_non_excluded",
        "convergence_verdict": "converged" if converged else "not_converged",
        "converged_at_kmax": converged_at,
        "decision_rule": {
            "spearman_threshold": 0.95,
            "top_50_overlap_threshold_pct": 80,
            "method": "persistent_stability",
            "definition": "K* = smallest K_max such that all subsequent adjacent "
                          "comparisons from K* onward pass both thresholds",
        },
        "adjacent_comparisons": [m for m in convergence_results
                                  if m["k_max_b"] - m["k_max_a"] == step],
        "non_adjacent_comparisons": [m for m in convergence_results
                                      if m["k_max_b"] - m["k_max_a"] != step],
        "effective_k_summaries": {str(k): s for k, s in k_summaries.items()},
        "timing": {},
    }

    elapsed_total = time.time() - t_start
    summary["timing"] = {
        "total_seconds": round(elapsed_total, 1),
        "total_minutes": round(elapsed_total / 60, 1),
        "total_hours": round(elapsed_total / 3600, 2),
    }

    summary_path = f"{OUTPUT_DIR}/convergence_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {summary_path}")

    # Save excluded firms log (Minor #2: spec-compliant schema)
    excluded_path = f"{OUTPUT_DIR}/excluded_firms.csv"
    pd.DataFrame({
        "gvkey": [gk for gk, _ in excluded_firms],
        "n_patents": [n for _, n in excluded_firms],
        "reason": [f"below_min_patents_{portfolio_cfg['min_patents']}"
                   for _ in excluded_firms],
    }).to_csv(excluded_path, index=False)
    print(f"  Excluded firms: {excluded_path}")

    # ---- Final report ----
    print(f"\n{'=' * 70}")
    print(f"K_max Convergence Sweep Complete — Run {run_id}")
    print(f"{'=' * 70}")
    print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min, "
          f"{elapsed_total/3600:.2f} hr)")
    print(f"  K_max values tested: {k_max_values}")
    print(f"  Firms fitted: {len(tier_assignments):,} "
          f"(GMM: {n_gmm:,}, single-Gaussian: {n_sg:,})")
    print(f"  BC scope: all {n_all:,} non-excluded firms")
    print(f"  Excluded: {len(excluded_firms):,}")
    print()

    if converged:
        print(f"  VERDICT: CONVERGED at K_max={converged_at}")
        print(f"  (Persistent stability: all adjacent pairs from K_max={converged_at} onward pass)")
        print(f"  Recommendation: Adopt K_max={converged_at} as production default")
    else:
        print(f"  VERDICT: NOT CONVERGED by K_max={k_max_values[-1]}")
        print(f"  Action: Escalate per Codex trigger framework (reopen ADR-004)")

    print()
    print("Adjacent comparisons:")
    for m in adjacent_results:
        marker = " *" if passes_threshold(m) else ""
        print(f"  K={m['k_max_a']} vs K={m['k_max_b']}: "
              f"ρ={m['spearman_rho']:.4f}, "
              f"top-50={m.get('top_50_overlap_pct', 'N/A')}%, "
              f"NN-5={m['mean_nn5_overlap_pct']:.1f}%{marker}")

    print()
    print("Effective K progression:")
    for k_max in k_max_values:
        s = k_summaries[k_max]
        print(f"  K_max={k_max}: mean={s['mean_k']}, "
              f"p90={s['p90_k']}, ceiling={s['pct_at_ceiling']}%")

    print(f"\n{'=' * 70}")
    print("Sweep complete")
    print(f"{'=' * 70}")

    write_status("success" if converged else "completed_no_convergence", run_id, {
        "verdict": "converged" if converged else "not_converged",
        "converged_at": converged_at,
        "total_seconds": round(elapsed_total, 1),
    })


if __name__ == "__main__":
    main()
