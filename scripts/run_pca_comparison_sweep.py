"""PCA-50 comparison sweep for firm GMM portfolio fitting (UMAP baseline check).

Why this script exists
----------------------
The K_max sweep on UMAP-50 vectors returned NOT_CONVERGED and was subsequently
fixed by (a) deduplicating 464 duplicate firm records (aliases / subsidiaries /
predecessors) and (b) correcting the BC mixture formula to use linear weights
(π_i·π_j, not sqrt(π_i·π_j)). A 1,000-firm smoke test of the corrected pipeline
on UMAP-50 showed convergence at K_max=10.

This script tests whether UMAP's nonlinear distortion is contributing to any
residual issues. PCA is linear, preserves global variance structure by
construction, and is the natural baseline for a methodological sanity check.

Decision rule for interpretation (from kmax_diagnostic_findings.md):
    - PCA convergence curves close to UMAP → both representations are adequate
      for downstream BC analysis; no methodological alarm.
    - PCA noticeably better/worse than UMAP → important signal about whether
      UMAP is distorting firm portfolio geometry in ways that matter for M&A
      candidate ranking.

Pipeline
--------
1. Load 1536D pre-UMAP patent vectors (PatentSBERTa text ⊕ citation embeddings).
   - Preferred local path: output/embeddings/concatenated_1536d.parquet
   - Fallback CLI arg: --input-path PATH
   - S3 source (Week 1 production run):
       s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/
       concatenated_1536d.parquet
     Pull locally before running (aws s3 cp ... --profile torrin).
2. Fit sklearn.decomposition.PCA(n_components=50, svd_solver='randomized') and
   report cumulative explained variance ratio.
3. Group PCA-50 vectors by gvkey and classify tiers per ADR-005
   (exclude: n<5, single_gaussian: 5-49, gmm: 50+).
4. Filter to deduplicated firms per output/kmax_sweep/deduplication_decisions.parquet.
5. Compute global empirical Bayes priors from the PCA-50 patent matrix (not
   grouped per-firm vectors — avoids co-assignment double counting, see
   run_kmax_sweep.compute_global_priors docstring and Codex Major #3).
6. Fit Bayesian GMMs at K_max ∈ {10, 20, 30} for GMM-tier firms and single
   Gaussians for single_gaussian-tier firms. Uses float64 + reg_covar=1e-4
   (defense in depth against the float32 catastrophic cancellation bug that
   crashed the original production sweep — see fit_single_gaussian docstring
   in run_kmax_sweep.py).
7. Compute pairwise BC matrices using the CORRECTED linear weights (π_i·π_j),
   with SG-vs-SG block cached once and reused across K_max values.
8. Compute convergence metrics (Spearman ρ, top-{50,100,200} overlap, NN-5
   stability) between adjacent K_max transitions and the min-vs-max pair.
9. Side-by-side report of PCA vs UMAP convergence (if the corrected UMAP
   summary is available at output/kmax_sweep/convergence_summary_dedup_linear.json).

Outputs (all under output/kmax_sweep/pca_sweep/)
------------------------------------------------
- patent_vectors_pca50.parquet       — PCA-50 embeddings, same schema as 50D UMAP
- pca_explained_variance.json        — Per-component + cumulative EVR, 50th-comp cutoff
- firm_gmm_parameters_pca_k{N}.parquet — GMM parameters per K_max (same schema as sweep)
- bc_block_sg_vs_sg_pca.npz          — K_max-invariant SG-vs-SG BC block cache
- bc_matrix_pca_k{N}.npz             — Full pairwise BC matrix per K_max
- convergence_summary_pca.json       — Adjacent + non-adjacent convergence metrics

Usage
-----
    # Full sweep (expected runtime ~2-3 hours on c5.4xlarge)
    python scripts/run_pca_comparison_sweep.py

    # Test mode: 100 SG + 100 GMM firms (~5 min local / 1 min VM)
    python scripts/run_pca_comparison_sweep.py --test-mode

    # Explicit input path (if not at output/embeddings/concatenated_1536d.parquet)
    python scripts/run_pca_comparison_sweep.py \
        --input-path /mnt/data/concatenated_1536d.parquet

    # Reuse cached stages
    python scripts/run_pca_comparison_sweep.py --skip-pca         # reuse PCA-50 parquet
    python scripts/run_pca_comparison_sweep.py --skip-gmm         # reuse GMM parquets
    python scripts/run_pca_comparison_sweep.py --skip-bc          # reuse BC matrices,
                                                                   # only recompute convergence

Notes
-----
- K_max ∈ {10, 20, 30} is a deliberate 3-point subset of the original sweep's
  5-point grid. That is enough to (a) see whether the convergence curve is
  monotonic and (b) confirm whether the corrected pipeline converges as it did
  for UMAP. If we see interesting divergence we can fill in K_max=15,25 after.
- Priors: same values as run_kmax_sweep (γ=1.0, κ₀=1.0, ν₀=52 = d+2 with d=50).
  These are valid for PCA-50 because we match the UMAP output dimensionality.
- This script imports helpers from run_kmax_sweep and recompute_bc_corrected
  to minimize duplication. Only the input-loading / PCA / reporting layer is new.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure project root on sys.path so we can import from src/ and scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config  # noqa: E402

# Reuse all the helpers that are identical between the UMAP sweep and this one.
# bc_component_matrix is the correct diagonal BC kernel (no changes needed).
# The only BC-level change is the linear-weighted mixture aggregation, which
# we take from recompute_bc_corrected.
from scripts.run_kmax_sweep import (  # noqa: E402
    WEIGHT_PRUNE_THRESHOLD,
    compute_effective_k_summary,
    compute_global_priors,
    fit_bayesian_gmm,
    fit_single_gaussian,
    group_and_classify,
    load_gmm_results,
    serialize_gmm_results,
)
from scripts.recompute_bc_corrected import (  # noqa: E402
    compute_bc_matrix_linear,
    compute_convergence_metrics,
    compute_sg_block_linear,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output/kmax_sweep/pca_sweep")
DEFAULT_INPUT_PATH = Path("output/embeddings/concatenated_1536d.parquet")
DEDUP_DECISIONS_PATH = Path("output/kmax_sweep/deduplication_decisions.parquet")
GVKEY_MAP_PATH = Path("output/week2_inputs/gvkey_map.parquet")

# Corrected UMAP convergence summary (produced by recompute_bc_corrected.py).
# Used only for the final side-by-side report. Missing file is handled
# gracefully.
UMAP_CORRECTED_SUMMARY = Path(
    "output/kmax_sweep/convergence_summary_dedup_linear.json"
)

DEFAULT_K_MAX_VALUES = (10, 20, 30)
PCA_N_COMPONENTS = 50
PCA_RANDOM_STATE = 42

# Expected S3 source for the 1536D file. Only printed in error messages.
S3_SOURCE_HINT = (
    "s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/"
    "output/embeddings/concatenated_1536d.parquet"
)


# ---------------------------------------------------------------------------
# Stage 1: Load 1536D patent vectors
# ---------------------------------------------------------------------------


def load_1536d_vectors(input_path: Path) -> tuple[list[str], np.ndarray]:
    """Load the pre-UMAP 1536D patent vectors (PatentSBERTa text ⊕ citation).

    Expected schema (matches output/embeddings/sample_concatenated_1536d.parquet):
        patent_id: string
        embedding: binary (1536 float32 bytes per row)

    We return float32 here to keep peak memory low during PCA (~9 GB for the
    full 1.5M-patent matrix). sklearn.decomposition.PCA will upcast to float64
    internally where required by LAPACK.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"\n  Input file not found: {input_path}\n"
            f"  Pull from S3 first:\n"
            f"      aws s3 cp \\\n"
            f"          {S3_SOURCE_HINT} \\\n"
            f"          {input_path} --profile torrin\n"
            f"  Or pass a different path with --input-path."
        )

    print(f"  Loading 1536D vectors from {input_path}")
    t0 = time.time()
    table = pq.read_table(input_path)
    if "patent_id" not in table.column_names or "embedding" not in table.column_names:
        raise ValueError(
            f"Unexpected schema in {input_path}: columns={table.column_names}. "
            f"Expected ['patent_id', 'embedding']."
        )

    patent_ids = table["patent_id"].to_pylist()

    # embedding is stored as binary (float32 bytes). Materialize to a contiguous
    # float32 array of shape (N, 1536). We convert row-by-row because pyarrow
    # doesn't give us a direct fixed-size-binary → ndarray view for arbitrary
    # schemas, but the single-pass loop is fast relative to PCA fitting.
    embeddings_col = table["embedding"].to_pylist()
    n = len(embeddings_col)
    # Probe dim from the first non-null entry
    probe = np.frombuffer(embeddings_col[0], dtype=np.float32)
    dim = probe.shape[0]
    if dim != 1536:
        raise ValueError(
            f"Unexpected embedding dimensionality: got {dim}, expected 1536. "
            f"This script assumes concatenated PatentSBERTa (768D) + "
            f"citation (768D) vectors."
        )

    vectors = np.empty((n, dim), dtype=np.float32)
    for i, buf in enumerate(embeddings_col):
        vectors[i] = np.frombuffer(buf, dtype=np.float32)

    elapsed = time.time() - t0
    print(
        f"  Loaded {n:,} x {dim}D float32 vectors "
        f"({vectors.nbytes / 1e9:.2f} GB) in {elapsed:.0f}s"
    )
    return patent_ids, vectors


# ---------------------------------------------------------------------------
# Stage 2: Fit PCA and save PCA-50 embeddings
# ---------------------------------------------------------------------------


def fit_and_save_pca(
    patent_ids: list[str],
    vectors_1536d: np.ndarray,
    pca_parquet_path: Path,
    explained_variance_path: Path,
    source_input_path: Path | None = None,
) -> np.ndarray:
    """Fit PCA(n_components=50) and save the PCA-50 vectors + EVR report.

    We use svd_solver='randomized' because it is dramatically faster than the
    exact 'auto' solver when n_components << min(n_samples, n_features), which
    is exactly our regime (50 << 1536). Accuracy is within 1e-3 of exact PCA
    for the top components, which is far below any resolution that would
    affect downstream GMM fitting or BC ranking.
    """
    from sklearn.decomposition import PCA  # Local import: sklearn is heavy

    print(f"  Fitting PCA(n_components={PCA_N_COMPONENTS}, svd_solver='randomized')")
    print(f"  Input shape: {vectors_1536d.shape} (dtype={vectors_1536d.dtype})")
    t0 = time.time()

    pca = PCA(
        n_components=PCA_N_COMPONENTS,
        svd_solver="randomized",
        random_state=PCA_RANDOM_STATE,
        copy=False,  # Allow sklearn to modify in-place to reduce peak memory
    )
    pca_vectors = pca.fit_transform(vectors_1536d)  # (N, 50)
    elapsed = time.time() - t0

    # Cast to float32 for storage parity with the UMAP-50 file
    pca_vectors_f32 = pca_vectors.astype(np.float32)

    print(f"  PCA fit complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Output shape: {pca_vectors_f32.shape}")

    evr = pca.explained_variance_ratio_  # (50,)
    cum_evr = np.cumsum(evr)
    print(f"  Explained variance ratio:")
    print(f"    component   1: {evr[0]:.4f}  (cumulative: {cum_evr[0]:.4f})")
    print(f"    component   5: {evr[4]:.4f}  (cumulative: {cum_evr[4]:.4f})")
    print(f"    component  10: {evr[9]:.4f}  (cumulative: {cum_evr[9]:.4f})")
    print(f"    component  25: {evr[24]:.4f}  (cumulative: {cum_evr[24]:.4f})")
    print(f"    component  50: {evr[49]:.4f}  (cumulative: {cum_evr[49]:.4f})")
    print(f"  TOTAL variance retained by PCA-{PCA_N_COMPONENTS}: {cum_evr[-1] * 100:.2f}%")
    print(f"  (Recall: PCA-50 from 1536D is expected to retain 60-80%. A number")
    print(f"   at or below that floor would be a methodological flag to raise.)")

    # Save EVR report for downstream inspection
    evr_report = {
        "n_components": int(PCA_N_COMPONENTS),
        "svd_solver": "randomized",
        "random_state": int(PCA_RANDOM_STATE),
        "source_input_path": str(source_input_path) if source_input_path else None,
        "input_shape": [int(vectors_1536d.shape[0]), int(vectors_1536d.shape[1])],
        "explained_variance_ratio": [float(x) for x in evr.tolist()],
        "cumulative_explained_variance_ratio": [float(x) for x in cum_evr.tolist()],
        "total_variance_retained": float(cum_evr[-1]),
        "evr_at_component_50": float(evr[49]),
        "fit_seconds": round(elapsed, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    explained_variance_path.parent.mkdir(parents=True, exist_ok=True)
    with open(explained_variance_path, "w") as f:
        json.dump(evr_report, f, indent=2)
    print(f"  Wrote EVR report to {explained_variance_path}")

    # Save PCA-50 vectors with the same binary-embedding schema as the UMAP
    # production file so downstream code can load them the same way.
    embedding_bytes = [row.tobytes() for row in pca_vectors_f32]
    table = pa.table(
        {
            "patent_id": pa.array(patent_ids, type=pa.string()),
            "embedding": pa.array(embedding_bytes, type=pa.binary()),
        }
    )
    meta = {
        b"embedding_dim": str(PCA_N_COMPONENTS).encode(),
        b"row_count": str(len(patent_ids)).encode(),
        b"source": b"PCA-50 from 1536D PatentSBERTa (text + citation concat)",
        b"source_input_path": (
            str(source_input_path).encode() if source_input_path else b""
        ),
        b"svd_solver": b"randomized",
        b"random_state": str(PCA_RANDOM_STATE).encode(),
        b"total_variance_retained": f"{cum_evr[-1]:.6f}".encode(),
        b"created_at": datetime.now(timezone.utc).isoformat().encode(),
    }
    table = table.replace_schema_metadata(meta)
    pca_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(pca_parquet_path))
    print(f"  Wrote PCA-50 vectors to {pca_parquet_path}")

    return pca_vectors_f32


def load_pca_vectors(pca_parquet_path: Path) -> tuple[list[str], np.ndarray]:
    """Reload PCA-50 vectors from the parquet written by fit_and_save_pca."""
    print(f"  Loading cached PCA-50 vectors from {pca_parquet_path}")
    table = pq.read_table(pca_parquet_path)
    patent_ids = table["patent_id"].to_pylist()
    embs = table["embedding"].to_pylist()
    vectors = np.array(
        [np.frombuffer(b, dtype=np.float32) for b in embs],
        dtype=np.float32,
    )
    print(f"  Loaded {vectors.shape[0]:,} x {vectors.shape[1]}D float32 vectors")
    return patent_ids, vectors


# ---------------------------------------------------------------------------
# Stage 3 + 4: Grouping, tier classification, and deduplication
# ---------------------------------------------------------------------------


def load_dropped_firms() -> set[str]:
    """Load the set of gvkeys dropped by the duplicate firm dedup rule."""
    if not DEDUP_DECISIONS_PATH.exists():
        raise FileNotFoundError(
            f"Deduplication decisions not found at {DEDUP_DECISIONS_PATH}. "
            f"Run scripts/duplicate_firm_unified_rule.py first."
        )
    df = pd.read_parquet(DEDUP_DECISIONS_PATH)
    dropped = set(df["dropped"].tolist())
    print(f"  Loaded {len(dropped)} dropped firms from dedup decisions")
    return dropped


def build_firm_vectors(
    patent_ids: list[str],
    vectors: np.ndarray,
    gvkey_map: pd.DataFrame,
    dropped_firms: set[str],
    min_patents: int,
    single_gaussian_max: int,
) -> tuple[dict, dict, list[tuple[str, int]]]:
    """Group PCA-50 patent vectors by firm, classify tiers, apply dedup filter.

    This mirrors group_and_classify in run_kmax_sweep.py but takes an explicit
    patent_id list (rather than a pid_to_idx dict) and applies the
    deduplication filter inline. We could alternatively call group_and_classify
    and then filter the resulting dicts, but doing it inline makes the
    data flow easier to follow.
    """
    pid_to_idx = {pid: i for i, pid in enumerate(patent_ids)}

    grouped = gvkey_map.groupby("gvkey")["patent_id"].apply(list)

    firm_vectors: dict[str, np.ndarray] = {}
    tier_assignments: dict[str, str] = {}
    excluded_firms: list[tuple[str, int]] = []

    n_dropped_by_dedup = 0
    for gvkey, firm_patent_ids in grouped.items():
        if gvkey in dropped_firms:
            n_dropped_by_dedup += 1
            continue

        indices = [pid_to_idx[pid] for pid in firm_patent_ids if pid in pid_to_idx]
        n = len(indices)

        if n < min_patents:
            excluded_firms.append((gvkey, n))
            continue

        firm_vectors[gvkey] = vectors[indices]
        if n <= single_gaussian_max:
            tier_assignments[gvkey] = "single_gaussian"
        else:
            tier_assignments[gvkey] = "gmm"

    print(f"  Deduplication filter removed: {n_dropped_by_dedup} firms")
    print(f"  Below min_patents ({min_patents}): {len(excluded_firms):,}")
    n_sg = sum(1 for t in tier_assignments.values() if t == "single_gaussian")
    n_gmm = sum(1 for t in tier_assignments.values() if t == "gmm")
    print(f"  Non-excluded: {len(tier_assignments):,} "
          f"(SG: {n_sg:,}, GMM: {n_gmm:,})")

    return firm_vectors, tier_assignments, excluded_firms


# ---------------------------------------------------------------------------
# Stage 6: Fit GMMs at each K_max (reuses run_kmax_sweep helpers)
# ---------------------------------------------------------------------------


def fit_all_firms_pca(
    firm_vectors: dict,
    tier_assignments: dict,
    k_max: int,
    global_mean: np.ndarray,
    global_var: np.ndarray,
    config: dict,
) -> list[dict]:
    """Fit GMMs for all non-excluded firms at a given K_max.

    Thin wrapper around fit_single_gaussian / fit_bayesian_gmm from
    run_kmax_sweep — we don't import fit_all_firms directly because we want
    to control the progress logging label. The behaviour is identical.
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

        if (i + 1) % 500 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            print(
                f"    [K_max={k_max}] [{i+1}/{n_total}] {elapsed:.0f}s elapsed, "
                f"{rate:.1f} firms/s, ETA {eta:.0f}s"
            )

    return results


# ---------------------------------------------------------------------------
# Stage 8: Report — side-by-side PCA vs UMAP
# ---------------------------------------------------------------------------


def load_umap_corrected_summary() -> dict | None:
    """Load the deduplicated + linear-weighted UMAP convergence summary.

    Returns None gracefully if the file doesn't exist yet (the corrected UMAP
    recomputation may not have completed at the time this script runs).
    """
    if not UMAP_CORRECTED_SUMMARY.exists():
        return None
    with open(UMAP_CORRECTED_SUMMARY, "r") as f:
        return json.load(f)


def print_comparison_table(
    pca_adjacent: list[dict],
    pca_non_adjacent: list[dict],
    umap_summary: dict | None,
) -> None:
    """Print a side-by-side PCA vs UMAP convergence table.

    We compare only adjacent transitions that exist in both curves (by k_max_a,
    k_max_b). If UMAP ran on {10,15,20,25,30} and PCA on {10,20,30}, the two
    will share nothing at the adjacent granularity — so we also print the
    non-adjacent (10→30) comparison if available, which is apples-to-apples.
    """
    print()
    print("=" * 80)
    print("PCA-50 vs UMAP-50 convergence comparison")
    print("=" * 80)

    def _fmt_row(label: str, m: dict) -> str:
        rho = m.get("spearman_rho", float("nan"))
        t50 = m.get("top_50_overlap_pct", float("nan"))
        t100 = m.get("top_100_overlap_pct", float("nan"))
        nn5 = m.get("mean_nn5_overlap_pct", float("nan"))
        return (
            f"  {label:<20}  ρ={rho:.4f}  top50={t50:>5.1f}%  "
            f"top100={t100:>5.1f}%  meanNN5={nn5:>5.1f}%"
        )

    print("\nPCA-50 adjacent transitions:")
    for m in pca_adjacent:
        print(_fmt_row(f"K={m['k_max_a']:02d}→K={m['k_max_b']:02d}", m))

    if pca_non_adjacent:
        print("\nPCA-50 non-adjacent comparisons:")
        for m in pca_non_adjacent:
            print(_fmt_row(f"K={m['k_max_a']:02d}→K={m['k_max_b']:02d}", m))

    if umap_summary is None:
        print("\n[UMAP comparison skipped: "
              f"{UMAP_CORRECTED_SUMMARY} not found]")
        print("  → Run scripts/recompute_bc_corrected.py to generate the corrected")
        print("    UMAP summary, then re-run this script with --skip-pca --skip-gmm")
        print("    --skip-bc to regenerate the side-by-side table.")
        return

    umap_adj = umap_summary.get("adjacent_comparisons", [])
    umap_non_adj = umap_summary.get("non_adjacent_comparisons", [])

    print("\nUMAP-50 adjacent transitions (deduplicated + linear-weighted):")
    for m in umap_adj:
        print(_fmt_row(f"K={m['k_max_a']:02d}→K={m['k_max_b']:02d}", m))

    if umap_non_adj:
        print("\nUMAP-50 non-adjacent comparisons:")
        for m in umap_non_adj:
            print(_fmt_row(f"K={m['k_max_a']:02d}→K={m['k_max_b']:02d}", m))

    # Matched-pair comparisons: look up transitions that appear in BOTH lists
    def _key(m: dict) -> tuple[int, int]:
        return (m["k_max_a"], m["k_max_b"])

    umap_by_key: dict[tuple[int, int], dict] = {}
    for m in umap_adj + umap_non_adj:
        umap_by_key[_key(m)] = m
    pca_by_key: dict[tuple[int, int], dict] = {}
    for m in pca_adjacent + pca_non_adjacent:
        pca_by_key[_key(m)] = m

    matched = sorted(set(pca_by_key.keys()) & set(umap_by_key.keys()))
    if matched:
        print("\nMatched comparisons (same K_max transition in both):")
        header = (
            f"  {'transition':<12}  "
            f"{'PCA ρ':>8}  {'UMAP ρ':>8}  "
            f"{'PCA top50':>10}  {'UMAP top50':>10}  "
            f"{'PCA NN5':>8}  {'UMAP NN5':>8}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key in matched:
            p = pca_by_key[key]
            u = umap_by_key[key]
            print(
                f"  K={key[0]:02d}→K={key[1]:02d}    "
                f"{p['spearman_rho']:>8.4f}  {u['spearman_rho']:>8.4f}  "
                f"{p.get('top_50_overlap_pct', float('nan')):>9.1f}%  "
                f"{u.get('top_50_overlap_pct', float('nan')):>9.1f}%  "
                f"{p['mean_nn5_overlap_pct']:>7.1f}%  "
                f"{u['mean_nn5_overlap_pct']:>7.1f}%"
            )
    else:
        print("\n[No matched K_max transitions between PCA {10,20,30} and UMAP "
              f"{umap_summary.get('k_max_values')}.]")
        print("  Use the 10→30 non-adjacent row as the nearest apples-to-apples")
        print("  comparison.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _passes_threshold(m: dict) -> bool:
    return (
        m.get("spearman_rho", 0) > 0.95
        and m.get("top_50_overlap_pct", 0) > 80
    )


def determine_convergence(adjacent_metrics: list[dict]) -> tuple[bool, int | None]:
    """Persistent-stability convergence rule (matches run_kmax_sweep verdict).

    K* = smallest K_max such that this transition AND all subsequent adjacent
    transitions pass both thresholds (Spearman ρ > 0.95, top-50 overlap > 80%).
    """
    adj = sorted(adjacent_metrics, key=lambda m: m["k_max_a"])
    for start_idx in range(len(adj)):
        if all(_passes_threshold(adj[j]) for j in range(start_idx, len(adj))):
            return True, adj[start_idx]["k_max_b"]
    return False, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA-50 comparison sweep (baseline for UMAP-50 convergence).",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the 1536D concatenated embeddings parquet "
             f"(default: {DEFAULT_INPUT_PATH}). "
             f"S3 source: {S3_SOURCE_HINT}",
    )
    parser.add_argument(
        "--k-max-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_MAX_VALUES),
        help=f"K_max values to sweep (default: {list(DEFAULT_K_MAX_VALUES)}).",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run on a small stratified sample (default 100 SG + 100 GMM firms).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Per-tier sample size in test mode (default: 100).",
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Reuse existing PCA-50 parquet if present.",
    )
    parser.add_argument(
        "--skip-gmm",
        action="store_true",
        help="Reuse existing GMM parquets if present.",
    )
    parser.add_argument(
        "--skip-bc",
        action="store_true",
        help="Reuse existing BC matrices if present; only recompute convergence.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    portfolio_cfg = config["portfolio"]

    k_max_values = sorted(args.k_max_values)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Output paths
    pca_parquet_path = OUTPUT_DIR / "patent_vectors_pca50.parquet"
    evr_report_path = OUTPUT_DIR / "pca_explained_variance.json"
    sg_block_path = OUTPUT_DIR / "bc_block_sg_vs_sg_pca.npz"
    convergence_path = OUTPUT_DIR / "convergence_summary_pca.json"

    t_start = time.time()
    print("=" * 80)
    print("PCA-50 Comparison Sweep")
    print(f"  Output dir:  {OUTPUT_DIR}")
    print(f"  K_max grid:  {k_max_values}")
    print(f"  Input path:  {args.input_path}")
    if args.test_mode:
        print(f"  TEST MODE:   {args.test_size} SG + {args.test_size} GMM firms")
    print("=" * 80)

    # -------------------------------------------------------------------
    # Stage 1+2: Load 1536D vectors and fit PCA (or reuse cached PCA-50)
    # -------------------------------------------------------------------
    if args.skip_pca and pca_parquet_path.exists():
        print("\n[Stage 1+2] Reusing cached PCA-50 vectors...")
        patent_ids, pca_vectors = load_pca_vectors(pca_parquet_path)
    else:
        print("\n[Stage 1] Loading 1536D patent vectors...")
        patent_ids, vectors_1536d = load_1536d_vectors(args.input_path)
        print("\n[Stage 2] Fitting PCA(50) and saving PCA-50 vectors...")
        pca_vectors = fit_and_save_pca(
            patent_ids,
            vectors_1536d,
            pca_parquet_path,
            evr_report_path,
            source_input_path=args.input_path,
        )
        # Free the 1536D array (~9 GB) before the heavier GMM stage
        del vectors_1536d

    # -------------------------------------------------------------------
    # Stage 3: Load gvkey map, dedup decisions, and build firm vectors
    # -------------------------------------------------------------------
    print("\n[Stage 3] Grouping patents by firm and classifying tiers...")
    if not GVKEY_MAP_PATH.exists():
        print(f"  ERROR: gvkey map not found at {GVKEY_MAP_PATH}", file=sys.stderr)
        return 2
    gvkey_map = pd.read_parquet(GVKEY_MAP_PATH)
    print(f"  Loaded gvkey map: {len(gvkey_map):,} rows, "
          f"{gvkey_map['gvkey'].nunique():,} firms")

    dropped_firms = load_dropped_firms()

    firm_vectors, tier_assignments, excluded_firms = build_firm_vectors(
        patent_ids,
        pca_vectors,
        gvkey_map,
        dropped_firms,
        min_patents=portfolio_cfg["min_patents"],
        single_gaussian_max=portfolio_cfg["single_gaussian_max"],
    )

    # -------------------------------------------------------------------
    # Test mode: keep a stratified subsample to make local validation
    # feasible in ~5 minutes. Matches the sampling rule in
    # recompute_bc_corrected.py (first-N-per-tier, sorted by gvkey).
    # -------------------------------------------------------------------
    if args.test_mode:
        sg_firms = sorted(
            [gk for gk, t in tier_assignments.items() if t == "single_gaussian"]
        )[: args.test_size]
        gmm_firms = sorted(
            [gk for gk, t in tier_assignments.items() if t == "gmm"]
        )[: args.test_size]
        kept = set(sg_firms) | set(gmm_firms)
        firm_vectors = {gk: firm_vectors[gk] for gk in kept}
        tier_assignments = {gk: tier_assignments[gk] for gk in kept}
        print(
            f"  TEST MODE subsample: {len(firm_vectors)} firms "
            f"(SG: {len(sg_firms)}, GMM: {len(gmm_firms)})"
        )

    # -------------------------------------------------------------------
    # Stage 4: Compute global empirical Bayes priors from PCA-50 matrix
    # Important: priors come from the FULL patent matrix, not per-firm
    # grouped vectors (avoids co-assignment double counting — Codex Major
    # #3 in the design review).
    # -------------------------------------------------------------------
    print("\n[Stage 4] Computing global empirical Bayes priors (PCA-50)...")
    global_mean, global_var = compute_global_priors(pca_vectors)
    print(f"  global_mean range: [{global_mean.min():.4f}, {global_mean.max():.4f}]")
    print(f"  global_var  range: [{global_var.min():.6f}, {global_var.max():.6f}]")
    print(f"  d+2 (= degrees_of_freedom_prior): "
          f"{portfolio_cfg['degrees_of_freedom_prior']} "
          f"(matches PCA-50 dimensionality d=50 → 52)")

    # Release the large full patent array now that we have per-firm vectors
    # and priors. Keep patent_ids around in case we need it later (cheap).
    del pca_vectors

    # -------------------------------------------------------------------
    # Stage 5: Fit GMMs at each K_max
    # -------------------------------------------------------------------
    print("\n[Stage 5] Fitting GMMs at each K_max...")
    all_results: dict[int, list[dict]] = {}
    test_suffix = "_test" if args.test_mode else ""

    for k_max in k_max_values:
        ckpt_path = OUTPUT_DIR / f"firm_gmm_parameters_pca_k{k_max}{test_suffix}.parquet"
        print(f"\n  [K_max={k_max}] -> {ckpt_path}")
        if args.skip_gmm and ckpt_path.exists():
            print("    Reusing cached GMM parquet.")
            all_results[k_max] = load_gmm_results(str(ckpt_path))
        elif ckpt_path.exists() and not args.test_mode:
            # Always reuse a full-run checkpoint unless the user forces a rerun.
            # Test-mode output has a separate suffix so it can't collide.
            print("    Checkpoint exists, reusing.")
            all_results[k_max] = load_gmm_results(str(ckpt_path))
        else:
            t_k = time.time()
            results = fit_all_firms_pca(
                firm_vectors, tier_assignments, k_max,
                global_mean, global_var, config,
            )
            elapsed = time.time() - t_k
            k_summary = compute_effective_k_summary(results, k_max)
            print(
                f"    K_max={k_max} complete in {elapsed:.0f}s "
                f"({elapsed / 60:.1f} min)"
            )
            print(
                f"    Effective K: mean={k_summary.get('mean_k', 'N/A')}, "
                f"median={k_summary.get('median_k', 'N/A')}, "
                f"ceiling={k_summary.get('pct_at_ceiling', 'N/A')}%, "
                f"converged={k_summary.get('converged_pct', 'N/A')}%"
            )
            serialize_gmm_results(results, str(ckpt_path), k_max, config)
            print(f"    Saved to {ckpt_path}")
            all_results[k_max] = results

    # We no longer need the per-firm vectors. Free them before the BC stage.
    del firm_vectors

    # -------------------------------------------------------------------
    # Stage 6: Compute BC matrices using LINEAR weights
    #
    # SG-vs-SG is K_max-invariant for single-Gaussian firms. We compute it
    # once against the K_max=10 result list (which gives us the full SG
    # firm set) and reuse it across K_max values. The component-pair BC
    # kernel (bc_component_matrix) is unchanged between UMAP and PCA
    # because BC on two diagonal Gaussians is agnostic to how the 50D
    # embedding was produced.
    # -------------------------------------------------------------------
    print("\n[Stage 6] Computing pairwise BC matrices (linear weights)...")
    bc_matrices: dict[int, tuple[list[str], np.ndarray]] = {}

    # 6a: SG-vs-SG block (compute or reuse)
    sg_block_cached = None
    sg_gvkeys_cached: list[str] | None = None

    # In test mode the SG firm set changes per run, so we don't trust any
    # cached file — always recompute.
    if (
        not args.test_mode
        and args.skip_bc
        and sg_block_path.exists()
    ):
        print(f"  Loading cached SG-vs-SG block from {sg_block_path}")
        sg_data = np.load(sg_block_path, allow_pickle=True)
        sg_gvkeys_cached = list(sg_data["gvkeys"])
        sg_block_cached = sg_data["bc_matrix"]
    else:
        print("  Computing SG-vs-SG block (K_max-invariant)...")
        first_k = k_max_values[0]
        sg_only = [r for r in all_results[first_k] if r["tier"] == "single_gaussian"]
        t_sg = time.time()
        sg_gvkeys_cached, sg_block_cached = compute_sg_block_linear(sg_only)
        elapsed = time.time() - t_sg
        print(f"    SG block: {sg_block_cached.shape}, {elapsed:.0f}s")
        if not args.test_mode:
            np.savez_compressed(
                sg_block_path,
                gvkeys=np.array(sg_gvkeys_cached),
                bc_matrix=sg_block_cached,
            )
            print(f"    Cached to {sg_block_path}")

    # 6b: Full BC matrix per K_max (reusing SG block)
    for k_max in k_max_values:
        bc_path = OUTPUT_DIR / f"bc_matrix_pca_k{k_max}{test_suffix}.npz"
        print(f"\n  [K_max={k_max}] -> {bc_path}")

        if args.skip_bc and bc_path.exists():
            print("    Reusing cached BC matrix.")
            data = np.load(bc_path, allow_pickle=True)
            bc_matrices[k_max] = (list(data["gvkeys"]), data["bc_matrix"])
            continue

        # Always sort results by gvkey so that bc matrices are comparable
        # across K_max values (required by compute_convergence_metrics).
        results_sorted = sorted(all_results[k_max], key=lambda r: r["gvkey"])

        t_bc = time.time()
        gvkeys, bc_matrix = compute_bc_matrix_linear(
            results_sorted,
            label=f"pca_k{k_max}",
            sg_block=sg_block_cached,
            sg_gvkeys=sg_gvkeys_cached,
        )
        elapsed = time.time() - t_bc
        print(f"    BC matrix: {bc_matrix.shape}, {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        # Sanity check: linear-weighted BC must be bounded in [0, 1]
        upper = bc_matrix[np.triu_indices_from(bc_matrix, k=1)]
        max_bc = float(upper.max())
        n_above_1 = int(np.sum(upper > 1.0001))
        print(f"    Sanity: max BC={max_bc:.6f}, #pairs > 1.0001 = {n_above_1}")
        if n_above_1 > 0:
            print("    WARNING: linear-weighted BC should be bounded in [0,1]. "
                  "Investigate potential numerical issue.")

        np.savez_compressed(bc_path, gvkeys=np.array(gvkeys), bc_matrix=bc_matrix)
        print(f"    Saved to {bc_path}")
        bc_matrices[k_max] = (gvkeys, bc_matrix)

    # -------------------------------------------------------------------
    # Stage 7: Convergence metrics
    # -------------------------------------------------------------------
    print("\n[Stage 7] Computing convergence metrics...")
    sorted_kmax = sorted(bc_matrices.keys())
    adjacent_metrics: list[dict] = []
    non_adjacent_metrics: list[dict] = []

    for i in range(len(sorted_kmax) - 1):
        k_a = sorted_kmax[i]
        k_b = sorted_kmax[i + 1]
        gvkeys_a, bc_a = bc_matrices[k_a]
        gvkeys_b, bc_b = bc_matrices[k_b]
        m = compute_convergence_metrics(gvkeys_a, bc_a, gvkeys_b, bc_b, k_a, k_b)
        adjacent_metrics.append(m)
        print(
            f"  K={k_a}→K={k_b}: ρ={m['spearman_rho']:.4f}  "
            f"top50={m['top_50_overlap_pct']:.1f}%  "
            f"top100={m['top_100_overlap_pct']:.1f}%  "
            f"meanNN5={m['mean_nn5_overlap_pct']:.1f}%  "
            f"{'PASS' if _passes_threshold(m) else 'FAIL'}"
        )

    # Non-adjacent: always include min vs max for a robustness read
    if len(sorted_kmax) >= 2:
        k_a, k_b = sorted_kmax[0], sorted_kmax[-1]
        gvkeys_a, bc_a = bc_matrices[k_a]
        gvkeys_b, bc_b = bc_matrices[k_b]
        m = compute_convergence_metrics(gvkeys_a, bc_a, gvkeys_b, bc_b, k_a, k_b)
        non_adjacent_metrics.append(m)
        print(
            f"  [non-adj] K={k_a}→K={k_b}: ρ={m['spearman_rho']:.4f}  "
            f"top50={m['top_50_overlap_pct']:.1f}%  "
            f"meanNN5={m['mean_nn5_overlap_pct']:.1f}%"
        )

    converged, converged_at = determine_convergence(adjacent_metrics)

    # -------------------------------------------------------------------
    # Stage 8: Save PCA summary, report, and PCA-vs-UMAP comparison
    # -------------------------------------------------------------------
    elapsed_total = time.time() - t_start

    # Recover effective-K summary per K_max (same shape as UMAP sweep)
    effective_k_summaries: dict[str, dict] = {}
    for k_max in k_max_values:
        effective_k_summaries[str(k_max)] = compute_effective_k_summary(
            all_results[k_max], k_max
        )

    summary = {
        "run_kind": "pca_comparison_sweep",
        "representation": "pca50",
        "source_dimensionality": 1536,
        "k_max_values": k_max_values,
        "bc_formula": "linear_weights_pi_i_times_pi_j",
        "dedup_applied": True,
        "dedup_decisions_path": str(DEDUP_DECISIONS_PATH),
        "n_firms": len(bc_matrices[sorted_kmax[0]][0]),
        "convergence_verdict": "converged" if converged else "not_converged",
        "converged_at_kmax": converged_at,
        "decision_rule": {
            "spearman_threshold": 0.95,
            "top_50_overlap_threshold_pct": 80,
            "method": "persistent_stability",
            "definition": "K* = smallest K_max such that all subsequent adjacent "
                          "comparisons from K* onward pass both thresholds",
        },
        "adjacent_comparisons": adjacent_metrics,
        "non_adjacent_comparisons": non_adjacent_metrics,
        "effective_k_summaries": effective_k_summaries,
        "test_mode": bool(args.test_mode),
        "timing": {
            "total_seconds": round(elapsed_total, 1),
            "total_minutes": round(elapsed_total / 60, 1),
            "total_hours": round(elapsed_total / 3600, 2),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Attach PCA EVR snapshot if available (it's in a separate file but it's
    # convenient to have the summary self-contained for downstream notebooks).
    if evr_report_path.exists():
        with open(evr_report_path, "r") as f:
            summary["pca_explained_variance"] = json.load(f)

    with open(convergence_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved convergence summary to {convergence_path}")

    # -------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------
    print()
    print("=" * 80)
    print("PCA-50 Comparison Sweep — Final Report")
    print("=" * 80)
    print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total / 60:.1f} min)")
    print(f"  K_max values tested: {k_max_values}")
    print(f"  Firms fitted: {len(bc_matrices[sorted_kmax[0]][0]):,}")
    print(f"  Test mode: {args.test_mode}")
    print()
    if converged:
        print(f"  PCA VERDICT: CONVERGED at K_max={converged_at}")
    else:
        print(f"  PCA VERDICT: NOT CONVERGED by K_max={k_max_values[-1]}")

    umap_summary = load_umap_corrected_summary()
    print_comparison_table(adjacent_metrics, non_adjacent_metrics, umap_summary)

    print()
    print("=" * 80)
    print("Sweep complete")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
