"""Synthetic merger case study — prototype for Step 4 (Arthur, 2026-04-19).

Scope: four hand-picked (acquirer, target) pairs. For each pair, concatenate
patent vectors, re-fit a Bayesian GMM with the same production priors and
hyperparameters, and compute BC(synthetic, f) for the top-20 firms by
BC(acquirer, f). Report ΔBC = BC(synthetic, f) − BC(acquirer, f) per comparator.

This is NOT the production synthetic-portfolio methodology. No ADR/spec approval,
no unit tests. See docs/epics/week2_firm_portfolios/synthetic_merger_case_study.md
for scope, limitations, and the promotion path.

Null-by-construction: |ΔBC| is bounded on the order of target_patent_share.
For pair 4 (target share 0.023%) the synthetic distribution is
indistinguishable from the acquirer's. This is a math result, not a bug.

Usage:
    source venv/bin/activate
    python scripts/case_study_synthetic_mergers.py

    # Single pair (dry-run):
    python scripts/case_study_synthetic_mergers.py --pairs 001161:022325

    # Custom K_max / comparator count:
    python scripts/case_study_synthetic_mergers.py --k-max 15 --n-comparators 20
"""

import argparse
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from scripts.run_kmax_sweep import (
    load_inputs,
    compute_global_priors,
    fit_single_gaussian,
    fit_bayesian_gmm,
    load_gmm_results,
    serialize_gmm_results,
)
from scripts.recompute_bc_corrected import bc_mixture_linear


# ---------------------------------------------------------------------------
# Hardcoded pairs (from Arthur's email, 2026-04-19)
# ---------------------------------------------------------------------------

PAIRS = [
    ("001161", "022325"),  # 17347 + 4466  | target share 20.5%
    ("001632", "014256"),  #  4320 + 1123  | target share 20.6%
    ("007257", "018510"),  # 12415 +   99  | target share  0.79% — WEAK
    ("005606", "008633"),  # 39719 +    9  | target share  0.023% — NULL BY CONSTRUCTION
]

OUTPUT_DIR = Path("output/case_studies/synthetic_mergers")
GMM_PARAMS_PATH = "output/kmax_sweep/corrected/output/kmax_sweep/firm_gmm_parameters_k15.parquet"
BC_MATRIX_PATH = "output/kmax_sweep/corrected/output/kmax_sweep/bc_matrix_all_k15_dedup_linear.npz"

# Null-by-construction / weak-signal thresholds on target_patent_share
NULL_BY_CONSTRUCTION_THRESHOLD = 0.01   # <1% target share → ΔBC is noise-bounded
WEAK_SIGNAL_THRESHOLD = 0.05             # <5% target share → interpret with caution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_firm_vectors(
    gvkey: str,
    gvkey_map: pd.DataFrame,
    vectors: np.ndarray,
    pid_to_idx: dict,
) -> np.ndarray:
    """Return (n_patents, 50) float64 matrix for a single firm.

    Mirrors run_kmax_sweep.group_and_classify patent-lookup: skip patent_ids
    not in pid_to_idx (the 50D vector file doesn't cover every row in gvkey_map).
    Casts to float64 here — downstream fitters expect float64.
    """
    patent_ids = gvkey_map.loc[gvkey_map["gvkey"] == gvkey, "patent_id"].tolist()
    indices = [pid_to_idx[pid] for pid in patent_ids if pid in pid_to_idx]
    if not indices:
        return np.empty((0, 50), dtype=np.float64)
    return vectors[indices].astype(np.float64, copy=False)


def load_bc_lookup(path: str) -> tuple[dict, np.ndarray]:
    """Load the deduplicated BC matrix; return (gvkey_to_idx, bc_matrix)."""
    data = np.load(path, allow_pickle=True)
    gvkeys = list(data["gvkeys"])
    bc_matrix = data["bc_matrix"].astype(np.float64, copy=False)
    assert bc_matrix.shape == (len(gvkeys), len(gvkeys)), (
        f"BC shape {bc_matrix.shape} does not match gvkey count {len(gvkeys)}"
    )
    return {gv: i for i, gv in enumerate(gvkeys)}, bc_matrix


def top_n_comparators(
    acquirer: str,
    target: str,
    gvkey_to_idx: dict,
    bc_matrix: np.ndarray,
    n: int,
) -> list[tuple[str, float, int]]:
    """Return the n firms with highest BC(acquirer, *), excluding acquirer and target.

    Any gvkey not in the deduplicated BC universe is also excluded (backfill past
    rank n if needed). Returns [(comparator_gvkey, bc_to_acquirer, rank), ...].
    """
    if acquirer not in gvkey_to_idx:
        raise KeyError(f"acquirer {acquirer} not in deduplicated BC universe")

    row_idx = gvkey_to_idx[acquirer]
    # Sort all gvkeys by BC(acquirer, gvkey) descending
    # Build (gvkey, bc) in one pass
    bc_row = bc_matrix[row_idx]
    # argsort descending; then filter
    order = np.argsort(-bc_row)

    idx_to_gvkey = {i: g for g, i in gvkey_to_idx.items()}
    excluded = {acquirer, target}
    out: list[tuple[str, float, int]] = []
    rank = 0
    for j in order:
        gv = idx_to_gvkey[int(j)]
        if gv in excluded:
            continue
        rank += 1
        out.append((gv, float(bc_row[j]), rank))
        if len(out) >= n:
            break
    return out


def fit_synthetic_gmm(
    synth_gvkey: str,
    X: np.ndarray,
    k_max: int,
    global_mean: np.ndarray,
    global_var: np.ndarray,
    config: dict,
) -> dict:
    """Dispatch to single_gaussian or bayesian fitter; mirrors fit_all_firms logic."""
    pc = config["portfolio"]
    n = len(X)
    if n < pc["min_patents"]:
        raise ValueError(
            f"{synth_gvkey}: combined portfolio has {n} patents, below "
            f"min_patents={pc['min_patents']}"
        )
    if n <= pc["single_gaussian_max"]:
        return fit_single_gaussian(synth_gvkey, X)
    return fit_bayesian_gmm(synth_gvkey, X, k_max, global_mean, global_var, config)


def run_synthetic_merger(
    acquirer: str,
    target: str,
    pair_idx: int,
    gvkey_map: pd.DataFrame,
    vectors: np.ndarray,
    pid_to_idx: dict,
    global_mean: np.ndarray,
    global_var: np.ndarray,
    existing_gmms: dict,
    bc_matrix: np.ndarray,
    gvkey_to_idx: dict,
    config: dict,
    k_max: int,
    n_comparators: int,
) -> tuple[dict, dict, list[dict], dict]:
    """Run the full per-pair pipeline.

    Returns:
        synthetic_gmm, acquirer_refit_gmm, comparator_rows, pair_metadata

    Why we also refit the acquirer alone:
        The synthetic is a fresh Bayesian VI fit; the production acquirer GMM is
        another fresh Bayesian VI fit — but they were run at different times with
        different data (39K acq-only vs 39K+9 combined). Even with identical priors
        and random_state=42, VI lands on different local optima in a 50-D K=15
        mixture — observed empirically via BC(synth, acq_production) = 0.03-0.06.
        Refitting the acquirer alone with the same priors/seed/n_init as the
        synthetic gives a comparable baseline: delta_bc_clean = BC(synth, f) −
        BC(acq_refit, f) cancels the refit stochasticity and isolates the merger
        effect.
    """
    t0 = time.time()

    # 1. Build vectors for acquirer and target
    X_acq = build_firm_vectors(acquirer, gvkey_map, vectors, pid_to_idx)
    X_tgt = build_firm_vectors(target, gvkey_map, vectors, pid_to_idx)

    # 2. Patent-id overlap (surface; don't silently dedup)
    acq_pids = set(gvkey_map.loc[gvkey_map["gvkey"] == acquirer, "patent_id"])
    tgt_pids = set(gvkey_map.loc[gvkey_map["gvkey"] == target, "patent_id"])
    shared_pids = acq_pids & tgt_pids
    n_shared = len(shared_pids)

    # 3. Combined vector matrix — concatenate (zero-overlap was verified upstream
    # for the hardcoded 4 pairs; we surface n_shared so future swapped pairs are caught).
    X_combined = np.concatenate([X_acq, X_tgt], axis=0)
    n_acq, n_tgt = len(X_acq), len(X_tgt)
    n_combined = len(X_combined)
    target_share = n_tgt / n_combined if n_combined else 0.0

    # 4a. Fit synthetic GMM (combined portfolio)
    synth_gvkey = f"SYNTH_{acquirer}_{target}"
    synth_gmm = fit_synthetic_gmm(
        synth_gvkey, X_combined, k_max, global_mean, global_var, config
    )

    # 4b. Fit acquirer alone from scratch — refit-noise baseline
    refit_gvkey = f"REFIT_{acquirer}"
    acq_refit_gmm = fit_synthetic_gmm(
        refit_gvkey, X_acq, k_max, global_mean, global_var, config
    )

    # 5. Comparator set — top-N by BC(acquirer, *), excl. acquirer + target
    comparators = top_n_comparators(acquirer, target, gvkey_to_idx, bc_matrix, n_comparators)

    # 6. Pre-merger BC(acquirer, target)
    pre_merger_bc = float(bc_matrix[gvkey_to_idx[acquirer], gvkey_to_idx[target]])

    # 7. Compute BC(synthetic, *) AND BC(acq_refit, *) against comparators + self rows
    comparator_rows: list[dict] = []

    def _row(comp_gv: str, role: str, rank: int) -> dict:
        comp_idx = gvkey_to_idx.get(comp_gv)
        bc_acq = float(bc_matrix[gvkey_to_idx[acquirer], comp_idx]) if comp_idx is not None else float("nan")
        bc_tgt = float(bc_matrix[gvkey_to_idx[target], comp_idx]) if comp_idx is not None else float("nan")
        bc_synth = bc_mixture_linear(synth_gmm, existing_gmms[comp_gv])
        bc_acq_refit = bc_mixture_linear(acq_refit_gmm, existing_gmms[comp_gv])
        comp_info = existing_gmms.get(comp_gv, {})
        return {
            "pair_id": pair_idx,
            "acquirer": acquirer,
            "target": target,
            "comparator": comp_gv,
            "role": role,
            "bc_acquirer_to_comparator": bc_acq,
            "bc_target_to_comparator": bc_tgt,
            "bc_synthetic_to_comparator": bc_synth,
            "bc_acquirer_refit_to_comparator": bc_acq_refit,
            "delta_bc": bc_synth - bc_acq,                     # raw (mixes merger + refit noise)
            "delta_bc_clean": bc_synth - bc_acq_refit,         # noise-cancelled merger effect
            "refit_noise": bc_acq_refit - bc_acq,              # production GMM vs refit GMM alignment
            "rank_in_acquirer_top20": rank,
            "target_patent_share": target_share,
            "comparator_n_patents": int(comp_info.get("n_patents", -1)),
            "comparator_tier": comp_info.get("tier", ""),
        }

    for comp_gv, _bc, rank in comparators:
        comparator_rows.append(_row(comp_gv, "comparator", rank))
    comparator_rows.append(_row(acquirer, "acquirer_self", -1))
    comparator_rows.append(_row(target, "target_self", -1))

    # 8. Pair metadata — compute summaries on "comparator" rows only
    comp_only = [r for r in comparator_rows if r["role"] == "comparator"]

    def _max_abs(values: list[float], rows: list[dict]) -> tuple[float, str]:
        if not values:
            return 0.0, ""
        idx = int(np.argmax(np.abs(values)))
        return values[idx], rows[idx]["comparator"]

    raw_deltas = [r["delta_bc"] for r in comp_only]
    clean_deltas = [r["delta_bc_clean"] for r in comp_only]
    noise_values = [r["refit_noise"] for r in comp_only]
    max_delta, argmax_gv = _max_abs(raw_deltas, comp_only)
    max_delta_clean, argmax_clean_gv = _max_abs(clean_deltas, comp_only)
    max_abs_noise, _ = _max_abs(noise_values, comp_only)

    # BC(acq_refit, acq_production) — how much VI landed on a different optimum
    bc_refit_vs_prod = bc_mixture_linear(acq_refit_gmm, existing_gmms[acquirer])

    pair_metadata = {
        "pair_id": pair_idx,
        "acquirer": acquirer,
        "target": target,
        "synthetic_gvkey": synth_gvkey,
        "acquirer_refit_gvkey": refit_gvkey,
        "n_patents_acquirer": n_acq,
        "n_patents_target": n_tgt,
        "n_patents_combined": n_combined,
        "n_shared_patents": n_shared,
        "target_patent_share": target_share,
        "pre_merger_bc": pre_merger_bc,
        "synthetic_k_effective": int(synth_gmm["n_components"]),
        "synthetic_tier": synth_gmm["tier"],
        "synthetic_converged": bool(synth_gmm["converged"]),
        "synthetic_n_iter": int(synth_gmm["n_iter"]),
        "acquirer_refit_k_effective": int(acq_refit_gmm["n_components"]),
        "acquirer_refit_tier": acq_refit_gmm["tier"],
        "acquirer_refit_converged": bool(acq_refit_gmm["converged"]),
        "acquirer_refit_n_iter": int(acq_refit_gmm["n_iter"]),
        "bc_acquirer_refit_vs_production": float(bc_refit_vs_prod),
        "null_by_construction": bool(target_share < NULL_BY_CONSTRUCTION_THRESHOLD),
        "weak_signal": bool(target_share < WEAK_SIGNAL_THRESHOLD),
        "max_delta_bc": float(max_delta),
        "argmax_delta_bc_comparator": argmax_gv,
        "max_delta_bc_clean": float(max_delta_clean),
        "argmax_delta_bc_clean_comparator": argmax_clean_gv,
        "max_abs_refit_noise": float(max_abs_noise),
        "fit_elapsed_s": round(time.time() - t0, 2),
    }

    print(
        f"    pair {pair_idx} done: n_combined={n_combined} target_share={target_share:.4f} "
        f"synth_k={pair_metadata['synthetic_k_effective']} refit_k={pair_metadata['acquirer_refit_k_effective']} "
        f"BC(refit,prod_acq)={bc_refit_vs_prod:.4f} "
        f"max|ΔBC_raw|={abs(max_delta):.4f} ({argmax_gv}) "
        f"max|ΔBC_clean|={abs(max_delta_clean):.4f} ({argmax_clean_gv}) "
        f"{pair_metadata['fit_elapsed_s']:.1f}s"
    )
    return synth_gmm, acq_refit_gmm, comparator_rows, pair_metadata


# ---------------------------------------------------------------------------
# Output serialization
# ---------------------------------------------------------------------------

def write_parquet(rows: list[dict], path: Path, column_types: dict = None) -> None:
    """Write rows to parquet with explicit schema hints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Explicit int32 where requested
    if column_types:
        for col, dtype in column_types.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
    df.to_parquet(path, index=False)


def write_run_metadata(path: Path, pair_metas: list[dict], k_max: int, n_comparators: int) -> None:
    """Write run metadata JSON — git SHA + timestamp + per-pair timings."""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    meta = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git_sha": git_sha,
        "k_max": k_max,
        "n_comparators": n_comparators,
        "gmm_params_source": GMM_PARAMS_PATH,
        "bc_matrix_source": BC_MATRIX_PATH,
        "per_pair_elapsed_s": [p["fit_elapsed_s"] for p in pair_metas],
        "total_elapsed_s": round(sum(p["fit_elapsed_s"] for p in pair_metas), 2),
        "provenance": "prototype — NOT production methodology (see synthetic_merger_case_study.md)",
    }
    path.write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_pairs_cli(raw: str) -> list[tuple[str, str]]:
    """Parse --pairs CLI arg: 'acq:tgt,acq:tgt' → [(acq, tgt), ...]. Keep as strings."""
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) != 2:
            raise ValueError(f"Expected acq:tgt, got {tok!r}")
        acq, tgt = parts[0].strip(), parts[1].strip()
        for gv in (acq, tgt):
            if not isinstance(gv, str) or len(gv) < 3:
                raise ValueError(f"Suspicious gvkey {gv!r} — keep as string with leading zeros")
        out.append((acq, tgt))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--pairs", type=str, default=None,
        help="Override PAIRS: 'acq:tgt,acq:tgt'. Default uses hardcoded PAIRS.",
    )
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--n-comparators", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    pairs = _parse_pairs_cli(args.pairs) if args.pairs else PAIRS
    print("=" * 70)
    print("SYNTHETIC MERGER CASE STUDY")
    print("=" * 70)
    print(f"Pairs: {pairs}")
    print(f"K_max={args.k_max}, n_comparators={args.n_comparators}")
    print(f"Output: {args.output_dir}")

    # 1. Load config and production artifacts once
    print("\n[1/6] Loading config + production artifacts...")
    config = load_config()

    vectors, gvkey_map, pid_to_idx = load_inputs(config, local=True)
    vectors_f64_for_priors = vectors.astype(np.float64, copy=False)

    print("  Computing global empirical priors from unique patent matrix...")
    global_mean, global_var = compute_global_priors(vectors_f64_for_priors)
    print(f"  global_mean shape={global_mean.shape}, global_var shape={global_var.shape}")

    print(f"  Loading existing K={args.k_max} GMM parameters from {GMM_PARAMS_PATH}")
    existing_results = load_gmm_results(GMM_PARAMS_PATH)
    existing_gmms = {r["gvkey"]: r for r in existing_results}
    print(f"  Loaded {len(existing_gmms)} existing firm GMMs")

    print(f"  Loading existing BC matrix from {BC_MATRIX_PATH}")
    gvkey_to_idx, bc_matrix = load_bc_lookup(BC_MATRIX_PATH)
    print(f"  BC matrix shape={bc_matrix.shape}; {len(gvkey_to_idx)} deduplicated firms")

    # 2. Sanity-check all input gvkeys present (fail fast)
    print("\n[2/6] Verifying input gvkeys...")
    input_gvkeys = {gv for pair in pairs for gv in pair}
    missing_in_gmm = input_gvkeys - existing_gmms.keys()
    missing_in_bc = input_gvkeys - gvkey_to_idx.keys()
    if missing_in_gmm:
        raise ValueError(f"gvkeys missing from GMM parquet: {missing_in_gmm}")
    if missing_in_bc:
        raise ValueError(f"gvkeys missing from BC matrix (may have been deduplicated): {missing_in_bc}")
    print(f"  All {len(input_gvkeys)} input gvkeys present in both GMM file and BC matrix")

    # 3. Per-pair runs
    print("\n[3/6] Running synthetic mergers (with acquirer-refit baseline)...")
    all_synthetic_gmms: list[dict] = []
    all_acquirer_refit_gmms: list[dict] = []
    all_comparator_rows: list[dict] = []
    all_pair_metas: list[dict] = []

    for i, (acq, tgt) in enumerate(pairs, start=1):
        print(f"\n  Pair {i}/{len(pairs)}: acq={acq} tgt={tgt}")
        synth_gmm, acq_refit_gmm, comparator_rows, pair_meta = run_synthetic_merger(
            acq, tgt, i,
            gvkey_map, vectors, pid_to_idx,
            global_mean, global_var,
            existing_gmms, bc_matrix, gvkey_to_idx,
            config, args.k_max, args.n_comparators,
        )
        all_synthetic_gmms.append(synth_gmm)
        all_acquirer_refit_gmms.append(acq_refit_gmm)
        all_comparator_rows.extend(comparator_rows)
        all_pair_metas.append(pair_meta)

    # 4. Assertions
    print("\n[4/6] Running assertions...")
    _run_assertions(all_pair_metas, all_comparator_rows, args.n_comparators)

    # 5. Serialize outputs
    print("\n[5/6] Serializing outputs...")
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_path = out_dir / "synthetic_merger_comparators_k15.parquet"
    write_parquet(all_comparator_rows, comp_path, column_types={
        "pair_id": "int32",
        "rank_in_acquirer_top20": "int32",
        "comparator_n_patents": "int32",
    })
    print(f"  Wrote {comp_path} ({len(all_comparator_rows)} rows)")

    pair_path = out_dir / "synthetic_merger_pairs_k15.parquet"
    write_parquet(all_pair_metas, pair_path, column_types={
        "pair_id": "int32",
        "n_patents_acquirer": "int32",
        "n_patents_target": "int32",
        "n_patents_combined": "int32",
        "n_shared_patents": "int32",
        "synthetic_k_effective": "int32",
        "synthetic_n_iter": "int32",
        "acquirer_refit_k_effective": "int32",
        "acquirer_refit_n_iter": "int32",
    })
    print(f"  Wrote {pair_path} ({len(all_pair_metas)} rows)")

    # Combined GMM parquet — synthetic firms + acquirer-refit baselines (schema-compliant)
    all_refit_gmms = all_synthetic_gmms + all_acquirer_refit_gmms
    synth_gmm_path = out_dir / "synthetic_firm_gmm_parameters_k15.parquet"
    serialize_gmm_results(all_refit_gmms, str(synth_gmm_path), args.k_max, config)
    # Round-trip sanity — assertion #10
    reloaded = load_gmm_results(str(synth_gmm_path))
    assert len(reloaded) == len(all_refit_gmms), "refit GMM round-trip row count mismatch"
    print(
        f"  Wrote {synth_gmm_path} ({len(all_refit_gmms)} rows = "
        f"{len(all_synthetic_gmms)} synth + {len(all_acquirer_refit_gmms)} acq_refit; "
        f"load_gmm_results round-trip OK)"
    )

    meta_path = out_dir / "run_metadata.json"
    write_run_metadata(meta_path, all_pair_metas, args.k_max, args.n_comparators)
    print(f"  Wrote {meta_path}")

    # 6. Summary
    print("\n[6/6] Summary:")
    for m in all_pair_metas:
        flag = ""
        if m["null_by_construction"]:
            flag = "  ← target share < 1%"
        elif m["weak_signal"]:
            flag = "  ← target share < 5%"
        print(
            f"  pair {m['pair_id']}: {m['acquirer']}+{m['target']}  "
            f"target_share={m['target_patent_share']:.4f}  "
            f"BC(refit,prod_acq)={m['bc_acquirer_refit_vs_production']:.4f}  "
            f"max|ΔBC_raw|={abs(m['max_delta_bc']):.4f} ({m['argmax_delta_bc_comparator']})  "
            f"max|ΔBC_clean|={abs(m['max_delta_bc_clean']):.4f} ({m['argmax_delta_bc_clean_comparator']}){flag}"
        )
    print("\nDone.")


def _run_assertions(pair_metas: list[dict], comparator_rows: list[dict], n_comparators: int) -> None:
    """Hard sanity assertions before serialization (plan Step 5)."""
    # Shape
    expected_comp_rows = len(pair_metas) * (n_comparators + 2)
    assert len(comparator_rows) == expected_comp_rows, (
        f"Expected {expected_comp_rows} comparator rows, got {len(comparator_rows)}"
    )

    # BC bounds on all four BC columns
    bc_cols = (
        "bc_acquirer_to_comparator",
        "bc_target_to_comparator",
        "bc_synthetic_to_comparator",
        "bc_acquirer_refit_to_comparator",
    )
    for r in comparator_rows:
        for k in bc_cols:
            v = r[k]
            if not (np.isnan(v) or (0.0 <= v <= 1.0001)):
                raise AssertionError(f"{k}={v} out of [0, 1.0001] in row {r}")

    # acquirer_self row: production BC(acq,acq) ≈ 1.0 (self-identity of production matrix)
    for r in comparator_rows:
        if r["role"] == "acquirer_self":
            assert abs(r["bc_acquirer_to_comparator"] - 1.0) < 1e-6, (
                f"acquirer_self row BC(acq,acq) = {r['bc_acquirer_to_comparator']} != 1.0"
            )

    # No shared patents in the hardcoded 4 pairs (asserted for safety)
    assert sum(m["n_shared_patents"] for m in pair_metas) == 0, (
        "Unexpected shared patents — investigate before claiming vector concat is correct"
    )

    # Unique synthetic + refit gvkeys
    synth_keys = [m["synthetic_gvkey"] for m in pair_metas]
    refit_keys = [m["acquirer_refit_gvkey"] for m in pair_metas]
    assert len(set(synth_keys)) == len(synth_keys), "Duplicate synthetic gvkeys"
    assert len(set(refit_keys)) == len(refit_keys), "Duplicate refit gvkeys"
    assert not (set(synth_keys) & set(refit_keys)), "Synthetic and refit gvkey namespaces collide"

    # Clean-ΔBC sanity: for pair 4 (target ≈ 0.023% of merged), |delta_bc_clean|
    # should be small — the acquirer-refit baseline cancels the dominant refit
    # noise source so the residual should be tiny.
    for m in pair_metas:
        if m["acquirer"] == "005606" and m["target"] == "008633":
            mx_clean = abs(m["max_delta_bc_clean"])
            mx_raw = abs(m["max_delta_bc"])
            if mx_clean >= mx_raw:
                warnings.warn(
                    f"Pair 4 max|ΔBC_clean|={mx_clean:.4f} >= max|ΔBC_raw|={mx_raw:.4f} — "
                    "expected baseline to cancel refit noise, not amplify it"
                )

    print(f"  {len(comparator_rows)} comparator rows + {len(pair_metas)} pair rows; all checks passed")


if __name__ == "__main__":
    main()
