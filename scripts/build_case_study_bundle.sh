#!/bin/bash
# Assembles the synthetic merger case-study bundle for transfer to Arthur.
# Does NOT commit to git; staging area only.
# Kept separate from scripts/build_handover_bundle.sh so the Week 2 handover
# bundle (already distributed) stays untouched.
#
# Usage: bash scripts/build_case_study_bundle.sh
# Output: output/case_study_bundle_20260421/ + MANIFEST.txt + SHA256SUMS.txt

set -euo pipefail

STAGE="output/case_study_bundle_20260421"
CASE_DIR="output/case_studies/synthetic_mergers"

# --- Fail fast if source directory is missing ---
if [ ! -d "$CASE_DIR" ]; then
    echo "ERROR: case-study output directory not found: $CASE_DIR" >&2
    echo "Run: python scripts/case_study_synthetic_mergers.py" >&2
    exit 1
fi

mkdir -p "$STAGE"

# --- Core outputs ---
cp "$CASE_DIR/synthetic_merger_pairs_k15.parquet"       "$STAGE/"
cp "$CASE_DIR/synthetic_merger_comparators_k15.parquet" "$STAGE/"
cp "$CASE_DIR/synthetic_firm_gmm_parameters_k15.parquet" "$STAGE/"
cp "$CASE_DIR/run_metadata.json"                         "$STAGE/"

# --- Walkthrough notebook ---
if [ -f "notebooks/05_synthetic_merger_case_study.ipynb" ]; then
    cp "notebooks/05_synthetic_merger_case_study.ipynb" "$STAGE/"
else
    echo "WARN: notebooks/05_synthetic_merger_case_study.ipynb not found" >&2
fi

# --- Methodology doc (so the bundle is self-contained for Arthur) ---
cp "docs/epics/week2_firm_portfolios/synthetic_merger_case_study.md" "$STAGE/"

# --- MANIFEST.txt ---
GIT_SHA=$(git rev-parse --short=12 HEAD 2>/dev/null || echo "unknown")
BUILT_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$STAGE/MANIFEST.txt" <<EOF
Bundle:  Synthetic Merger Case Study — Prototype (Arthur's request, 2026-04-19)
Built:   $BUILT_AT (UTC)
Git SHA: $GIT_SHA
Source:  output/case_studies/synthetic_mergers/ (local run)
Status:  PROTOTYPE — NOT production methodology. See synthetic_merger_case_study.md §8.

Contents:
  synthetic_merger_pairs_k15.parquet        - 4 rows: per-pair metadata, target shares, raw + clean argmax ΔBC, refit diagnostics
  synthetic_merger_comparators_k15.parquet  - 88 rows: 4 pairs × 20 comparators + 2 self-sanity rows; raw + clean ΔBC + refit_noise
  synthetic_firm_gmm_parameters_k15.parquet - 8 rows: 4 synthetic firms + 4 acquirer-refit baselines (load via scripts/run_kmax_sweep.load_gmm_results)
  run_metadata.json                          - git SHA, K_max, input sources, per-pair timings
  05_synthetic_merger_case_study.ipynb       - walkthrough notebook (start here)
  synthetic_merger_case_study.md             - methodology record + promotion path
  MANIFEST.txt                               - this file
  SHA256SUMS.txt                             - integrity checksums

Key finding to read BEFORE the per-pair numbers (Notebook Section 4):
  An acquirer-refit baseline was run per pair to test whether VI refit
  stochasticity contaminates ΔBC measurements to external firms. Result:
  refit noise on external-firm BCs is effectively zero (|raw ΔBC − clean ΔBC|
  bounded by ~4e-5 across all 88 comparator rows). Raw and clean ΔBC are
  interchangeable at the refit-noise level.

  However, the baseline exposed a DIFFERENT remaining confound: VI-
  sensitivity to small data perturbations. Pair 4 (9 target patents into a
  40K-patent acquirer, 0.023% share) still shows |ΔBC|=0.019 — mechanically
  impossible for the target's content to explain. This floor applies to all
  pairs; only pair 1 (20.5% share, ΔBC=0.027 on PRIV_AILTECHNOLOGIES) plausibly
  exceeds it.

  Cleanly separating merger effect from VI-perturbation requires a random-
  patent-injection null-distribution calibration (notebook §10, methodology
  doc §6.2 + §8) as the natural follow-up. Until that's run, no per-pair
  ΔBC should be cited as a causal finding in the paper.

Verification (Linux/Mac/WSL):
  cd <bundle>
  sha256sum -c SHA256SUMS.txt
EOF

# --- Integrity checksums (remove stale first for idempotent re-runs) ---
rm -f "$STAGE/SHA256SUMS.txt"
(cd "$STAGE" && sha256sum * > SHA256SUMS.txt)

N_FILES=$(ls "$STAGE" | wc -l)
TOTAL_MB=$(du -sm "$STAGE" | cut -f1)
echo "Bundle staged at $STAGE with $N_FILES files (${TOTAL_MB} MB)"
echo "MANIFEST: git SHA $GIT_SHA, built $BUILT_AT"
echo "Verify integrity: cd $STAGE && sha256sum -c SHA256SUMS.txt"
