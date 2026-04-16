#!/bin/bash
# Assembles the Week 2 handover bundle into a staging directory for email/link transfer.
# Does NOT commit to git; this is a transfer staging area.
# Idempotent — re-running overwrites STAGE contents.
#
# Usage: bash scripts/build_handover_bundle.sh
# Output: output/handover_bundle_20260416/ (staging dir) + SHA256SUMS.txt

set -euo pipefail

STAGE="output/handover_bundle_20260416"
CORRECTED="output/kmax_sweep/corrected/output/kmax_sweep"
SWEEP="output/kmax_sweep"

# --- Fail fast if source path is missing ---
if [ ! -d "$CORRECTED" ]; then
    echo "ERROR: corrected artifact path not found: $CORRECTED" >&2
    echo "Re-sync via: aws s3 sync s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260412T043407Z-dedup-linear/output/kmax_sweep/ $CORRECTED/ --profile torrin" >&2
    exit 1
fi

mkdir -p "$STAGE"

# --- Primary artifacts (K=15 production) ---
cp "$CORRECTED/firm_gmm_parameters_k15.parquet"    "$STAGE/"
cp "$CORRECTED/bc_matrix_all_k15_dedup_linear.npz" "$STAGE/"

# --- Convergence-floor reference (K=10) ---
cp "$CORRECTED/firm_gmm_parameters_k10.parquet"    "$STAGE/"
cp "$CORRECTED/bc_matrix_all_k10_dedup_linear.npz" "$STAGE/"

# --- Audit trail ---
cp "$SWEEP/deduplication_decisions.csv" "$STAGE/"
cp "$SWEEP/excluded_firms.csv"          "$STAGE/"

# --- Co-assignment audit (Step 4 output) — may not exist on first run ---
if [ -f "$SWEEP/coassignment_audit.parquet" ]; then
    cp "$SWEEP/coassignment_audit.parquet" "$STAGE/"
else
    echo "WARN: $SWEEP/coassignment_audit.parquet not found — run scripts/coassignment_audit.py first" >&2
fi

# --- Walkthrough notebook ---
if [ -f "notebooks/04_pipeline_output_overview.ipynb" ]; then
    cp "notebooks/04_pipeline_output_overview.ipynb" "$STAGE/"
else
    echo "WARN: notebooks/04_pipeline_output_overview.ipynb not found — bundle will ship without walkthrough" >&2
fi

# --- MANIFEST.txt (human-readable versioning anchor) ---
GIT_SHA=$(git rev-parse --short=12 HEAD 2>/dev/null || echo "unknown")
BUILT_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$STAGE/MANIFEST.txt" <<EOF
Bundle:  Week 2 Handover — Firm Portfolio Distance Matrix
Built:   $BUILT_AT (UTC)
Git SHA: $GIT_SHA
Source:  s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260412T043407Z-dedup-linear/
Project: firm-pair_merger_prediction (K_max=15 production, ADR-004 2026-04-14)

Contents:
  firm_gmm_parameters_k15.parquet    - Per-firm Bayesian GMM (K_max=15), PRIMARY
  bc_matrix_all_k15_dedup_linear.npz - Pairwise BC matrix (7485 x 7485), PRIMARY
  firm_gmm_parameters_k10.parquet    - Convergence-floor reference (K_max=10)
  bc_matrix_all_k10_dedup_linear.npz - Reference BC matrix at K=10
  deduplication_decisions.csv        - 464 firms removed (aliases, subs, predecessors)
  excluded_firms.csv                 - Firms removed for <5 patents
  coassignment_audit.parquet         - Top-100 BC pair shared-patent audit
  04_pipeline_output_overview.ipynb  - Teammate walkthrough notebook
  MANIFEST.txt                       - This file
  SHA256SUMS.txt                     - Integrity checksums

Verification (Linux/Mac/WSL):
  cd <bundle>
  sha256sum -c SHA256SUMS.txt

Verification (Windows PowerShell): see README "Quickstart for Teammates" section.
EOF

# --- Integrity checksums (remove stale SHA256SUMS first to ensure idempotent re-runs) ---
rm -f "$STAGE/SHA256SUMS.txt"
(cd "$STAGE" && sha256sum * > SHA256SUMS.txt)

N_FILES=$(ls "$STAGE" | wc -l)
TOTAL_MB=$(du -sm "$STAGE" | cut -f1)
echo "Bundle staged at $STAGE with $N_FILES files (${TOTAL_MB} MB)"
echo "MANIFEST: git SHA $GIT_SHA, built $BUILT_AT"
echo "Verify integrity: cd $STAGE && sha256sum -c SHA256SUMS.txt"
