#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/home/ubuntu/firm-pair_merger_prediction}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKDIR/output/kmax_sweep}"
STATUS_DIR="${STATUS_DIR:-$OUTPUT_ROOT/status}"
LOG_PATH="${LOG_PATH:-$OUTPUT_ROOT/recompute.log}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
AWS_PROFILE="${AWS_PROFILE:-torrin}"
AWS_REGION="${AWS_REGION:-us-west-2}"
S3_BUCKET="${S3_BUCKET:-ubc-torrin}"
S3_PREFIX="${S3_PREFIX:-firm-pair-merger}"
WATCHDOG_MAX_RUNTIME_SECONDS="${WATCHDOG_MAX_RUNTIME_SECONDS:-14400}"
BACKSTOP_SECONDS="${BACKSTOP_SECONDS:-16200}"

mkdir -p "${STATUS_DIR}"

required_inputs=(
  "${WORKDIR}/output/kmax_sweep/deduplication_decisions.parquet"
  "${WORKDIR}/output/kmax_sweep/firm_gmm_parameters_k10.parquet"
  "${WORKDIR}/output/kmax_sweep/firm_gmm_parameters_k15.parquet"
  "${WORKDIR}/output/kmax_sweep/firm_gmm_parameters_k20.parquet"
  "${WORKDIR}/output/kmax_sweep/firm_gmm_parameters_k25.parquet"
  "${WORKDIR}/output/kmax_sweep/firm_gmm_parameters_k30.parquet"
)

for input_path in "${required_inputs[@]}"; do
  if [ ! -f "${input_path}" ]; then
    echo "Missing required input: ${input_path}" >&2
    exit 1
  fi
done

if pgrep -f "recompute_bc_corrected.py" >/dev/null 2>&1; then
  echo "A corrected BC recomputation process is already running." >&2
  exit 1
fi

{
  echo ""
  echo "=== Corrected BC recompute launch ${RUN_ID} at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
} >> "${LOG_PATH}"

cat > "${STATUS_DIR}/recompute_status.json" <<EOF
{"run_id":"${RUN_ID}","status":"running","timestamp":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF

cd "${WORKDIR}"
nohup bash -lc \
  "cd '${WORKDIR}' && source venv/bin/activate && RUN_ID='${RUN_ID}' python -u scripts/recompute_bc_corrected.py >> '${LOG_PATH}' 2>&1" \
  >/dev/null 2>&1 &
RECOMPUTE_PID=$!

nohup env \
  AWS_PROFILE="${AWS_PROFILE}" \
  AWS_REGION="${AWS_REGION}" \
  S3_BUCKET="${S3_BUCKET}" \
  S3_PREFIX="${S3_PREFIX}" \
  WORKDIR="${WORKDIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  LOG_PATH="${LOG_PATH}" \
  STATUS_DIR="${STATUS_DIR}" \
  MAX_RUNTIME_SECONDS="${WATCHDOG_MAX_RUNTIME_SECONDS}" \
  bash "${WORKDIR}/scripts/watch_recompute_and_shutdown.sh" "${RECOMPUTE_PID}" "${RUN_ID}" \
  >/dev/null 2>&1 &
WATCHDOG_PID=$!

nohup bash -lc "sleep ${BACKSTOP_SECONDS}; sudo shutdown -h now" >/dev/null 2>&1 &
BACKSTOP_PID=$!

cat > "${STATUS_DIR}/launch_metadata_recompute.json" <<EOF
{"run_id":"${RUN_ID}","recompute_pid":${RECOMPUTE_PID},"watchdog_pid":${WATCHDOG_PID},"backstop_pid":${BACKSTOP_PID},"launched_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF

echo "RUN_ID=${RUN_ID}"
echo "RECOMPUTE_PID=${RECOMPUTE_PID}"
echo "WATCHDOG_PID=${WATCHDOG_PID}"
echo "BACKSTOP_PID=${BACKSTOP_PID}"
echo "LOG_PATH=${LOG_PATH}"
