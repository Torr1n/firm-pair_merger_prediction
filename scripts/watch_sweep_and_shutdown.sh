#!/usr/bin/env bash
set -euo pipefail

PID="$1"
RUN_ID="$2"

AWS_PROFILE="${AWS_PROFILE:-torrin}"
AWS_REGION="${AWS_REGION:-us-west-2}"
S3_BUCKET="${S3_BUCKET:-ubc-torrin}"
S3_PREFIX="${S3_PREFIX:-firm-pair-merger}"
WORKDIR="${WORKDIR:-/home/ubuntu/firm-pair_merger_prediction}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKDIR/output/kmax_sweep}"
LOG_PATH="${LOG_PATH:-$OUTPUT_ROOT/sweep.log}"
STATUS_DIR="${STATUS_DIR:-$OUTPUT_ROOT/status}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-57600}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RETRIES_ON_FAILURE="${RETRIES_ON_FAILURE:-1}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-1800}"

RUN_PREFIX="s3://${S3_BUCKET}/${S3_PREFIX}/week2/kmax_sweep/runs/${RUN_ID}"
WATCHDOG_STATUS_PATH="${STATUS_DIR}/watchdog_status.json"
SWEEP_STATUS_PATH="${STATUS_DIR}/sweep_status.json"

mkdir -p "$STATUS_DIR"

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
attempt=0

write_watchdog_status() {
  local status="$1"
  local ended_at="${2:-}"

  cat > "${WATCHDOG_STATUS_PATH}" <<EOF
{"run_id":"${RUN_ID}","status":"${status}","started_at":"${started_at}","ended_at":"${ended_at}","pid":${PID},"attempt":${attempt}}
EOF
}

sync_artifacts() {
  local sync_status="$1"
  local ended_at
  ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  write_watchdog_status "${sync_status}" "${ended_at}"

  for attempt_num in 1 2 3; do
    if AWS_PROFILE="${AWS_PROFILE}" AWS_REGION="${AWS_REGION}" \
      aws s3 sync "${OUTPUT_ROOT}/" "${RUN_PREFIX}/output/kmax_sweep/" --profile "${AWS_PROFILE}"; then
      break
    fi
    sleep 15
  done

  AWS_PROFILE="${AWS_PROFILE}" AWS_REGION="${AWS_REGION}" \
    aws s3 cp "${WATCHDOG_STATUS_PATH}" "${RUN_PREFIX}/status/watchdog_status.json" --profile "${AWS_PROFILE}" || true

  if [ -f "${SWEEP_STATUS_PATH}" ]; then
    AWS_PROFILE="${AWS_PROFILE}" AWS_REGION="${AWS_REGION}" \
      aws s3 cp "${SWEEP_STATUS_PATH}" "${RUN_PREFIX}/status/sweep_status.json" --profile "${AWS_PROFILE}" || true
  fi
}

classify_result() {
  if [ -f "${SWEEP_STATUS_PATH}" ]; then
    sweep_status="$(python3 - "${SWEEP_STATUS_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
    print(data.get("status", ""))
except Exception:
    print("")
PY
)"
    case "${sweep_status}" in
      success|completed_no_convergence)
        echo "${sweep_status}"
        return
        ;;
    esac
  fi

  if grep -q "VERDICT: CONVERGED" "${LOG_PATH}" 2>/dev/null; then
    echo "success"
    return
  fi

  if grep -q "VERDICT: NOT CONVERGED" "${LOG_PATH}" 2>/dev/null; then
    echo "completed_no_convergence"
    return
  fi

  echo "failed"
}

launch_retry() {
  cd "${WORKDIR}"
  nohup bash -lc \
    "cd '${WORKDIR}' && source venv/bin/activate && RUN_ID='${RUN_ID}' python -u scripts/run_kmax_sweep.py --local >> '${LOG_PATH}' 2>&1" \
    >/dev/null 2>&1 &
  PID=$!
  attempt=$(( attempt + 1 ))
  write_watchdog_status "running"
}

write_watchdog_status "running"

deadline=$(( $(date +%s) + MAX_RUNTIME_SECONDS ))
next_sync=$(( $(date +%s) + SYNC_INTERVAL_SECONDS ))

while kill -0 "${PID}" 2>/dev/null; do
  now="$(date +%s)"

  if [ "${now}" -ge "${deadline}" ]; then
    sync_artifacts "timed_out"
    rm -f /home/ubuntu/.aws/credentials /home/ubuntu/.aws/config
    sudo shutdown -h now
    exit 0
  fi

  if [ "${now}" -ge "${next_sync}" ]; then
    sync_artifacts "running"
    next_sync=$(( now + SYNC_INTERVAL_SECONDS ))
  fi

  sleep "${POLL_SECONDS}"
done

final_status="$(classify_result)"
if [ "${final_status}" = "failed" ] && [ "${attempt}" -lt "${RETRIES_ON_FAILURE}" ]; then
  sync_artifacts "retrying"
  launch_retry
  next_sync=$(( $(date +%s) + SYNC_INTERVAL_SECONDS ))

  while kill -0 "${PID}" 2>/dev/null; do
    now="$(date +%s)"

    if [ "${now}" -ge "${deadline}" ]; then
      sync_artifacts "timed_out"
      rm -f /home/ubuntu/.aws/credentials /home/ubuntu/.aws/config
      sudo shutdown -h now
      exit 0
    fi

    if [ "${now}" -ge "${next_sync}" ]; then
      sync_artifacts "running"
      next_sync=$(( now + SYNC_INTERVAL_SECONDS ))
    fi

    sleep "${POLL_SECONDS}"
  done

  final_status="$(classify_result)"
fi

sync_artifacts "${final_status}"
rm -f /home/ubuntu/.aws/credentials /home/ubuntu/.aws/config
sudo shutdown -h now
