#!/usr/bin/env bash
set -euo pipefail

PID="$1"
RUN_ID="$2"

AWS_PROFILE="${AWS_PROFILE:-torrin}"
AWS_REGION="${AWS_REGION:-us-west-2}"
S3_BUCKET="${S3_BUCKET:-ubc-torrin}"
S3_PREFIX="${S3_PREFIX:-firm-pair-merger}"
WORKDIR="${WORKDIR:-/home/ubuntu/firm-pair_merger_prediction}"
LOG_PATH="${LOG_PATH:-$WORKDIR/output/pipeline.log}"
STATUS_DIR="${STATUS_DIR:-$WORKDIR/output/status}"
RUN_PREFIX="s3://${S3_BUCKET}/${S3_PREFIX}/runs/${RUN_ID}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-28800}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RETRIES_ON_FAILURE="${RETRIES_ON_FAILURE:-1}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-1800}"

mkdir -p "$STATUS_DIR"

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "${STATUS_DIR}/run_started.json" <<EOF
{"run_id":"${RUN_ID}","status":"running","started_at":"${started_at}","pid":${PID}}
EOF

sync_artifacts() {
  local sync_status="$1"
  local ended_at
  ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  cat > "${STATUS_DIR}/run_status.json" <<EOF
{"run_id":"${RUN_ID}","status":"${sync_status}","started_at":"${started_at}","ended_at":"${ended_at}","pid":${PID}}
EOF

  for attempt in 1 2 3; do
    if AWS_PROFILE="${AWS_PROFILE}" AWS_REGION="${AWS_REGION}" aws s3 sync "${WORKDIR}/output/" "${RUN_PREFIX}/output/" --profile "${AWS_PROFILE}"; then
      break
    fi
    sleep 15
  done

  AWS_PROFILE="${AWS_PROFILE}" AWS_REGION="${AWS_REGION}" aws s3 cp "${LOG_PATH}" "${RUN_PREFIX}/logs/pipeline.log" --profile "${AWS_PROFILE}" || true
  AWS_PROFILE="${AWS_PROFILE}" AWS_REGION="${AWS_REGION}" aws s3 cp "${STATUS_DIR}/run_status.json" "${RUN_PREFIX}/status/run_status.json" --profile "${AWS_PROFILE}" || true
}

classify_result() {
  if grep -q "Pipeline complete" "${LOG_PATH}" 2>/dev/null; then
    echo "success"
    return
  fi

  if [ -f "${WORKDIR}/output/embeddings/patent_vectors_50d.parquet" ]; then
    echo "success"
    return
  fi

  echo "failed"
}

deadline=$(( $(date +%s) + MAX_RUNTIME_SECONDS ))
next_sync=$(( $(date +%s) + SYNC_INTERVAL_SECONDS ))
attempt=0

launch_retry() {
  cd "$WORKDIR"
  # Retry by appending to the existing log and reusing on-disk checkpoints.
  nohup bash -lc "cd '$WORKDIR' && source venv/bin/activate && python scripts/run_full_pipeline.py >> output/pipeline.log 2>&1" >/dev/null 2>&1 &
  PID=$!
  attempt=$(( attempt + 1 ))
  cat > "${STATUS_DIR}/run_started.json" <<EOF
{"run_id":"${RUN_ID}","status":"running","started_at":"${started_at}","pid":${PID},"attempt":${attempt}}
EOF
}

while kill -0 "${PID}" 2>/dev/null; do
  now="$(date +%s)"
  if [ "$(date +%s)" -ge "${deadline}" ]; then
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
if [ "${final_status}" != "success" ] && [ "${attempt}" -lt "${RETRIES_ON_FAILURE}" ]; then
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
