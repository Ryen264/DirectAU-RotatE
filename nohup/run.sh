#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS_DIR="$ROOT_DIR/nohup/jobs"
LOGS_DIR="$ROOT_DIR/nohup/logs"

mkdir -p "$JOBS_DIR" "$LOGS_DIR"

usage() {
    echo "Usage:"
    echo "  bash nohup/run.sh <run.sh args...>"
    echo "Examples:"
    echo "  bash nohup/run.sh train RotatE wn18rr 0 0 256 1024 500 6.0 0.5 0.00005 80000 8 -de"
    echo "  bash nohup/run.sh config_RotatE_wn18rr.yaml"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

sanitize_name() {
    echo "$1" | tr ' /' '__' | tr -cd '[:alnum:]_.-'
}

build_job_suffix() {
    if [[ "$1" == *.yaml || "$1" == *.yml ]]; then
        sanitize_name "${1##*/}"
        return
    fi

    local mode="${1:-unknown}"
    local model="${2:-unknown}"
    local dataset="${3:-unknown}"
    sanitize_name "${mode}_${model}_${dataset}"
}

TS="$(date +%Y%m%d_%H%M%S)"
SUFFIX="$(build_job_suffix "$@")"
JOB_ID="${TS}_${SUFFIX}"
PID_FILE="$JOBS_DIR/${JOB_ID}.pid"
META_FILE="$JOBS_DIR/${JOB_ID}.meta"
LOG_FILE="$LOGS_DIR/${JOB_ID}.out"

CMD=(bash run.sh "$@")

(
    cd "$ROOT_DIR"
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"
)

PID="$(cat "$PID_FILE")"

{
    echo "job_id=$JOB_ID"
    echo "pid=$PID"
    echo "started_at=$TS"
    echo "log_file=$LOG_FILE"
    echo -n "command="
    printf '%q ' "${CMD[@]}"
    echo
} > "$META_FILE"

echo "Started background job"
echo "job_id: $JOB_ID"
echo "pid:    $PID"
echo "log:    $LOG_FILE"
echo "meta:   $META_FILE"
echo "check:  bash nohup/check.sh $JOB_ID"
echo "stop:   bash nohup/stop.sh $JOB_ID"
