#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS_DIR="$ROOT_DIR/nohup/jobs"

usage() {
    echo "Usage:"
    echo "  bash nohup/stop.sh <job_id|pid|all>"
}

is_running() {
    local pid="$1"
    kill -0 "$pid" 2>/dev/null
}

stop_pid() {
    local pid="$1"
    if ! is_running "$pid"; then
        echo "Process already stopped: $pid"
        return
    fi

    kill "$pid" 2>/dev/null || true
    sleep 1

    if is_running "$pid"; then
        kill -9 "$pid" 2>/dev/null || true
    fi

    if is_running "$pid"; then
        echo "Failed to stop process: $pid"
        return 1
    fi

    echo "Stopped process: $pid"
}

find_meta_by_job_or_pid() {
    local target="$1"

    if [[ -f "$JOBS_DIR/${target}.meta" ]]; then
        echo "$JOBS_DIR/${target}.meta"
        return
    fi

    for meta_file in "$JOBS_DIR"/*.meta; do
        [[ -e "$meta_file" ]] || continue
        local pid
        pid="$(sed -n 's/^pid=//p' "$meta_file")"
        if [[ "$pid" == "$target" ]]; then
            echo "$meta_file"
            return
        fi
    done

    echo ""
}

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

if [[ ! -d "$JOBS_DIR" ]]; then
    echo "No jobs directory found: $JOBS_DIR"
    exit 0
fi

TARGET="$1"

if [[ "$TARGET" == "all" ]]; then
    found=0
    for meta_file in "$JOBS_DIR"/*.meta; do
        [[ -e "$meta_file" ]] || continue
        found=1
        pid="$(sed -n 's/^pid=//p' "$meta_file")"
        job_id="$(sed -n 's/^job_id=//p' "$meta_file")"
        echo "Stopping job: $job_id (pid=$pid)"
        stop_pid "$pid" || true
    done

    if [[ "$found" -eq 0 ]]; then
        echo "No jobs found."
    fi
    exit 0
fi

META_FILE="$(find_meta_by_job_or_pid "$TARGET")"
if [[ -z "$META_FILE" ]]; then
    echo "Job not found for: $TARGET"
    exit 1
fi

PID="$(sed -n 's/^pid=//p' "$META_FILE")"
JOB_ID="$(sed -n 's/^job_id=//p' "$META_FILE")"

echo "Stopping job: $JOB_ID (pid=$PID)"
stop_pid "$PID"
