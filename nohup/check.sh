#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS_DIR="$ROOT_DIR/nohup/jobs"

usage() {
    echo "Usage:"
    echo "  bash nohup/check.sh                # list all jobs"
    echo "  bash nohup/check.sh <job_id|pid>   # inspect one job"
}

is_running() {
    local pid="$1"
    kill -0 "$pid" 2>/dev/null
}

print_job() {
    local meta_file="$1"
    local job_id pid log_file command started_at status

    job_id="$(sed -n 's/^job_id=//p' "$meta_file")"
    pid="$(sed -n 's/^pid=//p' "$meta_file")"
    started_at="$(sed -n 's/^started_at=//p' "$meta_file")"
    log_file="$(sed -n 's/^log_file=//p' "$meta_file")"
    command="$(sed -n 's/^command=//p' "$meta_file")"

    if [[ -n "$pid" ]] && is_running "$pid"; then
        status="RUNNING"
    else
        status="STOPPED"
    fi

    echo "job_id:    $job_id"
    echo "status:    $status"
    echo "pid:       ${pid:-N/A}"
    echo "started:   ${started_at:-N/A}"
    echo "log_file:  ${log_file:-N/A}"
    echo "command:   ${command:-N/A}"

    if [[ -n "$log_file" && -f "$log_file" ]]; then
        echo "---- last 20 log lines ----"
        tail -n 20 "$log_file"
    fi
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

if [[ ! -d "$JOBS_DIR" ]]; then
    echo "No jobs directory found: $JOBS_DIR"
    exit 0
fi

if [[ $# -gt 1 ]]; then
    usage
    exit 1
fi

if [[ $# -eq 0 ]]; then
    found=0
    for meta_file in "$JOBS_DIR"/*.meta; do
        [[ -e "$meta_file" ]] || continue
        found=1
        print_job "$meta_file"
        echo
    done

    if [[ "$found" -eq 0 ]]; then
        echo "No jobs found."
    fi
    exit 0
fi

META_FILE="$(find_meta_by_job_or_pid "$1")"
if [[ -z "$META_FILE" ]]; then
    echo "Job not found for: $1"
    exit 1
fi

print_job "$META_FILE"
