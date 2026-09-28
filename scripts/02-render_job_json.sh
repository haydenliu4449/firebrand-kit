#!/usr/bin/env bash
# Write the Batch job file for one step, without submitting it. 04 does this
# for you; run it on its own to see exactly what a job will do.
#
#   bash scripts/02-render_job_json.sh detect --tasks 10 -- --max-frames 300
#   bash scripts/02-render_job_json.sh --help          # list the steps
#
# Output: jobs/<job-name>.json (filled in from templates/job.<gpu|cpu>.template.json)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PY=$(pick_python)
"${PY}" "${SCRIPT_DIR}/render_job.py" "$@"
