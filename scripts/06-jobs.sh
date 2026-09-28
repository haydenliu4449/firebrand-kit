#!/usr/bin/env bash
# Watch, read and stop Batch jobs.
#
#   bash scripts/06-jobs.sh                   # recent jobs (same as: list)
#   bash scripts/06-jobs.sh running           # anything still billing?
#   bash scripts/06-jobs.sh status JOB        # state + per-task progress
#   bash scripts/06-jobs.sh logs   JOB        # the pipeline's output (Cloud Logging)
#   bash scripts/06-jobs.sh log    JOB [N]    # saved log of task N from the bucket
#   bash scripts/06-jobs.sh cancel JOB        # stop it; its VMs are deleted
#   bash scripts/06-jobs.sh delete JOB        # remove a finished job from the list
#
# Job states: QUEUED -> SCHEDULED (finding a VM) -> RUNNING -> SUCCEEDED/FAILED.
# A job stuck in SCHEDULED for a long time usually means no GPU quota, or no
# free GPUs in the region right now (common with SPOT); `status` shows why.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_gcloud

CMD="${1:-list}"
JOB="${2:-}"
need_job() { [[ -n "${JOB}" ]] || die "Usage: $0 ${CMD} JOB_NAME   (names: $0 list)"; }

case "${CMD}" in
  list)
    gc batch jobs list --location "${LOCATION}" --sort-by=~createTime --limit=15 \
      --format="table(name.basename():label=JOB, status.state:label=STATE, createTime.date('%Y-%m-%d %H:%M'):label=CREATED, status.runDuration:label=RAN)"
    ;;
  running)
    out=$(gc batch jobs list --location "${LOCATION}" \
            --filter="status.state:(QUEUED SCHEDULED RUNNING DELETION_IN_PROGRESS)" \
            --format="table(name.basename():label=JOB, status.state:label=STATE, createTime.date('%Y-%m-%d %H:%M'):label=CREATED)")
    if [[ -z "${out}" ]]; then
      bold "No Batch jobs queued or running in ${LOCATION}: nothing is billing by the hour."
    else
      echo "${out}"
      echo
      echo "Stop one with:  bash scripts/06-jobs.sh cancel JOB"
    fi
    ;;
  status)
    need_job
    gc batch jobs describe "${JOB}" --location "${LOCATION}" \
      --format="yaml(status.state, status.runDuration, status.taskGroups, status.statusEvents)"
    echo
    gc batch tasks list --job "${JOB}" --location "${LOCATION}" \
      --format="table(name.basename():label=TASK, status.state:label=STATE)" 2>/dev/null || true
    ;;
  logs)
    need_job
    UID_=$(gc batch jobs describe "${JOB}" --location "${LOCATION}" --format='value(uid)')
    [[ -n "${UID_}" ]] || die "No job ${JOB} in ${LOCATION}."
    # Logs arrive with a delay of a few seconds to a minute.
    gc logging read "labels.job_uid=\"${UID_}\" AND logName=\"projects/${PROJECT_ID}/logs/batch_task_logs\"" \
      --order=asc --freshness=30d --limit="${LIMIT:-2000}" \
      --format='value(textPayload)'
    ;;
  log)
    need_job
    gc storage cat "${GCS_ROOT}/work/logs/${JOB}/task${3:-0}.log" \
      || die "No saved log yet -- it is written when the task ends. While it runs use:  $0 logs ${JOB}"
    ;;
  cancel)
    need_job
    gc batch jobs cancel "${JOB}" --location "${LOCATION}" --quiet
    bold "Cancel requested. Completed stages stay in the bucket; re-submitting the"
    bold "same step skips them and resumes."
    ;;
  delete)
    need_job
    gc batch jobs delete "${JOB}" --location "${LOCATION}" --quiet
    ;;
  *)
    sed -n '2,15p' "$0"; exit 1
    ;;
esac
