#!/usr/bin/env bash
# Submit one step of the pipeline as a Cloud Batch job.
#
#   bash scripts/04-submit_job.sh smoke                   # do this first
#   bash scripts/04-submit_job.sh detect -- --max-frames 300 --overlay
#   bash scripts/04-submit_job.sh train  -- --epochs 30
#   bash scripts/04-submit_job.sh tag
#
# Options before `--`:  --tasks N   --hours H   --spot   --name NAME
# Anything after `--` goes to run_all.py unchanged (--force, --max-frames,
# --synth-frames, --epochs, --threshold, --overlay, --no-mask ...).
#
# Steps (one run_all.py stage group each):
#   smoke    GPU  image + GPU + training + resume check           ~5 min
#   detect   CPU  classical detector, one task per clip            then REVIEW
#   tiles    CPU  optional: build training tiles without a GPU
#   train    GPU  tiles -> train -> eval (report.json)
#   tag      GPU  trained model tags every clip, one task per clip
#
# A VM exists only while the job runs. The job name is unique per submission
# (Batch refuses to reuse one), and the rendered JSON is kept in jobs/ as a
# record of exactly what ran.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_gcloud
PY=$(pick_python)

STEP="${1:-}"
[[ -n "${STEP}" && "${STEP}" != -* ]] || { sed -n '2,24p' "$0"; exit 1; }
shift

# Split our options from run_all.py's (everything after `--`).
NAME=""; TASKS=""; PASS=(); EXTRA=()
while (( $# )); do
  case "$1" in
    --)      shift; EXTRA=("$@"); break ;;
    --name)  NAME="${2:?--name needs a value}"; shift 2 ;;
    --tasks) TASKS="${2:?--tasks needs a value}"; shift 2 ;;
    *)       PASS+=("$1"); shift ;;
  esac
done
NAME="${NAME:-fb-${STEP}-$(date +%Y%m%d-%H%M%S)}"

# ---- cheap checks before paying for a machine --------------------------------
if ! gc artifacts docker images describe "${IMAGE_URI}" >/dev/null 2>&1; then
  die "Image ${IMAGE_URI} not found. Build it first:  bash scripts/01-build_and_push.sh"
fi

case "${STEP}" in
  detect|tag)
    N=$(count_bucket_clips)
    [[ "${N}" -gt 0 ]] || die "No clips in ${GCS_ROOT}/clips/. Upload them:  bash scripts/03-upload_inputs.sh"
    # One task per clip: Batch runs up to CPU_PARALLEL / GPU_PARALLEL of them
    # at once and feeds the rest to the same machines as they free up.
    TASKS="${TASKS:-${N}}"
    echo "${N} clip(s) in the bucket -> ${TASKS} task(s)"
    ;;
esac
case "${STEP}" in
  tiles|train)
    gc storage ls "${GCS_ROOT}/work/clips/*/done.json" >/dev/null 2>&1 \
      || die "No finished detect output in ${GCS_ROOT}/work/clips/. Run:  bash scripts/04-submit_job.sh detect"
    if ! gc storage ls "${GCS_ROOT}/work/clips/*/reject.txt" >/dev/null 2>&1; then
      warn "No reject.txt in the bucket: the generator will be fitted to UNREVIEWED tracks."
      warn "That is fine for a first pass. To review: 05-gcs_sync.sh review, write"
      warn "work/clips/<clip>/reject.txt, then 03-upload_inputs.sh."
    fi
    ;;
  tag)
    gc storage ls "${GCS_ROOT}/work/runs/streak/best.pt" >/dev/null 2>&1 \
      || die "No trained model at ${GCS_ROOT}/work/runs/streak/best.pt. Run:  bash scripts/04-submit_job.sh train"
    ;;
esac

# ---- render + submit ----------------------------------------------------------
[[ -n "${TASKS}" ]] && PASS+=(--tasks "${TASKS}")
CONFIG=$("${PY}" "${SCRIPT_DIR}/render_job.py" "${STEP}" --name "${NAME}" \
           ${PASS[@]+"${PASS[@]}"} -- ${EXTRA[@]+"${EXTRA[@]}"} 2>&1) || die "${CONFIG}"
CONFIG=$(echo "${CONFIG}" | tail -1)

bold "Submitting ${NAME} in ${LOCATION}"
echo "  config ${CONFIG#"${ROOT_DIR}/"}"
gc batch jobs submit "${NAME}" --location "${LOCATION}" --config "${CONFIG}" \
  --format='value(name,status.state)'

cat <<EOF

$(bold "Submitted.") It queues, starts a VM (a few minutes: boot + image pull), runs,
and the VM is deleted when it finishes. Follow it with:

  bash scripts/06-jobs.sh status ${NAME}
  bash scripts/06-jobs.sh logs   ${NAME}      # live output
  bash scripts/06-jobs.sh cancel ${NAME}      # stop it (and the billing)

Results land in ${GCS_ROOT}/work/ as they are produced.
EOF
