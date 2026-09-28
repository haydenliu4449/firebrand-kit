#!/usr/bin/env bash
# Pull results down from the bucket into ./work/ (same layout as the bucket).
#
#   bash scripts/05-gcs_sync.sh review     # contact sheets + CSVs from detect   (default)
#   bash scripts/05-gcs_sync.sh results    # model, training curve, eval report
#   bash scripts/05-gcs_sync.sh tags       # the model's tags for every clip
#   bash scripts/05-gcs_sync.sh logs       # full log of every task
#   bash scripts/05-gcs_sync.sh all
#   bash scripts/05-gcs_sync.sh tags --video   # include overlay.mp4 files
#
# Downloading costs ~$0.12/GB, so the bulky things that only the jobs need
# (tiles.npz, background frames, pickles, resume checkpoints, overlay videos)
# stay in the bucket unless you ask. Everything pulled here is small.
#
# Your reject.txt files are NEVER overwritten by this: the copy on your side is
# the one you are editing, and 03-upload_inputs.sh is what pushes it up.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_gcloud

WHAT="${1:-review}"
VIDEO=0
[[ "${2:-}" == "--video" || "${1:-}" == "--video" ]] && VIDEO=1
[[ "${WHAT}" == "--video" ]] && WHAT=review
LOCAL="${ROOT_DIR}/work"

skip='(^|.*/)bg/.*|.*\.pkl$|.*\.npz$|.*last\.pt(\.tmp)?$|.*\.tmp$|(^|.*/)reject\.txt$'
(( VIDEO )) || skip="${skip}|.*\.mp4$"

pull() {  # pull <bucket subpath> <label>
  local src="${GCS_ROOT}/work/$1" dst="${LOCAL}/$1"
  if ! gc storage ls "${src}/" >/dev/null 2>&1; then
    warn "  nothing at ${src}/ yet"
    return 0
  fi
  mkdir -p "${dst}"
  bold "$2: ${src}/ -> work/$1/"
  gc storage rsync --recursive "${src}" "${dst}" --exclude="${skip}"
}

case "${WHAT}" in
  review)  pull clips "detect output" ;;
  results) pull runs "model + report"
           gc storage cp "${GCS_ROOT}/work/config.json" "${LOCAL}/config.json" 2>/dev/null || true ;;
  tags)    pull tags "tags" ;;
  logs)    pull logs "logs" ;;
  all)     pull clips "detect output"; pull runs "model + report"
           pull tags "tags"; pull logs "logs" ;;
  *)       sed -n '2,12p' "$0"; exit 1 ;;
esac

echo
case "${WHAT}" in
  review|all)
    shopt -s nullglob
    sheets=("${LOCAL}"/clips/*/review/sheet_*.png)
    if (( ${#sheets[@]} )); then
      bold "${#sheets[@]} contact sheet(s) to review:  work/clips/<clip>/review/"
      echo "  For each clip, write the track IDs that are NOT firebrands into"
      echo "  work/clips/<clip>/reject.txt (e.g. 12, 47, 51), then:"
      echo "    bash scripts/03-upload_inputs.sh"
      echo "    bash scripts/04-submit_job.sh train"
    fi ;;
esac
case "${WHAT}" in
  results|all)
    [[ -f "${LOCAL}/runs/streak/report.json" ]] && {
      PY=$(pick_python)
      "${PY}" - "${LOCAL}/runs/streak/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
l, c = r["learned"], r["classical"]
print(f"eval report  (threshold {r['threshold']})")
print(f"  track recall      classical {c['trk_track_recall']:.3f}   learned {l['trk_track_recall']:.3f}")
print(f"  detection P / R   classical {c['det_precision']:.3f} / {c['det_recall']:.3f}"
      f"   learned {l['det_precision']:.3f} / {l['det_recall']:.3f}")
PY
    } ;;
esac
