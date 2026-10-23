#!/usr/bin/env bash
# Push inputs up to the bucket: footage, masks, and your review decisions.
#
#   bash scripts/03-upload_inputs.sh              # clips/ + masks/ + reject lists
#   bash scripts/03-upload_inputs.sh /path/clips  # footage from somewhere else
#
# Uploading is free. rsync only sends files that are new or changed, so re-run
# it freely -- after a dropped connection, or after writing more reject.txt files.
#
# What goes where:
#   clips/*.mp4                   -> <bucket>/clips/
#   negclips/*.mp4                -> <bucket>/negclips/  (firebrand-free videos; detect
#                                    and tag skip them. A big one is quicker uploaded
#                                    in the Cloud Console straight into negclips/.)
#   masks/*.json                  -> <bucket>/masks/
#   work/clips/<clip>/reject.txt  -> <bucket>/work/clips/<clip>/reject.txt
#   work/tags/<clip>/reject.txt   -> <bucket>/work/tags/<clip>/reject.txt  (your review
#                                    of the model's tags; --real-positives skips those)
#   work/negatives.txt            -> <bucket>/work/negatives.txt  (firebrand-free frames)
#
# Nothing else under work/ is uploaded: the bucket's copy of work/ is the one
# the jobs write, and this script never overwrites it.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_gcloud

CLIPS_DIR="${1:-${ROOT_DIR}/clips}"

if [[ -d "${CLIPS_DIR}" ]]; then
  n=$(find "${CLIPS_DIR}" -maxdepth 1 -type f | grep -Eic "${VIDEO_RE}" || true)
  bold "Footage: ${n} video file(s) in ${CLIPS_DIR} -> ${GCS_ROOT}/clips/"
  if [[ "${n}" -gt 0 ]]; then
    # Only video files; README.md and friends stay behind.
    gc storage rsync "${CLIPS_DIR}" "${GCS_ROOT}/clips" --exclude='(?i)^(?!.*\.(mp4|avi|mov|mkv|m4v)$).*'
  fi
else
  warn "No clips folder at ${CLIPS_DIR} -- skipping footage."
fi

if [[ -d "${ROOT_DIR}/negclips" ]]; then
  n=$(find "${ROOT_DIR}/negclips" -maxdepth 1 -type f | grep -Eic "${VIDEO_RE}" || true)
  bold "Firebrand-free videos: ${n} file(s) in negclips/ -> ${GCS_ROOT}/negclips/"
  if [[ "${n}" -gt 0 ]]; then
    gc storage rsync "${ROOT_DIR}/negclips" "${GCS_ROOT}/negclips" --exclude='(?i)^(?!.*\.(mp4|avi|mov|mkv|m4v)$).*'
  fi
fi

if [[ -d "${ROOT_DIR}/masks" ]]; then
  bold "Masks: ${ROOT_DIR}/masks/*.json -> ${GCS_ROOT}/masks/"
  gc storage rsync "${ROOT_DIR}/masks" "${GCS_ROOT}/masks" --exclude='(?i)^(?!.*\.json$).*'
  if [[ ! -f "${ROOT_DIR}/masks/${MASK}" ]]; then
    warn "  MASK=${MASK} in .env but masks/${MASK} does not exist locally."
    warn "  Jobs will fall back to masking only the burned-in overlays."
  fi
fi

if [[ -f "${ROOT_DIR}/work/negatives.txt" ]]; then
  n=$(grep -cvE '^\s*(#|$)' "${ROOT_DIR}/work/negatives.txt" || true)
  bold "Firebrand-free footage: work/negatives.txt (${n} line(s))"
  gc storage cp "${ROOT_DIR}/work/negatives.txt" "${GCS_ROOT}/work/negatives.txt" --quiet
fi

shopt -s nullglob
for kind in clips tags; do
  rejects=("${ROOT_DIR}"/work/${kind}/*/reject.txt)
  (( ${#rejects[@]} )) || continue
  if [[ "${kind}" == clips ]]; then
    bold "Review decisions (classical tracks): ${#rejects[@]} reject.txt file(s)"
  else
    bold "Review decisions (the model's tags): ${#rejects[@]} reject.txt file(s)"
  fi
  for f in "${rejects[@]}"; do
    clip=$(basename "$(dirname "${f}")")
    ids=$(tr ',\r\n' '   ' < "${f}" | wc -w | tr -d ' ')
    gc storage cp "${f}" "${GCS_ROOT}/work/${kind}/${clip}/reject.txt" --quiet
    echo "  ${clip}: ${ids} track(s) rejected"
  done
done

echo
bold "In the bucket now:"
gc storage ls "${GCS_ROOT}/clips/" "${GCS_ROOT}/negclips/" "${GCS_ROOT}/masks/" 2>/dev/null | sed 's/^/  /' | head -40
