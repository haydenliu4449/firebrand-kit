#!/usr/bin/env bash
# Shared setup for every numbered script: load .env, validate it, derive names.
# Not meant to be run on its own.

set -euo pipefail

# gcloud sometimes asks a question on stdin ("install the beta component?").
# Inside a script that looks exactly like a hang. Make it fail loudly instead.
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

if [[ ! -f "${ROOT_DIR}/.env" ]]; then
  die "No .env yet. Create it from the template and check PROJECT_ID, BUCKET_PATH, LOCATION:
    cp .env.example .env"
fi
# tr strips Windows line endings: a .env saved by Notepad would otherwise put
# an invisible \r on the end of every value ("us-central1\r" is not a region).
set -a
# shellcheck disable=SC1090
source <(tr -d '\r' < "${ROOT_DIR}/.env")
set +a

[[ -n "${PROJECT_ID:-}" ]]  || die "Set PROJECT_ID in .env"
[[ -n "${BUCKET_PATH:-}" ]] || die "Set BUCKET_PATH in .env"
[[ -n "${LOCATION:-}" ]]    || die "Set LOCATION in .env"
ARTIFACT_REPOSITORY="${ARTIFACT_REPOSITORY:-firebrand}"
IMAGE_NAME="${IMAGE_NAME:-firebrand}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
GPU_MACHINE="${GPU_MACHINE:-g2-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
CPU_MACHINE="${CPU_MACHINE:-n2-standard-4}"
PROVISIONING="${PROVISIONING:-STANDARD}"
CPU_PARALLEL="${CPU_PARALLEL:-4}"
GPU_PARALLEL="${GPU_PARALLEL:-1}"
MASK="${MASK:-left_front.json}"

# BUCKET_PATH may be "bucket" or "bucket/some/folder", with or without gs://.
# (The lab template's check rejected the folder form, although Batch accepts it.)
BUCKET_PATH="${BUCKET_PATH#gs://}"
BUCKET_PATH="${BUCKET_PATH%/}"
BUCKET="${BUCKET_PATH%%/*}"
if ! [[ "${BUCKET}" =~ ^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$ ]]; then
  die "Invalid bucket name '${BUCKET}' in BUCKET_PATH: 3-63 characters, lowercase letters, digits, '.', '_' or '-'."
fi
for v in ARTIFACT_REPOSITORY IMAGE_NAME; do
  [[ "${!v}" =~ ^[a-z0-9._-]+$ ]] || die "Invalid ${v}='${!v}': lowercase letters, digits, '.', '_' or '-'."
done
[[ "${IMAGE_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Invalid IMAGE_TAG='${IMAGE_TAG}'"
case "${PROVISIONING}" in STANDARD|SPOT) ;; *) die "PROVISIONING must be STANDARD or SPOT" ;; esac

GCS_ROOT="gs://${BUCKET_PATH}"
REGISTRY_HOST="${LOCATION}-docker.pkg.dev"
IMAGE_URI="${REGISTRY_HOST}/${PROJECT_ID}/${ARTIFACT_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

export PROJECT_ID BUCKET_PATH BUCKET LOCATION IMAGE_URI GPU_MACHINE GPU_TYPE \
       CPU_MACHINE PROVISIONING CPU_PARALLEL GPU_PARALLEL MASK GCS_ROOT

need_gcloud() {
  command -v gcloud >/dev/null 2>&1 || die "gcloud is not installed or not on PATH.
  Easiest: run these scripts in Cloud Shell (the >_ icon in the Cloud Console),
  where gcloud is preinstalled and already logged in."
}

# python3 on Linux/Cloud Shell; on Windows Git Bash it may be python or py.
pick_python() {
  local c
  for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  die "Python 3.8+ not found (needed to render job files)."
}

# Every gcloud call names the project explicitly, so whatever `gcloud config`
# happens to point at on this machine cannot redirect a job or an upload.
gc() { gcloud --project "${PROJECT_ID}" "$@"; }

VIDEO_RE='\.(mp4|avi|mov|mkv|m4v)$'

count_bucket_clips() {
  gc storage ls "${GCS_ROOT}/clips/" 2>/dev/null | grep -Eic "${VIDEO_RE}" || true
}
