#!/usr/bin/env bash
# Build the container image and push it to Artifact Registry.
#
#   bash scripts/01-build_and_push.sh
#
# Rebuild after changing anything in firebrand/, run_all.py, batch_entry.py or
# tests/. Changing run_all.py ARGUMENTS (epochs, max-frames...) needs no rebuild
# -- those go on the job:  bash scripts/04-submit_job.sh train -- --epochs 40
#
# BUILD_METHOD=cloudbuild (default)  builds on Google's machines. No Docker on
#                                    your side, works from Cloud Shell or Windows.
#                                    ~10 min: it reinstalls torch every time.
# BUILD_METHOD=docker                builds with local Docker (Cloud Shell has it)
#                                    and caches the torch layer, so a code-only
#                                    rebuild takes about a minute after the first.
#
# Differences from the lab's gcp-example 01 script: amd64 only (Batch VMs are
# x86; building arm64 too doubled the time for an image nothing would run), and
# a longer Cloud Build timeout, because the default 10 minutes is shorter than
# a torch install.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_gcloud
BUILD_METHOD="${BUILD_METHOD:-cloudbuild}"

if ! gc artifacts repositories describe "${ARTIFACT_REPOSITORY}" --location="${LOCATION}" >/dev/null 2>&1; then
  bold "Creating Artifact Registry repository ${ARTIFACT_REPOSITORY} in ${LOCATION}"
  gc artifacts repositories create "${ARTIFACT_REPOSITORY}" --repository-format=docker \
    --location="${LOCATION}" --description="firebrand pipeline images"
fi

bold "Building ${IMAGE_URI}  (method: ${BUILD_METHOD})"
cd "${ROOT_DIR}"

case "${BUILD_METHOD}" in
  docker)
    command -v docker >/dev/null 2>&1 || die "docker not found. Use BUILD_METHOD=cloudbuild (the default)."
    gcloud auth configure-docker "${REGISTRY_HOST}" --quiet
    docker build --platform linux/amd64 -t "${IMAGE_URI}" "${ROOT_DIR}"
    docker push "${IMAGE_URI}"
    ;;
  cloudbuild)
    gc builds submit "${ROOT_DIR}" --tag "${IMAGE_URI}" --timeout=45m
    ;;
  *)
    die "BUILD_METHOD must be cloudbuild or docker"
    ;;
esac

DIGEST=$(gc artifacts docker images describe "${IMAGE_URI}" \
           --format='value(image_summary.digest)' 2>/dev/null || true)
bold "Pushed ${IMAGE_URI}"
[[ -n "${DIGEST}" ]] && echo "  digest ${DIGEST}  (jobs started from now on use this build)"
echo
echo "Next, if you have not yet:  bash scripts/04-submit_job.sh smoke"
