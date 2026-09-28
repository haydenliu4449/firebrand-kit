#!/usr/bin/env bash
# One-time setup for running the firebrand pipeline on Cloud Batch.
#
#   bash scripts/00-setup.sh
#
# Safe to re-run: every step checks before it creates anything. Uses the
# values in .env. Needs Owner/Editor on the project for the API and IAM steps;
# if you only have narrower access, it says exactly what to ask for.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_gcloud

ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)
[[ -n "${ACCOUNT}" ]] || die "Not logged in. Run: gcloud auth login   (Cloud Shell is already logged in)"

bold "Account  ${ACCOUNT}"
bold "Project  ${PROJECT_ID}"
bold "Region   ${LOCATION}"
bold "Bucket   ${GCS_ROOT}"
bold "Image    ${IMAGE_URI}"
echo

gc projects describe "${PROJECT_ID}" >/dev/null 2>&1 || die "Cannot see project ${PROJECT_ID}.
  Check the ID in .env, or ask the project owner to add ${ACCOUNT}.
  Projects you can see:  gcloud projects list"
PROJECT_NUMBER=$(gc projects describe "${PROJECT_ID}" --format='value(projectNumber)')
# Batch VMs run as the Compute Engine default service account unless a job
# names another one. It needs to pull the image, read/write the bucket, report
# task status to Batch, and write logs.
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# ---------------------------------------------------------------- APIs
bold "1/5 Enabling APIs (a minute the first time)"
if ! out=$(gc services enable batch.googleapis.com compute.googleapis.com \
      artifactregistry.googleapis.com cloudbuild.googleapis.com \
      storage.googleapis.com logging.googleapis.com 2>&1); then
  echo "${out}" | sed 's/^/    /'
  if echo "${out}" | grep -qi billing; then
    die "Billing is not enabled on ${PROJECT_ID}. The project owner has to link a billing account."
  fi
  warn "    Could not enable APIs (permission?). If they are already on, continue:"
  warn "      gcloud services list --enabled --project=${PROJECT_ID}"
fi

# ---------------------------------------------------------------- bucket
bold "2/5 Bucket gs://${BUCKET}"
if gc storage buckets describe "gs://${BUCKET}" --format='value(location)' >/tmp/fb_loc 2>/dev/null; then
  BLOC=$(tr '[:upper:]' '[:lower:]' </tmp/fb_loc)
  echo "    exists, location ${BLOC}"
  if [[ "${BLOC}" != "${LOCATION}" ]]; then
    warn "    The bucket is in ${BLOC} but jobs run in ${LOCATION}. That works, but every"
    warn "    read crosses regions (slower, and a few cents per GB). Consider setting"
    warn "    LOCATION=${BLOC} in .env if ${BLOC} has the GPU you want."
  fi
else
  if ! out=$(gc storage buckets create "gs://${BUCKET}" --location="${LOCATION}" \
        --uniform-bucket-level-access --public-access-prevention 2>&1); then
    echo "${out}" | sed 's/^/    /'
    if echo "${out}" | grep -qiE '409|already (exists|own)'; then
      die "Bucket name '${BUCKET}' is taken -- names are global across all of Google Cloud.
  Pick another BUCKET_PATH in .env and re-run."
    fi
    die "Could not create the bucket. Ask for roles/storage.admin, or use an existing
  bucket (a folder inside one is fine):  BUCKET_PATH=existing-bucket/firebrands"
  fi
  echo "    created in ${LOCATION}"
fi
rm -f /tmp/fb_loc

# ---------------------------------------------------------------- registry
bold "3/5 Artifact Registry repository ${ARTIFACT_REPOSITORY}"
if gc artifacts repositories describe "${ARTIFACT_REPOSITORY}" --location="${LOCATION}" >/dev/null 2>&1; then
  echo "    exists"
else
  gc artifacts repositories create "${ARTIFACT_REPOSITORY}" --repository-format=docker \
    --location="${LOCATION}" --description="firebrand pipeline images" \
    || die "Could not create the repository (needs roles/artifactregistry.admin)."
  echo "    created"
fi

# ---------------------------------------------------------------- IAM
bold "4/5 Permissions for the job VMs (${SA})"
grant_project() {
  if gc projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${SA}" \
       --role="$1" --condition=None >/dev/null 2>&1; then
    echo "    ok  $1"
  else
    warn "    could not grant $1 (needs project IAM admin). Ask the owner to run:"
    warn "      gcloud projects add-iam-policy-binding ${PROJECT_ID} --member=serviceAccount:${SA} --role=$1"
  fi
}
grant_project roles/batch.agentReporter
grant_project roles/logging.logWriter
grant_project roles/artifactregistry.reader
# Cloud Build in newer projects also runs as this account and has to push.
if gc artifacts repositories add-iam-policy-binding "${ARTIFACT_REPOSITORY}" \
     --location="${LOCATION}" --member="serviceAccount:${SA}" \
     --role=roles/artifactregistry.writer >/dev/null 2>&1; then
  echo "    ok  roles/artifactregistry.writer on ${ARTIFACT_REPOSITORY}"
else
  warn "    could not grant artifactregistry.writer on the repository"
fi
if gc storage buckets add-iam-policy-binding "gs://${BUCKET}" \
     --member="serviceAccount:${SA}" --role=roles/storage.objectAdmin >/dev/null 2>&1; then
  echo "    ok  roles/storage.objectAdmin on gs://${BUCKET}"
else
  warn "    could not grant storage.objectAdmin on gs://${BUCKET}"
fi

# ---------------------------------------------------------------- GPU quota
bold "5/5 GPU quota in ${LOCATION}"
# New projects often have a GPU quota of zero. A job asking for a GPU you have
# no quota for does not fail straight away -- it waits in SCHEDULED for quota
# that never comes. Check here, and request an increase (IAM & Admin -> Quotas)
# before the day you need it.
QTYPE=$(echo "${GPU_TYPE}" | tr '[:lower:]-' '[:upper:]_' | sed 's/^NVIDIA_TESLA_/NVIDIA_/')
PY=$(pick_python)
show_quota() {  # stdin: JSON with a "quotas" list; args: metric names
  "${PY}" -c '
import json, sys
q = {x["metric"]: x for x in json.load(sys.stdin).get("quotas", [])}
for m in sys.argv[1:]:
    x = q.get(m)
    if x is None:
        print(f"    {m}: not listed (check IAM & Admin -> Quotas)")
        continue
    lim, use = int(x["limit"]), int(x["usage"])
    flag = "" if lim > 0 else "   <-- ZERO: request an increase"
    print(f"    {m}: limit {lim}, in use {use}{flag}")
' "$@"
}
if R=$(gc compute regions describe "${LOCATION}" --format=json 2>/dev/null); then
  echo "${R}" | show_quota "${QTYPE}_GPUS" "PREEMPTIBLE_${QTYPE}_GPUS"
else
  warn "    could not read regional quotas (compute API still enabling?)"
fi
if P=$(gc compute project-info describe --format=json 2>/dev/null); then
  echo "${P}" | show_quota GPUS_ALL_REGIONS
fi

cat <<EOF

$(bold "Done.") Next:

  bash scripts/01-build_and_push.sh      # build the image (~10 min, first time)
  bash scripts/03-upload_inputs.sh       # clips/ and masks/ -> ${GCS_ROOT}
  bash scripts/04-submit_job.sh smoke    # 5-minute GPU check before anything real

No budget alert is created here; cloud/setup.sh can make one, or set it in the
console (Billing -> Budgets & alerts). Batch jobs stop billing when they end,
so the thing to watch is a job that is still RUNNING:  bash scripts/06-jobs.sh running
EOF
