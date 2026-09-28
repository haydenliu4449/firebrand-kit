# One image for every step, CPU and GPU. Built and pushed by
# scripts/01-build_and_push.sh; rebuild after any change to firebrand/,
# run_all.py, batch_entry.py or tests/.
#
# Base: NVIDIA's minimal CUDA image. It carries no CUDA libraries of its own
# (torch brings those) -- what it provides is the PATH / LD_LIBRARY_PATH wiring
# to /usr/local/nvidia, which is where Batch mounts the host's GPU driver. The
# lab's gcp-example used the much larger cudnn-runtime variant of the same
# family; this is the same plumbing at a fraction of the size.
FROM nvidia/cuda:12.8.2-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ffmpeg: synth.reencode round-trips synthetic clips through H.264 so the model
# never learns "real embers have compression artefacts, synthetic ones don't".
# libglib2.0-0t64: some opencv-python-headless builds still link GLib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         python3 python3-venv ffmpeg libglib2.0-0t64 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 refuses system-wide pip installs (PEP 668), hence the venv.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Dependencies before code: this layer (~5 GB, mostly torch + CUDA) is reused
# by a local Docker build when only the code changed.
COPY requirements-batch.txt /tmp/requirements-batch.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements-batch.txt

WORKDIR /app
COPY firebrand/ /app/firebrand/
COPY tests/ /app/tests/
COPY run_all.py run_pipeline.py batch_entry.py /app/
RUN chmod -R a+rX /app \
    && python -c "import firebrand, cv2, torch, scipy, pandas; print('image ok: torch', torch.__version__, 'cuda', torch.version.cuda, 'opencv', cv2.__version__)"

# For local `docker run IMAGE run --stage detect`. Batch jobs do NOT rely on
# this: the job JSON sets `entrypoint` and the full command explicitly (see
# scripts/render_job.py), because Batch ran args-only `commands` as the program.
ENTRYPOINT ["/opt/venv/bin/python", "/app/batch_entry.py"]
CMD ["smoke"]
