ARG BASE_IMAGE=mcr.microsoft.com/devcontainers/python:1-3.11-bookworm
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGE=torch==2.1.0+cpu
ARG TORCHVISION_PACKAGE=torchvision==0.16.0+cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
    VIDEO_PLATFORM_BASE_DIR=/root/video_platform \
    PROJECT_ROOT=/root/video_platform \
    PYTHONPATH=/root/video_platform \
    TESSERACT_CMD=/usr/bin/tesseract

RUN rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn.list.save && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/video_platform

COPY root/video_platform/requirements-ai.txt /tmp/requirements-ai.txt

RUN python -m pip install setuptools wheel && \
    python -m pip install --index-url "${TORCH_INDEX_URL}" \
    "${TORCH_PACKAGE}" \
    "${TORCHVISION_PACKAGE}" && \
    grep -v '^openai-whisper==' /tmp/requirements-ai.txt > /tmp/requirements-ai-docker.txt && \
    python -m pip install --no-build-isolation -r /tmp/requirements-ai-docker.txt && \
    python -m pip install --no-deps --no-build-isolation openai-whisper==20231106

COPY root/video_platform /root/video_platform
COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh && \
    mkdir -p /root/video_platform/uploads \
             /root/video_platform/logs \
             /root/video_platform/video_data \
             /root/video_platform/static/uploads \
             /root/video_platform/static/video_data

EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["web"]
