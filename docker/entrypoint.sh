#!/usr/bin/env bash
set -euo pipefail

cd /root/video_platform

mkdir -p \
  /root/video_platform/uploads \
  /root/video_platform/logs \
  /root/video_platform/video_data \
  /root/video_platform/static/uploads \
  /root/video_platform/static/video_data

MODE="${1:-web}"
shift || true

case "$MODE" in
  web)
    exec python app.py "$@"
    ;;
  ai-smoke)
    exec python scripts/ai_smoke_test.py "$@"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
