#!/bin/bash
# 视频学习助手一键脚本（AutoDL云服务器）
if [ "" = "install" ]; then
  echo "🔧 安装依赖..."
  pip install -r requirements.txt
  apt update && apt install ffmpeg -y
  echo "✅ 依赖安装完成！"
elif [ "" = "run" ]; then
  echo "🚀 启动服务（端口6006，GPU自动检测）..."
  lsof -i:6006 | grep -v PID | awk '{print }' | xargs kill -9 2>/dev/null
  python app.py
elif [ "" = "stop" ]; then
  echo "🛑 停止服务..."
  lsof -i:6006 | grep -v PID | awk '{print }' | xargs kill -9 2>/dev/null
  echo "✅ 服务已停止！"
else
  echo "📚 使用说明："
  echo "  install - 安装依赖"
  echo "  run    - 启动服务（端口6006）"
  echo "  stop   - 停止服务"
fi
