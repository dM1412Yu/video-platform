# -*- coding: utf-8 -*-
import os
from pathlib import Path

# ========== 基础路径配置 ==========
BASE_DIR = Path(__file__).parent.absolute()
UPLOAD_DIR = BASE_DIR / "uploads"  # 视频上传目录
SECRET_KEY = "video_platform_2026_secure_1234567890"  # Flask加密密钥
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 最大上传500MB

# ========== 设备配置 ==========
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# ========== 视频处理配置 ==========
GLOBAL_KEYFRAME_NUM = 20  # 全局关键帧数量
SUB_VIDEO_MAX_KEYFRAME = 5  # 子视频最大关键帧数量
SSIM_THRESHOLD = 0.8  # 关键帧去重阈值
MIN_SUB_VIDEO_DURATION = 5  # 子视频最短时长（秒）
MAX_SUB_VIDEO_DURATION = 30  # 子视频最长时长（秒）

# ========== 通义千问API配置 ==========
# ！！！替换成你自己的API Key！！！
DASHSCOPE_API_KEY = "sk-24feda02d5524ed89a3ff3f5e0cee735"