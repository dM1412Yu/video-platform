# -*- coding: utf-8 -*-
import os

# 基础路径配置
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
KEYFRAME_FOLDER = os.path.join(os.path.dirname(__file__), 'keyframes')
SPLIT_VIDEO_FOLDER = os.path.join(os.path.dirname(__file__), 'split_videos')
TRANSCRIPT_FOLDER = os.path.join(os.path.dirname(__file__), 'transcripts')
KNOWLEDGE_FOLDER = os.path.join(os.path.dirname(__file__), 'knowledge_points')
RELATION_NETWORK_FOLDER = os.path.join(os.path.dirname(__file__), 'relation_networks')

# OCR去重配置
SSIM_THRESHOLD = 0.9
OCR_CHANGE_THRESHOLD = 0.3

# 语音引导帧配置
SPEECH_GUIDED_FRAME_NUM = 10

# 语义去重配置
SEMANTIC_DUPLICATE_THRESHOLD = 0.85

# 视频切割配置
MIN_SEGMENT_DURATION = 30.0  # 最小30秒
MAX_SEGMENT_DURATION = 270.0  # 最大270秒

# 通义千问配置
TONGYI_API_KEY = "sk-24feda02d5524ed89a3ff3f5e0cee735"  # 替换为你的实际密钥
VLM_MODEL = "qwen-turbo"
VLM_TEMPERATURE = 0.1
VLM_MAX_TOKENS = 2000
