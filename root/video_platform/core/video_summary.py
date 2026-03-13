import os
import cv2
import json
import numpy as np
import torch
import dashscope
from dashscope import Generation
import whisper
import subprocess
import re
import datetime
import requests
import time
import pytesseract
import logging
import traceback
from pathlib import Path

# ====================== 日志（新增报错日志） ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ====================== Tesseract OCR ======================
pytesseract.pytesseract_cmd = '/usr/bin/tesseract'

# ====================== 全局配置（路径统一配置，避免硬编码） ======================
# 基础路径配置（可通过环境变量覆盖，适配不同部署环境）
# ====================== 全局配置（修正后，匹配你的实际路径） ======================
BASE_DIR = os.getenv("VIDEO_PLATFORM_BASE_DIR", "/root/video_platform")
# 修正：拆分视频根目录 = video_data（因为你的split_videos在video_data/{ID}下）
SPLIT_VIDEO_ROOT = os.path.join(BASE_DIR, "video_data")  
OUTPUT_ROOT = os.path.join(BASE_DIR, "video_data")  # 输出根目录不变

# 模型/环境配置
DEVICE_TORCH = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16
FRAME_SAMPLING_RATE = 4
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-24feda02d5524ed89a3ff3f5e0cee735")
dashscope.api_key = DASHSCOPE_API_KEY

# AutoDL 环境适配
requests.packages.urllib3.disable_warnings()
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ====================== 工具函数（新增路径校验/适配） ======================
def init_env(input_video_dir, output_dir):
    """初始化环境，增强路径校验和错误提示"""
    # 1. 检查输入目录是否存在
    input_video_dir = os.path.abspath(input_video_dir)
    if not os.path.exists(input_video_dir):
        # 自动查找备选路径（解决实际视频在video_data下的问题）
        alt_path = os.path.join(OUTPUT_ROOT, os.path.basename(input_video_dir), "split_videos")
        if os.path.exists(alt_path):
            logger.warning(f"⚠️ 原输入路径不存在，自动切换到备选路径：{alt_path}")
            input_video_dir = alt_path
        else:
            # 列出所有可用的视频目录，方便排查
            available_dirs = []
            if os.path.exists(SPLIT_VIDEO_ROOT):
                available_dirs = [d for d in os.listdir(SPLIT_VIDEO_ROOT) if os.path.isdir(os.path.join(SPLIT_VIDEO_ROOT, d))]
            if os.path.exists(OUTPUT_ROOT):
                for root, dirs, _ in os.walk(OUTPUT_ROOT):
                    if "split_videos" in dirs:
                        available_dirs.append(os.path.join(root, "split_videos"))
            
            raise FileNotFoundError(
                f"子视频文件夹不存在：{input_video_dir}\n"
                f"🔍 可用的视频目录：{available_dirs if available_dirs else '无'}\n"
                f"💡 建议：检查video_id是否正确，或确认视频已拆分到split_videos目录"
            )
    
    # 2. 检查ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        raise RuntimeError("ffmpeg未安装，请执行：apt install ffmpeg")
    
    # 3. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"✅ 环境初始化完成（输入：{input_video_dir} | 输出：{output_dir}）")
    return input_video_dir  # 返回实际使用的输入路径

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return round(frame_count / fps, 2) if fps > 0 and frame_count > 0 else 0.0

def extract_number_from_filename(filename):
    match = re.match(r'^(\d+)_', filename)
    return int(match.group(1)) if match else 0

# ====================== 【核心】模型改为 qwen-plus，函数名不变 ======================
def call_qwen35_flash(prompt, max_retries=3, timeout=120):
    for retry in range(max_retries):
        try:
            response = Generation.call(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                result_format="text",
                temperature=0.1,
                top_p=0.9,
                timeout=timeout
            )
            if response.status_code == 200:
                return response.output.text.strip()
            else:
                raise Exception(f"code={response.status_code}")
        except Exception as e:
            logger.error(f"❌ 第{retry+1}次调用失败：{str(e)[:80]}")
            if retry == max_retries - 1:
                raise
            time.sleep(2 ** retry)
    return ""

# ====================== 音频转文字 ======================
def extract_audio_to_text(video_path):
    temp_audio = f"{video_path}.tmp.mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", "-y", temp_audio],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        model = whisper.load_model("base", device=DEVICE_TORCH)
        res = model.transcribe(temp_audio, language="zh", verbose=False, fp16=(DEVICE_TORCH == "cuda"))
        return res["text"].strip()
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

# ====================== 视频OCR ======================
def extract_video_ocr_only(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video_description": "视频无法打开", "frame_count": 0, "feature_shape": (1, 1)}

    frames = []
    frame_idx = 0
    while len(frames) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SAMPLING_RATE == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (224, 224))
            frames.append(frame_rgb)
        frame_idx += 1
    cap.release()

    if len(frames) == 0:
        frames = [np.zeros((224,224,3), dtype=np.uint8)] * NUM_FRAMES
    elif len(frames) < NUM_FRAMES:
        frames += [frames[-1]] * (NUM_FRAMES - len(frames))

    def frame_to_desc(frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return pytesseract.image_to_string(gray, lang='chi_sim+eng', config='--psm 6 --oem 3').strip() or "无明显文字"
        except:
            return "OCR 识别失败"

    descs = [f"第{i}帧：{frame_to_desc(frames[i])}" for i in range(0, len(frames), 4)]
    return {
        "video_description": "；".join(descs),
        "frame_count": len(frames),
        "feature_shape": (1, 1)
    }

# ====================== 批量多模态提取 ======================
def batch_extract_multimodal_features(input_video_dir):
    video_list = sorted(
        [f for f in os.listdir(input_video_dir) if f.endswith((".mp4", ".avi", ".mov"))],
        key=extract_number_from_filename
    )
    if not video_list:
        raise ValueError(f"无有效视频：{input_video_dir}\n💡 检查：目录下是否有mp4/avi/mov文件，文件名是否符合命名规则")

    data = {}
    total = len(video_list)
    for i, name in enumerate(video_list, 1):
        path = os.path.join(input_video_dir, name)
        try:
            logger.info(f"🔧 处理 {i}/{total}：{name}")
            dur = get_video_duration(path)
            txt = extract_audio_to_text(path)
            ocr = extract_video_ocr_only(path)
            data[name] = {
                "video_path": path,
                "duration_seconds": dur,
                "audio_text": txt,
                "video_description": ocr["video_description"],
                "video_feature_shape": ocr["feature_shape"],
                "status": "success"
            }
        except Exception as e:
            logger.error(f"⚠️ 失败 {name}：{str(e)[:50]}")
            data[name] = {"video_path": path, "status": "failed", "error": str(e)[:100]}
    return data

# ====================== 结构化摘要生成（Prompt不变） ======================
def generate_structured_summary(multimodal_info):
    video_name = os.path.basename(multimodal_info["video_path"])
    try:
        audio_text = multimodal_info['audio_text'][:300]
        video_desc = multimodal_info['video_description']

        # 原版Prompt（一字未改）
        prompt = f"""
你是专业的教学视频结构化摘要专家，需基于以下信息生成符合要求的JSON格式摘要，无任何多余文字：

### 输入信息
1. 视频视觉内容描述：{video_desc}
2. 音频转文字辅助内容：{audio_text}

### 核心校正规则（优先级最高）
1. 先对比视觉描述和音频文字的核心词汇，识别视觉中的明显OCR错误（如生僻字+数字组合、无意义字符）；
2. 仅当满足以下条件时，才替换视觉错误文本：
   - 视觉中有明显错误文本（如“蟊敏68”“moumin68”等无意义组合）；
   - 音频中明确出现对应的正确核心词汇（如“函数”）；
   - 能80%确定错误文本是正确词汇的识别错误；
3. 若无法80%确定正确内容（如仅视觉有模糊文字、音频无对应内容），不做任何替换，保留原始信息；

### 输出要求
1. 严格输出JSON格式，字段如下（缺一不可）：
   - video_id: 视频唯一标识（使用视频文件名，去掉后缀，如"kmeans_step_2"；若文件名无意义，用音频核心词汇生成）
   - title: 视频核心标题（简洁；基于核心词汇）
   - key_concepts: 核心知识点列表（数组形式，如["积分", "定积分"]）
   - logic_flow: 逻辑流程（字符串，步骤序列，如"1. 定义积分；2. 举例计算；3. 应用到几何"）
   - details: 细节提取（包含实体、关系、视觉描述，字符串；视觉描述中的错误文本需按规则校正）
   - timestamps: 子视频内关键时刻（字符串，如"00:30 - 公式介绍；01:20 - 案例演示"）
   - emphasis: 教学重点/强调内容（字符串，无则填"无"）
2. JSON字段值需贴合视频实际内容，不添加外部知识；
3. 确保JSON格式合法，可直接解析；
4.输出语音转文字的辅助内容时按照视频内容校正同音错字。

### 示例输出
{{
  "video_id": "function_characteristics",
  "title": "函数的几种特性解析",
  "key_concepts": ["函数", "有界性", "单调性", "奇偶性"],
  "logic_flow": "1. 介绍函数的基本概念；2. 讲解函数的几种特性；3. 举例说明特性应用",
  "details": "实体：函数；关系：函数特性是函数的重要属性；视觉描述：第0帧：函数；第4帧：有界性；第8帧：单调性；第12帧：无明显文字，以画面讲解为主",
  "timestamps": "00:10 - 函数概念介绍；00:30 - 函数特性讲解；01:00 - 案例演示",
  "emphasis": "强调：函数特性是后续学习的基础，需重点掌握"
}}

### 错误校正示例（必看）
- 若视觉描述含“蟊敏68”，音频含“函数的几种特性”→ 校正为“函数”；
- 若视觉描述含“乱码/无意义字符”，音频无对应内容 → 保留原始视觉文本，不修改；
- 若视觉和音频内容一致 → 直接使用，无需校正。
"""
        
        text = call_qwen35_flash(prompt)
        if not text:
            raise Exception("返回为空")
        
        text = text.strip().strip("```json").strip("```")
        summary = json.loads(text)
        summary["video_path"] = multimodal_info["video_path"]
        summary["duration_seconds"] = multimodal_info["duration_seconds"]
        summary["status"] = "success"
        return summary

    except Exception as e:
        logger.error(f"⚠️ 摘要失败 {video_name}：{str(e)[:50]}")
        return {
            "video_id": video_name.split(".")[0],
            "title": "",
            "key_concepts": [],
            "logic_flow": "",
            "details": "",
            "timestamps": "",
            "emphasis": "",
            "video_path": multimodal_info["video_path"],
            "duration_seconds": multimodal_info["duration_seconds"],
            "status": "failed",
            "error": str(e)[:100]
        }

def batch_generate_summaries(multimodal_data):
    valid = sorted([k for k,v in multimodal_data.items() if v["status"]=="success"], key=extract_number_from_filename)
    if not valid:
        raise ValueError("无有效视频可生成摘要")
    
    summaries = {}
    total = len(valid)
    for i, name in enumerate(valid, 1):
        logger.info(f"📝 摘要 {i}/{total}：{name}")
        summaries[name] = generate_structured_summary(multimodal_data[name])
    return summaries

# ====================== 保存结果（路径逻辑优化） ======================
def save_all_results(all_summaries, multimodal_data, output_dir, video_id):
    final_output_path = os.path.join(output_dir, video_id, "subvideo_summaries_all.json")
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

    sorted_names = sorted(all_summaries.keys(), key=extract_number_from_filename)
    result = {
        "metadata": {
            "input_dir": multimodal_data[next(iter(multimodal_data))]["video_path"].rsplit("/",1)[0],
            "output_file": final_output_path,
            "total_videos": len(multimodal_data),
            "success_extract": len([v for v in multimodal_data.values() if v["status"]=="success"]),
            "success_summary": len([s for s in all_summaries.values() if s["status"]=="success"]),
            "generate_time": str(datetime.datetime.now()),
            "video_id": video_id
        },
        "subvideo_summaries": {n:all_summaries[n] for n in sorted_names},
        "multimodal_features": multimodal_data
    }

    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 已保存：{final_output_path}")

# ====================== 核心函数（路径逻辑优化） ======================
def run_video_summary(input_video_dir, output_root_dir, video_id):
    """
    核心函数：统一路径逻辑，增强容错性
    :param input_video_dir: 输入视频目录（支持相对/绝对路径）
    :param output_root_dir: 输出根目录
    :param video_id: 视频ID（用于输出路径命名）
    :return: 执行结果
    """
    try:
        # 初始化环境（自动适配路径）
        actual_input_dir = init_env(input_video_dir, os.path.join(output_root_dir, video_id))
        
        # 多模态特征提取
        mm_data = batch_extract_multimodal_features(actual_input_dir)
        
        # 生成摘要
        summaries = batch_generate_summaries(mm_data)
        
        # 保存结果
        save_all_results(summaries, mm_data, output_root_dir, video_id)
        
        logger.info("🎉 视频摘要生成全流程完成！")
        return {
            "status": "success",
            "input_dir": actual_input_dir,  # 返回实际使用的输入路径
            "output_file": os.path.join(output_root_dir, video_id, "subvideo_summaries_all.json"),
            "video_id": video_id
        }
    except Exception as e:
        logger.error(f"❌ 执行失败：{e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_video_summary(
        input_video_dir="/root/video_platform/video_data/65/split_videos",  # 修正为实际路径
        output_root_dir="/root/video_platform/video_data",
        video_id="65"
    )