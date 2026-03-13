# -*- coding: utf-8 -*-
import os
import json
import subprocess
import uuid
import cv2
import numpy as np
import whisper
import pytesseract
from PIL import Image
import torch
import clip
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import dashscope
from dashscope import Generation
import traceback
import datetime

# 导入配置
from config import (
    DEVICE, GLOBAL_KEYFRAME_NUM, SUB_VIDEO_MAX_KEYFRAME,
    SSIM_THRESHOLD, MIN_SUB_VIDEO_DURATION, MAX_SUB_VIDEO_DURATION,
    DASHSCOPE_API_KEY
)

# 日志工具
def log_error(step, error_msg):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"video_processor_error_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"【时间】{datetime.datetime.now()}\n")
        f.write(f"【步骤】{step}\n")
        f.write(f"【错误】{error_msg}\n")
        f.write(f"【详细堆栈】\n{traceback.format_exc()}\n")
    print(f"❌ [{step}] 错误已保存到：{log_file}")

def log_info(step, info_msg):
    print(f"✅ [{datetime.datetime.now().strftime('%H:%M:%S')}] {step}：{info_msg}")

# 配置通义千问
log_info("初始化通义千问", "开始配置API Key")
try:
    dashscope.api_key = DASHSCOPE_API_KEY
    log_info("初始化通义千问", f"API Key配置完成（前8位：{DASHSCOPE_API_KEY[:8]}...）")
except Exception as e:
    error_msg = f"通义千问API Key配置失败：{str(e)}"
    log_error("初始化通义千问", error_msg)
    raise Exception(error_msg)

# 模型初始化
def init_models():
    asr_model = None
    clip_model = None
    clip_preprocess = None
    
    log_info("模型初始化", "开始加载Whisper ASR模型")
    try:
        log_info("模型初始化", f"尝试加载base模型（设备：{DEVICE}）")
        asr_model = whisper.load_model("base", device=DEVICE)
        log_info("模型初始化", "Whisper base模型加载成功")
    except Exception as e:
        log_error("模型初始化-ASR", f"base模型加载失败：{str(e)}")
        try:
            log_info("模型初始化", "降级加载tiny模型")
            asr_model = whisper.load_model("tiny", device=DEVICE)
            log_info("模型初始化", "Whisper tiny模型加载成功")
        except Exception as e2:
            error_msg = f"tiny模型也加载失败：{str(e2)}"
            log_error("模型初始化-ASR", error_msg)
            raise Exception(error_msg)
    
    log_info("模型初始化", "开始加载CLIP模型")
    try:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
        log_info("模型初始化", "CLIP ViT-B/32模型加载成功")
    except Exception as e:
        log_error("模型初始化-CLIP", f"CLIP模型加载失败：{str(e)}")
        log_info("模型初始化", "CLIP模型加载失败，后续功能将跳过CLIP相关逻辑")
        clip_model, clip_preprocess = None, None
    
    return asr_model, clip_model, clip_preprocess

ASR_MODEL, CLIP_MODEL, CLIP_PREPROCESS = init_models()
log_info("全局初始化", "所有模型加载完成")

# 提取音频（修复空音频问题）
def extract_audio(video_path, video_data_dir):
    audio_path = video_data_dir / "audio.wav"
    log_info("音频提取", f"开始提取音频：{video_path} → {audio_path}")
    
    # 先检查视频文件
    if not video_path.exists() or video_path.stat().st_size < 1024:
        log_error("音频提取", f"视频文件无效：{video_path}")
        return None
    
    # 使用ffmpeg提取音频，强制采样率
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log_error("音频提取", f"ffmpeg执行失败：{result.stderr}")
            return None
        
        # 检查音频文件是否有效
        if audio_path.exists() and audio_path.stat().st_size > 1024:
            log_info("音频提取", f"音频提取成功：{audio_path}（大小：{audio_path.stat().st_size/1024:.2f}KB）")
            return audio_path
        else:
            log_error("音频提取", "提取的音频文件为空或无效")
            return None
    except Exception as e:
        log_error("音频提取", f"音频提取异常：{str(e)}")
        return None

# ASR转写（修复空tensor问题）
def asr_transcribe(audio_path, video_data_dir):
    log_info("ASR转写", f"开始处理音频：{audio_path}")
    
    # 检查音频文件
    if not audio_path or not audio_path.exists() or audio_path.stat().st_size < 1024:
        log_error("ASR转写", "音频文件无效，跳过ASR")
        return [], ""
    
    try:
        result = ASR_MODEL.transcribe(
            str(audio_path), 
            language="zh", 
            fp16=DEVICE=="cuda",
            verbose=True
        )
        log_info("ASR转写", "Whisper转写完成")
    except Exception as e:
        error_msg = f"ASR转写失败：{str(e)}"
        log_error("ASR转写", error_msg)
        result = {"segments": []}
    
    asr_raw = []
    for seg in result.get("segments", []):
        asr_raw.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        })
    log_info("ASR转写", f"提取到{len(asr_raw)}段语音文本")
    
    # 保存结果
    try:
        asr_raw_path = video_data_dir / "asr_raw.json"
        with open(asr_raw_path, "w", encoding="utf-8") as f:
            json.dump(asr_raw, f, ensure_ascii=False, indent=2)
        
        asr_text = "\n".join([f"[{s['start']}-{s['end']}s] {s['text']}" for s in asr_raw])
        asr_text_path = video_data_dir / "asr_text.txt"
        with open(asr_text_path, "w", encoding="utf-8") as f:
            f.write(asr_text)
        log_info("ASR转写", f"ASR结果已保存")
    except Exception as e:
        log_error("ASR转写-保存", f"保存失败：{str(e)}")
    
    return asr_raw, asr_text

# 关键帧+OCR（修复中文语言包问题）
def extract_keyframes_ocr(video_path, video_data_dir):
    log_info("关键帧提取", f"开始处理视频：{video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        error_msg = f"视频打开失败：{video_path}"
        log_error("关键帧提取", error_msg)
        raise Exception(error_msg)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    log_info("关键帧提取", f"视频信息：FPS={fps}，时长={duration:.2f}秒")
    
    # 创建关键帧目录
    kf_dir = video_data_dir / "global_keyframes"
    kf_dir.mkdir(exist_ok=True)
    
    # 采样候选帧
    step = max(1, total_frames // (GLOBAL_KEYFRAME_NUM * 2))
    candidate_frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            candidate_frames.append((i/fps, frame))
    
    # SSIM去重
    dedup_frames = []
    prev_frame = None
    for ts, frame in candidate_frames:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))
            if prev_frame is None or ssim(prev_frame, gray, data_range=255) < SSIM_THRESHOLD:
                dedup_frames.append((ts, frame))
                prev_frame = gray
            if len(dedup_frames) >= GLOBAL_KEYFRAME_NUM:
                break
        except Exception as e:
            log_error("关键帧去重", f"失败（{ts}s）：{str(e)}")
            continue
    
    # OCR识别（指定中文语言包）
    ocr_results = []
    for idx, (ts, frame) in enumerate(dedup_frames[:GLOBAL_KEYFRAME_NUM]):
        try:
            kf_filename = f"frame_{ts:.2f}s.jpg"
            kf_path = kf_dir / kf_filename
            cv2.imwrite(str(kf_path), frame)
            
            # 强制使用中文语言包
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ocr_text = pytesseract.image_to_string(pil_frame, lang='chi_sim').strip()
            
            ocr_results.append({
                "ts": round(ts,2),
                "frame_path": str(kf_path),
                "ocr_text": ocr_text
            })
            log_info("OCR识别", f"第{idx+1}帧：{ocr_text[:50]}")
        except Exception as e:
            log_error("OCR识别", f"第{idx+1}帧失败：{str(e)}")
            ocr_results.append({
                "ts": round(ts,2),
                "frame_path": "",
                "ocr_text": ""
            })
    
    cap.release()
    
    # 保存OCR结果
    try:
        ocr_path = video_data_dir / "global_ocr.json"
        with open(ocr_path, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_error("OCR保存", f"失败：{str(e)}")
    
    return ocr_results

# 切割视频片段+提取知识点
def split_video_segments(video_path, asr_raw, video_data_dir):
    log_info("视频切割", "开始提取知识点和切割片段")
    
    # 从ASR生成知识点
    segments = []
    if asr_raw:
        # 按时间分块（每30秒一个片段）
        current_segment = {"start": 0, "end": 0, "text": "", "index": 0}
        for idx, seg in enumerate(asr_raw):
            if current_segment["end"] == 0:
                current_segment["start"] = seg["start"]
            
            current_segment["end"] = seg["end"]
            current_segment["text"] += seg["text"] + " "
            
            # 达到最大时长或最后一段
            if (current_segment["end"] - current_segment["start"]) >= MAX_SUB_VIDEO_DURATION or idx == len(asr_raw)-1:
                if (current_segment["end"] - current_segment["start"]) >= MIN_SUB_VIDEO_DURATION:
                    current_segment["index"] = len(segments) + 1
                    current_segment["title"] = f"知识点{current_segment['index']}：{current_segment['text'][:20]}..."
                    segments.append(current_segment.copy())
                current_segment = {"start": 0, "end": 0, "text": "", "index": 0}
    else:
        # 无ASR时按时长均分
        cap = cv2.VideoCapture(str(video_path))
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25)
        cap.release()
        
        num_segments = max(1, int(duration / MAX_SUB_VIDEO_DURATION))
        for i in range(num_segments):
            start = i * MAX_SUB_VIDEO_DURATION
            end = min((i+1)*MAX_SUB_VIDEO_DURATION, duration)
            segments.append({
                "index": i+1,
                "title": f"知识点{i+1}：未识别内容",
                "start": start,
                "end": end,
                "text": ""
            })
    
    log_info("视频切割", f"提取到{len(segments)}个知识点片段")
    
    # 保存片段信息
    try:
        segments_path = video_data_dir / "segments.json"
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_error("片段保存", f"失败：{str(e)}")
    
    return segments

# 生成标题和摘要（贴合视频内容）
def gen_title_summary(asr_text, ocr_results, segments):
    log_info("LLM生成", "开始生成标题和摘要")
    
    # 拼接所有文本
    ocr_text = "\n".join([o["ocr_text"] for o in ocr_results if o["ocr_text"]])
    segments_text = "\n".join([f"【{s['title']}】{s['text']}" for s in segments])
    
    # 构建精准提示词（避免串学科）
    prompt = f"""你是专业的课程内容分析助手，仅基于提供的视频内容生成标题和摘要：
    视频语音内容：{asr_text[:3000]}
    视频画面文字：{ocr_text[:1000]}
    视频知识点片段：{segments_text[:2000]}
    
    要求：
    1. 标题：简洁明了，贴合视频实际内容（如"计算机网络-TCP三次握手详解"）
    2. 摘要：分点列出核心知识点，仅使用视频中的内容，不添加额外信息
    3. 语言：中文，专业且易懂
    """
    
    try:
        response = Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            result_format="text",
            temperature=0.3,
            top_p=0.8
        )
        
        if response.status_code == 200:
            result = response.output["text"].strip()
            # 解析标题和摘要
            title = "课程视频"
            summary = "暂无摘要"
            
            if "标题：" in result:
                title = result.split("标题：")[1].split("\n")[0].strip()
            if "摘要：" in result:
                summary = result.split("摘要：")[1].strip()
            
            log_info("LLM生成", f"标题：{title}")
            return title, summary
        else:
            log_error("LLM生成", f"API返回错误：{response}")
            return "课程视频", "暂无摘要"
    except Exception as e:
        log_error("LLM生成", f"失败：{str(e)}")
        return "课程视频", "暂无摘要"

# 智能问答（区分视频内/外内容）
def answer_user_question(video_data_dir, question):
    log_info("智能问答", f"收到问题：{question}")
    
    # 加载视频内容
    asr_text = ""
    ocr_text = ""
    segments_text = ""
    
    try:
        # 加载ASR
        asr_path = video_data_dir / "asr_text.txt"
        if asr_path.exists():
            with open(asr_path, "r", encoding="utf-8") as f:
                asr_text = f.read()
        
        # 加载OCR
        ocr_path = video_data_dir / "global_ocr.json"
        if ocr_path.exists():
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_results = json.load(f)
                ocr_text = "\n".join([o["ocr_text"] for o in ocr_results if o["ocr_text"]])
        
        # 加载片段
        segments_path = video_data_dir / "segments.json"
        if segments_path.exists():
            with open(segments_path, "r", encoding="utf-8") as f:
                segments = json.load(f)
                segments_text = "\n".join([f"【{s['title']}】{s['text']}" for s in segments])
    except Exception as e:
        log_error("问答加载数据", f"失败：{str(e)}")
    
    # 检查是否有视频内容
    has_video_content = len(asr_text) > 10 or len(ocr_text) > 10
    
    if has_video_content:
        # 基于视频内容回答
        prompt = f"""仅使用以下视频内容回答问题，不要添加任何额外信息：
        视频语音内容：{asr_text[:2000]}
        视频画面文字：{ocr_text[:1000]}
        视频知识点：{segments_text[:1000]}
        
        问题：{question}
        
        要求：
        1. 仅使用视频中的内容回答
        2. 如果视频中没有直接答案，明确说明"视频中未提及相关内容"
        3. 语言简洁，准确，中文
        """
        
        try:
            response = Generation.call(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                result_format="text",
                temperature=0.1,
                top_p=0.7
            )
            
            if response.status_code == 200:
                answer = response.output["text"].strip()
                # 检查是否未提及
                if "未提及" in answer or len(answer) < 5:
                    # 调用通用回答
                    return answer_user_question_general(question)
                else:
                    return f"【基于本视频内容回答】\n{answer}"
            else:
                log_error("问答LLM", f"API错误：{response}")
                return answer_user_question_general(question)
        except Exception as e:
            log_error("问答LLM", f"失败：{str(e)}")
            return answer_user_question_general(question)
    else:
        # 无视频内容，调用通用回答
        return answer_user_question_general(question)

# 通用回答（声明非视频内容）
def answer_user_question_general(question):
    log_info("智能问答", "调用通用知识回答")
    
    prompt = f"""回答以下问题，要求专业、准确、易懂：
    问题：{question}
    
    要求：
    1. 中文回答
    2. 内容专业准确
    3. 语言简洁明了
    """
    
    try:
        response = Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            result_format="text",
            temperature=0.5,
            top_p=0.8
        )
        
        if response.status_code == 200:
            answer = response.output["text"].strip()
            return f"【本问题无对应视频内容，以下为通用知识回答】\n{answer}"
        else:
            log_error("通用问答", f"API错误：{response}")
            return "【本问题无对应视频内容，以下为通用知识回答】\n暂无法回答该问题，请稍后重试。"
    except Exception as e:
        log_error("通用问答", f"失败：{str(e)}")
        return "【本问题无对应视频内容，以下为通用知识回答】\n暂无法回答该问题，请检查API Key是否正确。"

# 主处理函数
def process_video(video_path, video_id, user_id=1):
    log_info("视频处理", f"开始处理视频ID：{video_id}，路径：{video_path}")
    
    # 创建视频数据目录
    video_data_dir = Path(f"data/videos/{user_id}/{video_id}")
    video_data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 提取音频
        audio_path = extract_audio(video_path, video_data_dir)
        
        # 2. ASR转写
        asr_raw, asr_text = asr_transcribe(audio_path, video_data_dir)
        
        # 3. 关键帧+OCR
        ocr_results = extract_keyframes_ocr(video_path, video_data_dir)
        
        # 4. 切割视频片段
        segments = split_video_segments(video_path, asr_raw, video_data_dir)
        
        # 5. 生成标题和摘要
        title, summary = gen_title_summary(asr_text, ocr_results, segments)
        
        # 6. 整理结果
        result = {
            "video_id": video_id,
            "user_id": user_id,
            "title": title,
            "summary": summary,
            "segments": segments,
            "asr_text": asr_text,
            "ocr_results": ocr_results,
            "status": "completed",
            "processed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存结果
        result_path = video_data_dir / "process_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        log_info("视频处理", f"视频ID：{video_id} 处理完成")
        return result
    
    except Exception as e:
        log_error("视频处理主流程", f"失败：{str(e)}")
        return {
            "video_id": video_id,
            "user_id": user_id,
            "title": "视频处理失败",
            "summary": "视频处理过程中出现错误",
            "segments": [],
            "asr_text": "",
            "ocr_results": [],
            "status": "failed",
            "error": str(e),
            "processed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }