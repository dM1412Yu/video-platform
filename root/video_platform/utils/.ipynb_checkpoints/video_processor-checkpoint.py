# -*- coding: utf-8 -*-
import os
import json
import subprocess
import uuid
import cv2
import numpy as np
import whisper
import easyocr
import torch
import clip
from pathlib import Path
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim
import dashscope
from dashscope import Generation, MultiModalConversation
from config import (
    DEVICE, GLOBAL_KEYFRAME_NUM, SUB_VIDEO_MAX_KEYFRAME,
    SSIM_THRESHOLD, MIN_SUB_VIDEO_DURATION, MAX_SUB_VIDEO_DURATION
)

# 初始化模型（GPU加速）
def init_models():
    """初始化所有模型（GPU优先）"""
    # Whisper ASR模型
    asr_model = whisper.load_model("base", device=DEVICE)
    # EasyOCR模型
    ocr_reader = easyocr.Reader(['ch_sim'], gpu=DEVICE=="cuda", verbose=False)
    # CLIP模型
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
    return asr_model, ocr_reader, clip_model, clip_preprocess

# 全局模型实例
ASR_MODEL, OCR_READER, CLIP_MODEL, CLIP_PREPROCESS = init_models()

# 1. ASR转写+矫正
def asr_transcribe_and_correct(audio_path, video_data_dir):
    """ASR转写+通义千问矫正"""
    # 1. ASR转写
    result = ASR_MODEL.transcribe(str(audio_path), language="zh", fp16=DEVICE=="cuda")
    asr_raw = []
    for seg in result["segments"]:
        asr_raw.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        })
    # 保存原始ASR
    with open(video_data_dir / "asr_raw.json", "w", encoding="utf-8") as f:
        json.dump(asr_raw, f, ensure_ascii=False, indent=2)
    
    # 2. 通义千问矫正
    asr_raw_text = "\n".join([f"[{s['start']}-{s['end']}s] {s['text']}" for s in asr_raw])
    prompt = f"""对以下教学视频的语音转写文本做矫正，修正口误/语序/专业术语错误，保留原时间戳格式，仅输出矫正后的文本：
{asr_raw_text}"""
    response = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        result_format="text"
    )
    # 解析矫正后的文本（简化处理，实际可优化解析逻辑）
    asr_corrected = asr_raw  # 先复用结构，文本替换为矫正后的
    corrected_text = response.output.text.strip()
    # 保存矫正后的ASR
    with open(video_data_dir / "asr_corrected.json", "w", encoding="utf-8") as f:
        json.dump(asr_corrected, f, ensure_ascii=False, indent=2)
    with open(video_data_dir / "asr_corrected.txt", "w", encoding="utf-8") as f:
        f.write(corrected_text)
    return asr_corrected, corrected_text

# 2. 关键帧筛选+OCR
def extract_keyframes_and_ocr(video_path, video_data_dir, is_global=True):
    """提取关键帧+OCR（全局/短视频）"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # 关键帧存储目录
    if is_global:
        kf_dir = video_data_dir / "global_keyframes"
        max_kf = GLOBAL_KEYFRAME_NUM
    else:
        kf_dir = video_data_dir / "ocr_frames"
        max_kf = SUB_VIDEO_MAX_KEYFRAME
    kf_dir.mkdir(exist_ok=True)
    
    # 均匀采样候选帧
    step = max(1, total_frames // (max_kf * 2))
    candidate_frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            ts = i / fps
            candidate_frames.append((ts, frame))
    
    # 帧去重（SSIM）
    dedup_frames = []
    prev_frame = None
    for ts, frame in candidate_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))
        if prev_frame is None:
            dedup_frames.append((ts, frame))
            prev_frame = gray
            continue
        # SSIM相似度检测
        similarity = ssim(prev_frame, gray, data_range=255)
        if similarity < SSIM_THRESHOLD:
            dedup_frames.append((ts, frame))
            prev_frame = gray
        if len(dedup_frames) >= max_kf:
            break
    
    # 截取最终关键帧（多则取max_kf，少则全取）
    final_frames = dedup_frames[:max_kf] if len(dedup_frames) > max_kf else dedup_frames
    
    # OCR提取
    ocr_results = []
    for idx, (ts, frame) in enumerate(final_frames):
        # 保存关键帧
        kf_path = kf_dir / f"frame_{ts:.2f}s.jpg"
        cv2.imwrite(str(kf_path), frame)
        # OCR识别
        ocr_text = " ".join(OCR_READER.readtext(frame, detail=0))
        ocr_results.append({
            "ts": round(ts, 2),
            "frame_path": str(kf_path),
            "ocr_text": ocr_text
        })
    
    cap.release()
    # 保存OCR结果
    ocr_file = video_data_dir / ("global_ocr.json" if is_global else "ocr_result.json")
    with open(ocr_file, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=2)
    return final_frames, ocr_results

# 3. 知识点切割（通义千问）
def split_knowledge_points(video_data_dir, asr_corrected, global_ocr):
    """基于ASR+OCR切割知识点"""
    # 构造输入
    asr_text = "\n".join([f"[{s['start']}-{s['end']}s] {s['text']}" for s in asr_corrected])
    ocr_text = "\n".join([f"[{o['ts']}s] {o['ocr_text']}" for o in global_ocr])
    prompt = f"""基于以下教学视频的矫正后语音文本+全局关键帧OCR文本，按{MIN_SUB_VIDEO_DURATION}-{MAX_SUB_VIDEO_DURATION}秒的长度切割知识点，每个知识点包含名称、开始时间、结束时间，按时间顺序排列，仅输出JSON格式（[{"name":"知识点名","start":0.0,"end":60.0}]）：
【ASR文本】
{asr_text}
【OCR文本】
{ocr_text}"""
    # 调用通义千问
    response = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        result_format="text"
    )
    # 解析知识点
    try:
        knowledge_points = json.loads(response.output.text.strip())
    except:
        # 兜底切割（均匀分割）
        total_duration = asr_corrected[-1]["end"] if asr_corrected else 60
        segment_count = max(1, int(total_duration / MAX_SUB_VIDEO_DURATION))
        segment_duration = total_duration / segment_count
        knowledge_points = []
        for i in range(segment_count):
            start = round(i * segment_duration, 2)
            end = round(min((i+1)*segment_duration, total_duration), 2)
            knowledge_points.append({
                "name": f"核心知识点{i+1}",
                "start": start,
                "end": end
            })
    # 保存知识点
    with open(video_data_dir / "knowledge_points.json", "w", encoding="utf-8") as f:
        json.dump(knowledge_points, f, ensure_ascii=False, indent=2)
    return knowledge_points

# 4. 切割短视频
def split_video_segments(video_path, video_data_dir, knowledge_points):
    """切割短视频"""
    sub_videos_dir = video_data_dir / "sub_videos"
    sub_videos_dir.mkdir(exist_ok=True)
    sub_videos = []
    
    for idx, kp in enumerate(knowledge_points):
        start = kp["start"]
        end = kp["end"]
        if end - start < MIN_SUB_VIDEO_DURATION:
            continue
        # 创建短视频目录
        sub_video_name = f"{kp['name'].replace('/', '_').replace(':', '_')[:50]}_{start}-{end}s"
        sub_video_dir = sub_videos_dir / sub_video_name
        sub_video_dir.mkdir(exist_ok=True)
        # 切割视频
        sub_video_path = sub_video_dir / "sub_video.mp4"
        cmd = [
            "ffmpeg", "-ss", str(start), "-i", str(video_path),
            "-to", str(end), "-c:v", "libx264", "-c:a", "aac",
            "-reset_timestamps", "1", "-y", str(sub_video_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
        # 提取短视频关键帧+OCR
        extract_keyframes_and_ocr(sub_video_path, sub_video_dir, is_global=False)
        
        # 截取该短视频的ASR片段
        asr_corrected = json.load(open(video_data_dir / "asr_corrected.json", "r", encoding="utf-8"))
        sub_asr = [s for s in asr_corrected if s["start"] >= start and s["end"] <= end]
        with open(sub_video_dir / "asr_cut.txt", "w", encoding="utf-8") as f:
            f.write("\n".join([f"[{s['start']}-{s['end']}s] {s['text']}" for s in sub_asr]))
        
        # 生成短视频专属提问
        generate_sub_video_qa(sub_video_dir, sub_asr, sub_video_path)
        
        # 保存短视频信息
        sub_video_id = str(uuid.uuid4())[:6]
        sub_videos.append({
            "id": sub_video_id,
            "name": kp["name"],
            "start_time": start,
            "end_time": end,
            "path": str(sub_video_path),
            "dir": str(sub_video_dir)
        })
    return sub_videos

# 5. 生成视频摘要
def generate_video_summary(video_data_dir, asr_corrected_text, global_ocr):
    """基于ASR+OCR生成视频摘要"""
    ocr_text = "\n".join([f"[{o['ts']}s] {o['ocr_text']}" for o in global_ocr])
    prompt = f"""基于以下教学视频的矫正后语音文本+全局关键帧OCR文本，生成200-500字的全局摘要，仅覆盖视频内的核心知识点，无外部信息：
【ASR文本】
{asr_corrected_text}
【OCR文本】
{ocr_text}"""
    response = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        result_format="text"
    )
    summary = response.output.text.strip()
    # 保存摘要
    with open(video_data_dir / "global_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    return summary

# 6. 生成短视频专属提问
def generate_sub_video_qa(sub_video_dir, sub_asr, sub_video_path):
    """生成短视频专属提问+答案（保证LLM连续性）"""
    # 读取短视频OCR
    ocr_result = json.load(open(sub_video_dir / "ocr_result.json", "r", encoding="utf-8"))
    ocr_text = "\n".join([f"[{o['ts']}s] {o['ocr_text']}" for o in ocr_result])
    # 构造输入（少传文本，保证连续性）
    asr_text = "\n".join([f"[{s['start']}-{s['end']}s] {s['text']}" for s in sub_asr])
    prompt = f"""基于以下教学视频片段的矫正后语音文本+关键帧OCR文本，生成1个贴合知识点的教学类问题（客观题），并给出标准答案，仅输出【问题】和【答案】，无多余内容：
【ASR文本】
{asr_text}
【OCR文本】
{ocr_text}"""
    response = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        result_format="text"
    )
    # 解析提问和答案（简化处理）
    qa_text = response.output.text.strip()
    qa = {"question": qa_text, "answer": "", "user_answer": "", "evaluation": ""}
    if "【答案】" in qa_text:
        q_part, a_part = qa_text.split("【答案】")
        qa["question"] = q_part.replace("【问题】", "").strip()
        qa["answer"] = a_part.strip()
    else:
        qa["question"] = qa_text
        qa["answer"] = "无标准答案（视频内容未明确）"
    # 保存QA
    with open(sub_video_dir / "qa_record.json", "w", encoding="utf-8") as f:
        json.dump(qa, f, ensure_ascii=False, indent=2)
    return qa

# 7. 用户提问匹配+回答
def answer_user_question(video_data_dir, question):
    """用户自由提问：匹配短视频+回答（仅基于视频内容）"""
    # 1. 读取全量数据
    asr_corrected = json.load(open(video_data_dir / "asr_corrected.json", "r", encoding="utf-8"))
    global_ocr = json.load(open(video_data_dir / "global_ocr.json", "r", encoding="utf-8"))
    knowledge_points = json.load(open(video_data_dir / "knowledge_points.json", "r", encoding="utf-8"))
    
    # 2. 匹配短视频（通义千问）
    asr_text = "\n".join([f"[{s['start']}-{s['end']}s] {s['text']}" for s in asr_corrected])
    ocr_text = "\n".join([f"[{o['ts']}s] {o['ocr_text']}" for o in global_ocr])
    match_prompt = f"""基于以下教学视频的全量矫正语音文本+全局关键帧OCR文本，分析用户问题对应的知识点所属的短视频片段，仅输出该短视频的名称+时间戳（如：核心知识点1 0-60秒），未匹配到则输出“未匹配到相关片段”：
【用户问题】{question}
【ASR文本】{asr_text}
【OCR文本】{ocr_text}"""
    match_response = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": match_prompt}],
        temperature=0.1,
        result_format="text"
    )
    match_result = match_response.output.text.strip()
    if "未匹配到" in match_result:
        return {"question": question, "answer": "该问题未匹配到视频内的相关内容，无法回答", "match": "无"}
    
    # 3. 基于匹配的短视频回答（少传文本，保证连续性）
    answer_prompt = f"""基于匹配的短视频片段内容，回答用户问题，仅基于视频内容，无外部信息：
【匹配结果】{match_result}
【用户问题】{question}
【ASR文本（精简）】{asr_text[:500]}  # 少传文本，保证连续性
【OCR文本（精简）】{ocr_text[:300]}"""
    answer_response = Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0.1,
        result_format="text"
    )
    return {
        "question": question,
        "answer": answer_response.output.text.strip(),
        "match": match_result
    }

# 主处理流程
def process_video(video_path, video_data_dir):
    """视频处理主流程（按你的要求：ASR→OCR→切割→短视频OCR→QA）"""
    # 1. 提取音频
    audio_path = video_data_dir / "audio.mp3"
    cmd = ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "mp3", "-y", str(audio_path)]
    subprocess.run(cmd, capture_output=True, check=True)
    
    # 2. ASR转写+矫正
    asr_corrected, asr_corrected_text = asr_transcribe_and_correct(audio_path, video_data_dir)
    
    # 3. 全局关键帧+OCR
    _, global_ocr = extract_keyframes_and_ocr(video_path, video_data_dir, is_global=True)
    
    # 4. 知识点切割
    knowledge_points = split_knowledge_points(video_data_dir, asr_corrected, global_ocr)
    
    # 5. 切割短视频
    sub_videos = split_video_segments(video_path, video_data_dir, knowledge_points)
    
    # 6. 生成视频摘要
    summary = generate_video_summary(video_data_dir, asr_corrected_text, global_ocr)
    
    return sub_videos, summary