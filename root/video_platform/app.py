import os
import json
import uuid
import sqlite3
import subprocess
import traceback
import logging
import re
from pathlib import Path
from datetime import datetime

import torch
import whisper
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ====================== 日志配置 ======================
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - line %(lineno)d - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

setup_logging()
logger = logging.getLogger(__name__)

# ====================== 导入 core 模块（仅导入一次，避免变量冲突） ======================
try:
    from core import correct_asr
    from core import generate_web_title_summary
    from core import relation_network
    from core import generate_questions
    from core.clip_vlm_segment import run_knowledge_split  # 仅导入一次，核心修复点
    
    try:
        from core import knowledge_graph
        logger.info("✅ 知识图谱模块导入成功")
    except ImportError:
        logger.warning("⚠️ 未找到知识图谱核心模块，将使用兜底逻辑")
        knowledge_graph = None
    logger.info("✅ 所有core模块导入成功")
except ImportError as e:
    logger.error(f"❌ core模块导入失败：{e}，详情：{traceback.format_exc()}")
    raise

# ====================== 导入 answer_question 模块（全局作用域，修正缩进） ======================
try:
    from core.answer_question import run_answer_question
    logger.info("✅ 问答模块answer_question导入成功")
except ImportError as e:
    logger.error(f"❌ 问答模块导入失败：{e}")
    # 兜底：定义一个空函数，避免接口崩溃
    def run_answer_question(video_id, question):
        return {
            "answer": "问答模块加载失败，无法回答",
            "is_video_related": False,
            "matched_subvideo": "",
            "error": str(e),
            "code": 500
        }

# ====================== 全局配置 ======================
app = Flask(__name__)
app.secret_key = "super_secret_key_2025"
BASE_DIR = Path(__file__).resolve().parent
app.config["UPLOAD_FOLDER"] = str(BASE_DIR / "uploads")
app.config["VIDEO_DATA"] = str(BASE_DIR / "video_data")
app.config["MAX_CONTENT_LENGTH"] = 2147483648  # 2GB
app.config["DATABASE"] = str(BASE_DIR / "videos.db")
# 核心修改1：移除全局硬编码的CUSTOM_KG_PATH，改为动态生成
# app.config["CUSTOM_KG_PATH"] = str(BASE_DIR / "custom_kg.json")
# 核心修改2：移除无用的VIDEO_KG_DIR配置（知识图谱已改到video_data/视频ID/下）
# app.config["VIDEO_KG_DIR"] = str(BASE_DIR / "video_kg")
app.config["FFMPEG_PATH"] = "ffmpeg"

# 创建必要目录（仅保留必需的）
for dir_path in [app.config["UPLOAD_FOLDER"], app.config["VIDEO_DATA"]]:
    Path(dir_path).mkdir(exist_ok=True)

# ====================== 模型加载（增强容错） ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = None
try:
    whisper_model = whisper.load_model("base", device=DEVICE)
    logger.info(f"✅ Whisper模型加载成功，设备：{DEVICE}")
except Exception as e:
    logger.warning(f"⚠️ Whisper模型加载失败：{e}，将使用兜底ASR逻辑")

# ====================== 数据库初始化 ======================
def init_db():
    with sqlite3.connect(app.config["DATABASE"]) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                status TEXT DEFAULT 'processing',
                raw_asr TEXT,
                corrected_asr TEXT,
                knowledge_segments TEXT,
                title TEXT,
                summary TEXT,
                knowledge_graph TEXT,
                error_log TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
init_db()

# ====================== 工具函数（全加固） ======================
def update_video_status(video_id, status, error_log=None):
    """更新视频处理状态"""
    with sqlite3.connect(app.config["DATABASE"]) as conn:
        params = [status, video_id]
        sql = "UPDATE videos SET status=? WHERE id=?"
        if error_log:
            sql = "UPDATE videos SET status=?, error_log=? WHERE id=?"
            params.insert(1, error_log[:2000])
        conn.execute(sql, params)
        conn.commit()

def update_video_data(video_id, data_dict):
    """更新视频详情数据"""
    try:
        with sqlite3.connect(app.config["DATABASE"]) as conn:
            valid_fields = ["raw_asr", "corrected_asr", "knowledge_segments", "title", "summary", "knowledge_graph"]
            update_fields = [f"{k}=?" for k in valid_fields if k in data_dict]
            if not update_fields:
                return
            sql = f"UPDATE videos SET {','.join(update_fields)} WHERE id=?"
            params = [data_dict[k] for k in valid_fields if k in data_dict] + [video_id]
            conn.execute(sql, params)
            conn.commit()
    except Exception as e:
        logger.error(f"❌ 更新视频数据失败：{e}")

def get_video_data(video_id):
    """获取视频完整数据"""
    try:
        with sqlite3.connect(app.config["DATABASE"]) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM videos WHERE id=?", (video_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"❌ 获取视频数据失败：{e}")
        return None

def safe_json_load(s, default=None):
    """安全解析JSON"""
    if default is None:
        default = []
    try:
        if isinstance(s, (bytes, str)) and s.strip():
            return json.loads(s)
        return default
    except Exception as e:
        logger.error(f"❌ JSON解析失败：{e}")
        return default

def format_knowledge_segments(segments):
    """格式化知识点数据（完整逻辑，核心修复点）"""
    formatted = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        # 兼容start/start_time、end/end_time字段
        start = float(seg.get("start_time", seg.get("start", 0.0)))
        end = float(seg.get("end_time", seg.get("end", 0.0)))
        questions = seg.get("questions", [])
        
        # 格式化题目
        formatted_questions = []
        for q in questions:
            if isinstance(q, dict) and "question" in q and "answer" in q:
                formatted_questions.append({
                    "question": q["question"].strip(),
                    "answer": q["answer"].strip()
                })
        
        # 组装格式化数据
        formatted.append({
            "title": seg.get("title", f"知识点{len(formatted)+1}").strip(),
            "start_time": start,
            "end_time": end,
            "start": start,  # 兼容前端模板
            "end": end,      # 兼容前端模板
            "questions": formatted_questions,
            "content": seg.get("content", seg.get("knowledge_point", ""))
        })
    return formatted

# ====================== 主处理流程（全修复+全兜底） ======================
def process_video(video_id, video_path):
    error_log = []
    work_dir = os.path.join(app.config["VIDEO_DATA"], str(video_id))
    Path(work_dir).mkdir(exist_ok=True)
    audio_path = os.path.join(work_dir, "audio.wav")
    corrected_asr_path = os.path.join(work_dir, "corrected_asr.txt")
    
    # 提前定义所有文件路径，核心修复点
    raw_asr_txt_path = os.path.join(work_dir, "raw_asr.txt")
    raw_asr_json_path = os.path.join(work_dir, "raw_asr_segments.json")
    split_result_path = os.path.join(work_dir, "final_knowledge_splits.json")
    subvideo_summary_path = os.path.join(work_dir, "subvideo_summaries_all.json")
    # 核心修改3：动态生成当前视频的custom_kg路径
    video_custom_kg_path = os.path.join(work_dir, f"custom_kg_{video_id}.json")

    try:
        # 1. 获取视频时长（资源释放加固，核心修复点）
        video_duration = 600.0
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_duration = total_frames / fps if fps > 0 else 600.0
        finally:
            if cap:
                cap.release()  # 确保释放，核心修复点

        # 2. 提取音频
        update_video_status(video_id, "processing_audio")
        try:
            subprocess.run([
                app.config["FFMPEG_PATH"], "-i", video_path, "-vn",
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path
            ], check=True, capture_output=True, text=True)
            logger.info(f"✅ 音频提取完成：{audio_path}")
        except Exception as e:
            error_log.append(f"音频提取失败：{str(e)}")
            logger.error(f"❌ 音频提取失败：{e}")

        # 3. ASR语音识别（全兜底，核心修复点）
        update_video_status(video_id, "processing_asr")
        raw_asr = ""
        raw_asr_segments = []
        
        if whisper_model:
            try:
                result = whisper_model.transcribe(
                    audio_path, 
                    language="zh", 
                    verbose=False, 
                    word_timestamps=True
                )
                # 生成纯文本ASR
                raw_asr = "".join([seg["text"].strip() for seg in result["segments"]])
                with open(raw_asr_txt_path, "w", encoding="utf-8") as f:
                    f.write(raw_asr)
                # 生成带时间戳的结构化ASR
                raw_asr_segments = [
                    {
                        "start": round(seg["start"], 2),
                        "end": round(seg["end"], 2),
                        "text": seg["text"].strip(),
                        "segment_id": idx + 1
                    }
                    for idx, seg in enumerate(result["segments"])
                ]
                with open(raw_asr_json_path, "w", encoding="utf-8") as f:
                    json.dump(raw_asr_segments, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ ASR识别完成：{len(raw_asr_segments)}个片段")
            except Exception as e:
                error_log.append(f"ASR识别失败：{str(e)}")
                logger.error(f"❌ ASR识别失败：{e}", exc_info=True)
                # 兜底生成空文件
                with open(raw_asr_txt_path, "w", encoding="utf-8") as f:
                    f.write("")
                with open(raw_asr_json_path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
        else:
            # Whisper模型未加载时的兜底逻辑
            logger.warning("⚠️ Whisper模型未加载，生成兜底ASR文件")
            raw_asr = "Whisper模型未加载，兜底纯文本ASR"
            with open(raw_asr_txt_path, "w", encoding="utf-8") as f:
                f.write(raw_asr)
            # 基于视频时长生成兜底结构化ASR
            raw_asr_segments = [{
                "start": 0.0,
                "end": video_duration,
                "text": "Whisper模型未加载，兜底带时间戳ASR片段",
                "segment_id": 1
            }]
            with open(raw_asr_json_path, "w", encoding="utf-8") as f:
                json.dump(raw_asr_segments, f, ensure_ascii=False, indent=2)

        # 4. ASR矫正（参数严格匹配）
        update_video_status(video_id, "processing_correct_asr")
        corrected_asr = raw_asr
        try:
            corrected_asr = correct_asr.run_correct_asr(
                video_id=video_id, 
                raw_asr_text=raw_asr, 
                video_data_dir=app.config["VIDEO_DATA"]
            )
            with open(corrected_asr_path, "w", encoding="utf-8") as f:
                f.write(corrected_asr)
            logger.info(f"✅ ASR矫正完成：{corrected_asr_path}")
        except Exception as e:
            error_log.append(f"ASR矫正失败：{str(e)}")
            logger.error(f"❌ ASR矫正失败：{e}")
            # 兜底保存原始ASR
            with open(corrected_asr_path, "w", encoding="utf-8") as f:
                f.write(raw_asr)

        # 5. 知识点拆分（变量初始化+全兜底，核心修复点）
        update_video_status(video_id, "processing_knowledge_split")
        split_result = []  # 初始化，核心修复点
        try:
            split_result = run_knowledge_split(video_path, corrected_asr_path, video_id)
            with open(split_result_path, "w", encoding="utf-8") as f:
                json.dump(split_result, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 知识点拆分完成：{len(split_result)}个片段")
        except Exception as e:
            error_log.append(f"知识点拆分失败：{str(e)}")
            logger.error(f"❌ 知识点拆分失败：{e}", exc_info=True)
            # 强制兜底赋值
            split_result = [{
                "id": 1,
                "title": "核心知识点",
                "start_time": 0.0,
                "end_time": video_duration,
                "knowledge_point": "核心知识点",
                "questions": []
            }]
            with open(split_result_path, "w", encoding="utf-8") as f:
                json.dump(split_result, f, ensure_ascii=False, indent=2)

        # 6. Video Summary（文件存在性校验+全兜底，核心修复点）
        update_video_status(video_id, "processing_video_summary")
        try:
            # 读取拆分结果（优先读文件，无文件则用内存数据）
            if os.path.exists(split_result_path):
                with open(split_result_path, "r", encoding="utf-8") as f:
                    split_result = json.load(f)
            else:
                logger.warning("⚠️ 拆分结果文件不存在，使用内存兜底数据")
            
            # 生成subvideo_summaries_all.json
            subvideo_summaries = []
            for idx, seg in enumerate(split_result):
                subvideo_summaries.append({
                    "subvideo_id": f"{video_id}_{idx+1}",
                    "start_time": seg.get("start_time", 0.0),
                    "end_time": seg.get("end_time", video_duration),
                    "summary": seg.get("knowledge_point", f"知识点{idx+1}"),
                    "title": seg.get("knowledge_point", f"知识点{idx+1}")
                })
            with open(subvideo_summary_path, "w", encoding="utf-8") as f:
                json.dump(subvideo_summaries, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Video Summary完成：生成{len(subvideo_summaries)}个子视频摘要")
        except Exception as e:
            error_log.append(f"video_summary执行失败：{str(e)}")
            logger.error(f"❌ video_summary执行失败：{e}", exc_info=True)
            # 兜底生成subvideo_summaries_all.json
            subvideo_summaries = [{
                "subvideo_id": f"{video_id}_1",
                "start_time": 0.0,
                "end_time": video_duration,
                "summary": "核心知识点",
                "title": "核心知识点"
            }]
            with open(subvideo_summary_path, "w", encoding="utf-8") as f:
                json.dump(subvideo_summaries, f, ensure_ascii=False, indent=2)

        # 7. 知识图谱生成（参数严格匹配+动态路径）
        update_video_status(video_id, "processing_knowledge_graph")
        kg_json = json.dumps({"concept_hierarchy": [], "relations": []}, ensure_ascii=False)
        try:
            if knowledge_graph and hasattr(knowledge_graph, 'generate_video_kg'):
                custom_kg = knowledge_graph.generate_video_kg(
                    video_id=video_id,
                    # 核心修改4：传入动态生成的custom_kg路径（匹配知识图谱模块的输出路径）
                    custom_kg_path=video_custom_kg_path
                )
                kg_json = json.dumps(custom_kg, ensure_ascii=False)
                logger.info(f"✅ 知识图谱生成完成：{video_custom_kg_path}")
            else:
                logger.warning("⚠️ 知识图谱模块未导入，使用兜底数据")
                # 兜底生成视频专属的custom_kg文件
                fallback_kg = {"concept_hierarchy": [], "relations": []}
                with open(video_custom_kg_path, "w", encoding="utf-8") as f:
                    json.dump(fallback_kg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 异常时兜底生成视频专属的custom_kg文件
            fallback_kg = {"concept_hierarchy": [], "relations": []}
            with open(video_custom_kg_path, "w", encoding="utf-8") as f:
                json.dump(fallback_kg, f, ensure_ascii=False, indent=2)
            error_log.append(f"知识图谱生成失败：{str(e)}")
            logger.error(f"❌ 知识图谱生成失败：{e}", exc_info=True)

        # 8. 标题摘要生成（参数严格匹配）
        update_video_status(video_id, "processing_title_summary")
        title, summary = f"视频{video_id}", corrected_asr[:200]
        try:
            if hasattr(generate_web_title_summary, 'run_generate_web_title_summary'):
                title, summary = generate_web_title_summary.run_generate_web_title_summary(
                    corrected_asr=corrected_asr  # 核心参数
                )
                logger.info(f"✅ 标题摘要生成完成：{title}")
            else:
                logger.warning("⚠️ 标题摘要模块无对应函数，使用兜底数据")
        except Exception as e:
            error_log.append(f"标题摘要生成失败：{str(e)}")
            logger.error(f"❌ 标题摘要生成失败：{e}")

        # 9. 关系网生成（参数严格匹配+动态custom_kg路径）
        update_video_status(video_id, "processing_relation")
        try:
            # 兜底：确保视频专属的custom_kg文件存在
            if not os.path.exists(video_custom_kg_path):
                fallback_kg = {"concept_hierarchy": [], "relations": []}
                with open(video_custom_kg_path, "w", encoding="utf-8") as f:
                    json.dump(fallback_kg, f, ensure_ascii=False, indent=2)
            
            if hasattr(relation_network, 'generate_relation_network'):
                relation_network.generate_relation_network(
                    video_id=str(video_id),  # 强制字符串类型，核心修复点
                    work_dir=work_dir, 
                    # 核心修改5：传入视频专属的custom_kg路径（而非全局路径）
                    custom_kg_path=video_custom_kg_path
                )
                logger.info(f"✅ 关系网生成完成（使用{video_custom_kg_path}）")
            else:
                logger.warning("⚠️ relation_network无generate_relation_network函数，跳过")
        except Exception as e:
            error_log.append(f"关系网生成失败：{str(e)}")
            logger.error(f"❌ 关系网生成失败：{e}", exc_info=True)

        # 10. 生成知识点题目（核心补全，放在关系网之后）
        update_video_status(video_id, "processing_generate_questions")
        questions_result = []
        q_file = os.path.join(app.config["VIDEO_DATA"], str(video_id), "knowledge_questions.json")
        try:
            # 调用生成问题模块核心函数
            if hasattr(generate_questions, 'generate_questions_for_knowledge_points'):
                questions_result = generate_questions.generate_questions_for_knowledge_points(
                    video_id=video_id,
                    video_data_dir=app.config["VIDEO_DATA"]
                )
                logger.info(f"✅ 题目生成完成：{len(questions_result)}个知识点，共{sum(len(seg.get('questions', [])) for seg in questions_result)}道题")
            else:
                # 兜底：手动生成基础题目
                questions_result = []
                split_result = safe_json_load(split_result_path) if os.path.exists(split_result_path) else []
                for idx, seg in enumerate(split_result):
                    questions_result.append({
                        "title": seg.get("title", f"知识点{idx+1}"),
                        "start": seg.get("start_time", 0.0),
                        "end": seg.get("end_time", 0.0),
                        "segment_folder": f"t{idx+1}",
                        "segment_id": idx+1,
                        "questions": [
                            {"type": "引导题", "question": f"{seg.get('knowledge_point', '知识点')}的核心概念是什么？", "answer": "暂无标准答案"},
                            {"type": "考查题", "question": f"{seg.get('knowledge_point', '知识点')}的应用场景有哪些？", "answer": "暂无标准答案"}
                        ]
                    })
            # 保存题目文件（前端依赖）
            with open(q_file, "w", encoding="utf-8") as f:
                json.dump(questions_result, f, ensure_ascii=False, indent=2)
            # 更新拆分结果，合并题目
            split_result = questions_result
        except Exception as e:
            error_log.append(f"题目生成失败：{str(e)}")
            logger.error(f"❌ 题目生成失败：{e}", exc_info=True)
            # 兜底生成题目文件
            questions_result = [{
                "title": "核心知识点",
                "start": 0.0,
                "end": 600.0,
                "segment_folder": "t1",
                "segment_id": 1,
                "questions": [
                    {"type": "引导题", "question": "该知识点的核心内容是什么？", "answer": "暂无标准答案"},
                    {"type": "考查题", "question": "该知识点的关键步骤有哪些？", "answer": "暂无标准答案"}
                ]
            }]
            with open(q_file, "w", encoding="utf-8") as f:
                json.dump(questions_result, f, ensure_ascii=False, indent=2)

        # 11. 数据入库（最终兜底）
        update_video_status(video_id, "completed", "\n".join(error_log))
        update_video_data(video_id, {
            "raw_asr": raw_asr,
            "corrected_asr": corrected_asr,
            "knowledge_segments": json.dumps(split_result, ensure_ascii=False),
            "knowledge_graph": kg_json,
            "title": title,
            "summary": summary,
        })
        logger.info(f"✅ 视频{video_id}处理完成！")

    except Exception as e:
        # 主异常捕获（修复SyntaxError的核心）
        error_msg = str(e)
        error_log.append(error_msg)
        update_video_status(video_id, "failed", error_msg)
        logger.error(f"❌ 视频{video_id}处理失败：{e}", exc_info=True)

# ====================== 核心路由（完整逻辑） ======================
@app.route("/video_detail/<int:video_id>")
def video_detail(video_id):
    video_data = get_video_data(video_id)
    if not video_data:
        flash("❌ 视频不存在")
        return redirect("/")

    # 读取知识点题目文件
    q_file = os.path.join(app.config["VIDEO_DATA"], str(video_id), "knowledge_questions.json")
    knowledge_segments = []
    if os.path.exists(q_file):
        try:
            with open(q_file, "r", encoding="utf-8") as f:
                raw_segments = json.load(f)
            knowledge_segments = format_knowledge_segments(raw_segments)
        except Exception as e:
            logger.error(f"❌ 读取题目文件失败：{e}")
            knowledge_segments = [{
                "title": "核心知识点",
                "start_time": 0.0,
                "end_time": 600.0,
                "questions": [],
                "content": ""
            }]
    else:
        db_segments = safe_json_load(video_data.get("knowledge_segments", "[]"))
        knowledge_segments = format_knowledge_segments(db_segments)

    # 组装前端模板数据
    template_data = {
        "video_id": video_id,
        "title": video_data.get("title", f"视频{video_id}"),
        "filename": video_data.get("filename", ""),
        "filepath": os.path.basename(video_data.get("filepath", "")),
        "summary": video_data.get("summary", "暂无摘要"),
        "corrected_asr": video_data.get("corrected_asr", ""),
        "knowledge_segments": knowledge_segments,
        "segments": knowledge_segments  # 兼容前端模板
    }

    return render_template("video_detail.html", **template_data)

@app.route("/")
def index():
    try:
        with sqlite3.connect(app.config["DATABASE"]) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT id, filename, status, title, created_at, error_log FROM videos ORDER BY id DESC")
            videos = [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"❌ 读取视频列表失败：{e}")
        videos = []
    return render_template("index.html", videos=videos)

@app.route("/upload", methods=["GET","POST"])
def upload():
    if request.method == "POST":
        try:
            f = request.files.get("video_file")
            if not f or f.filename == "":
                flash("❌ 未选择文件")
                return redirect(url_for("upload"))
            
            # 安全处理文件名
            original_fn = secure_filename(f.filename)
            unique_fn = f"{uuid.uuid4().hex}_{original_fn}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_fn)
            f.save(save_path)
            
            # 入库
            with sqlite3.connect(app.config["DATABASE"]) as conn:
                cur = conn.execute(
                    "INSERT INTO videos (filename, filepath, status) VALUES (?,?, 'processing')",
                    (original_fn, save_path)
                )
                vid = cur.lastrowid
                conn.commit()
            
            # 异步处理（daemon=False，核心修复点）
            import threading
            threading.Thread(target=process_video, args=(vid, save_path), daemon=False).start()
            flash(f"✅ 上传成功！视频ID：{vid}")
        except Exception as e:
            logger.error(f"❌ 文件上传失败：{e}", exc_info=True)
            flash(f"❌ 上传失败：{str(e)}")
        return redirect("/")
    return render_template("upload.html")

import logging
from flask import request, jsonify
import dashscope
from dashscope import Generation
import json

# 配置千问API密钥（和你生成题目时的密钥一致）
DASHSCOPE_API_KEY = "sk-24feda02d5524ed89a3ff3f5e0cee735"
dashscope.api_key = DASHSCOPE_API_KEY

# 初始化日志
logger = logging.getLogger(__name__)

@app.route("/api/analyze_answer", methods=["POST"])
def analyze_answer():
    try:
        # ========== 第一步：从前端请求中获取分析所需的信息 ==========
        # 读取前端传入的JSON请求体
        data = request.get_json()
        # 1. 获取用户回答（必须）
        user_ans = data.get("user_answer", "").strip()
        # 2. 获取标准答案（必须）
        std_ans = data.get("standard_answer", "").strip()
        # 3. 获取题目（可选，有则分析更精准）
        question = data.get("question", "").strip()

        # ========== 第二步：调用千问做智能分析 ==========
        def call_qwen_analysis():
            # 兜底返回None，避免内部报错影响主流程
            qwen_result = None
            try:
                # 构建给千问的提示词（包含题目、用户回答、标准答案）
                prompt = f"""
请你作为计算机网络学科的评卷老师，完成以下回答分析：
1. 题目：{question if question else '无'}
2. 标准答案：{std_ans}
3. 用户回答：{user_ans}

要求：
1. 仅判断用户回答是否正确（核心语义匹配即可，不要求逐字一致）；
2. 分析说明控制在50字以内，指出核心对错原因；
3. 只返回JSON格式，字段：is_correct（布尔值）、analysis（字符串）；
4. 不要加任何多余文字、Markdown格式（如```json```）。
"""
                # 调用千问API
                response = Generation.call(
                    model="qwen-plus",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,  # 0随机性，保证分析稳定
                    result_format="text",
                    timeout=20  # 延长超时时间，避免网络波动
                )

                # 解析千问返回结果
                if response.status_code == 200 and response.output.text:
                    raw_resp = response.output.text.strip()
                    # 清理千问可能返回的多余格式（比如```json```）
                    if raw_resp.startswith("```") and raw_resp.endswith("```"):
                        raw_resp = raw_resp[raw_resp.find("{"):raw_resp.rfind("}")+1]
                    # 解析JSON
                    qwen_result = json.loads(raw_resp)
            except Exception as e:
                # 千问调用/解析失败，记录日志但不崩溃
                logger.warning(f"千问分析失败（回退到简单匹配）：{str(e)[:100]}")
            return qwen_result

        # ========== 第三步：执行分析（失败则回退） ==========
        qwen_result = call_qwen_analysis()
        if qwen_result and isinstance(qwen_result, dict):
            # 千问分析成功，用千问的结果
            is_correct = qwen_result.get("is_correct", False)
            analysis = qwen_result.get("analysis", "")
        else:
            # 千问分析失败，回退到原始字符串匹配
            user_ans_lower = user_ans.lower()
            std_ans_lower = std_ans.lower()
            is_correct = user_ans_lower in std_ans_lower or std_ans_lower in user_ans_lower
            analysis = "✅ 回答正确，准确覆盖核心知识点！" if is_correct else "❌ 回答有误，建议参考标准答案完善。"

        # ========== 第四步：返回结果（格式和原始版本一致） ==========
        return jsonify({
            "is_correct": is_correct,
            "analysis": analysis,
            "standard_answer": std_ans  # 原样返回标准答案
        })

    except Exception as e:
        # 最外层异常捕获，避免接口崩溃
        logger.error(f"答案分析接口异常：{e}")
        return jsonify({
            "is_correct": False,
            "analysis": "❌ 回答分析失败，请重试！",
            "standard_answer": ""
        })

@app.route("/api/ask", methods=["POST"])
def ask_ai():
    try:
        data = request.get_json()
        # 1. 校验核心参数（video_id是你的模块必需参数，必须传）
        video_id = data.get("video_id")
        question = data.get("question", "").strip()
        
        # 2. 参数合法性校验
        if not video_id:
            return jsonify({"answer": "❌ 缺少必要参数：video_id（视频ID）！"})
        if not question:
            return jsonify({"answer": "❌ 请输入有效的问题！"})
        
        # 3. 调用你的专业问答模块（核心）
        logger.info(f"调用answer_question模块 | video_id：{video_id} | 问题：{question}")
        qa_result = run_answer_question(video_id, question)
        
        # 4. 适配前端返回格式（保留原有接口的返回字段，兼容前端）
        return jsonify({
            "answer": qa_result.get("answer", "暂无有效回答"),
            # 额外返回模块的核心信息（可选，前端可用于扩展）
            "is_video_related": qa_result.get("is_video_related", False),
            "matched_subvideo": qa_result.get("matched_subvideo", ""),
            "code": qa_result.get("code", 200)
        })
    except Exception as e:
        logger.error(f"❌ AI问答失败：{e}", exc_info=True)
        return jsonify({
            "answer": f"❌ 问答服务异常：{str(e)[:50]}，请稍后重试！",
            "is_video_related": False,
            "matched_subvideo": "",
            "code": 500
        })

# ====================== 静态文件路由 ======================
@app.route('/uploads/<path:filename>')
def uploads(filename):
    try:
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except Exception as e:
        logger.error(f"❌ 读取上传文件失败：{e}")
        return "❌ 视频文件不存在", 404

@app.route("/video_data/<int:video_id>/knowledge_questions.json")
def serve_questions(video_id):
    try:
        q_dir = os.path.join(app.config["VIDEO_DATA"], str(video_id))
        return send_from_directory(q_dir, "knowledge_questions.json")
    except Exception as e:
        logger.error(f"❌ 读取题目文件失败：{e}")
        return jsonify([]), 200

# ====================== 启动服务 ======================
if __name__ == "__main__":
    logger.info("🚀 服务启动中...")
    app.run(host="0.0.0.0", port=6006, debug=False, threaded=True)