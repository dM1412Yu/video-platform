from flask import Flask, render_template, request, redirect, url_for, flash, abort, jsonify, send_from_directory, session
import os
import re
import time
import json
import threading
import logging
import subprocess
from pathlib import Path

import torch
import whisper
from werkzeug.utils import secure_filename

# ==========================================
# 0. 导入核心 AI 后端模块
# ==========================================
import sys
# 确保能够导入 core 文件夹下的模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.answer_question import run_answer_question
from core.generate_web_title_summary import run_generate_web_title_summary
# 预先导入流水线函数（用于后台线程调用）
from core.clip_vlm_segment import run_knowledge_split
from core.video_summary import run_video_summary
from core.knowledge_graph import generate_video_kg
from core.relation_network import generate_relation_network
from core.generate_questions import generate_questions_for_knowledge_points
from core.correct_asr import run_correct_asr

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ==========================================
# 全局环境与内存状态配置
# ==========================================
BASE_DIR = "/root/video_platform"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
VIDEO_DATA_DIR = os.path.join(BASE_DIR, "video_data")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEO_DATA_DIR, exist_ok=True)

# 轻量内存态档案（演示及临时存储）
LEARNING_MEMORY = {}
USER_VIDEOS = {}
NEXT_VIDEO_ID = 50

ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 辅助工具函数
# ==========================================
def _current_user_id():
    return session.get('user') or 'guest'

def _is_allowed_video(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_VIDEO_EXTS

def _tokenize_text(text):
    if not text:
        return []
    return re.findall(r"[\u4e00-\u9fff]{1,}|[a-zA-Z0-9_]+", text.lower())

def _evaluate_answer(user_answer, standard_answer):
    user_tokens = set(_tokenize_text(user_answer))
    std_tokens = set(_tokenize_text(standard_answer))

    if not std_tokens:
        coverage = 0.0
    else:
        coverage = len(user_tokens & std_tokens) / max(1, len(std_tokens))

    prefix_hit = bool(
        standard_answer and len(standard_answer) >= 3 and standard_answer[:3].lower() in (user_answer or "").lower()
    )

    is_correct = coverage >= 0.45 or prefix_hit
    score = min(100, int(coverage * 100) + (10 if prefix_hit else 0))

    if score >= 80:
        level, comment = "优秀", "回答结构清晰，核心概念覆盖完整。"
    elif score >= 55:
        level, comment = "良好", "关键点基本到位，再补充定义或应用场景会更好。"
    else:
        level, comment = "待提升", "已开始思考，但核心知识点覆盖不足，建议回看本节重点。"

    strengths, weaknesses = [], []
    if score >= 60: strengths.append("抓住了问题主干")
    if len((user_answer or "").strip()) >= 25: strengths.append("回答较完整，表达有延展")
    if not strengths: strengths.append("有主动作答意识")

    if score < 55: weaknesses.append("关键术语覆盖不足")
    if len((user_answer or "").strip()) < 12: weaknesses.append("回答偏短，论证不充分")
    if not weaknesses: weaknesses.append("可增加例子让答案更有说服力")

    return {"is_correct": is_correct, "score": score, "level": level, "comment": comment, "strengths": strengths, "weaknesses": weaknesses}

def _build_student_profile(memory):
    attempts = memory["attempts"]
    correct = memory["correct"]
    total_chars = memory["total_chars"]
    accuracy = (correct / attempts) if attempts else 0.0
    avg_len = (total_chars / attempts) if attempts else 0.0

    mastery_level = "高" if accuracy >= 0.8 else ("中" if accuracy >= 0.55 else "基础")
    difficulty = "挑战" if accuracy >= 0.8 else ("进阶" if accuracy >= 0.55 else "基础")
    learning_style = "解释型" if avg_len >= 35 else ("均衡型" if avg_len >= 18 else "速答型")
    confidence = min(100, int(accuracy * 100 * 0.7 + min(avg_len, 40) * 0.8))

    return {"mastery_level": mastery_level, "learning_style": learning_style, "confidence": confidence, "recommended_difficulty": difficulty, "accuracy": round(accuracy * 100, 1)}

def _build_path_recommendation(profile, memory, segment_title):
    segment_title = segment_title or "当前知识点"
    recent = memory["history"][-3:]
    recent_wrong = sum(1 for item in recent if not item["is_correct"])

    if recent_wrong >= 2:
        return {"action": "回看复盘", "recommendation": f"建议先返回前面重看「{segment_title}」并完成1道基础题，再继续后续内容。"}
    elif profile["recommended_difficulty"] == "挑战":
        return {"action": "深入学习", "recommendation": f"你可以深入学习「{segment_title}」的拓展应用，并尝试跨章节综合题。"}
    elif profile["recommended_difficulty"] == "进阶":
        return {"action": "稳步进阶", "recommendation": f"建议在当前章节继续进阶训练；若遇卡点，再回看「{segment_title}」关键定义。"}
    else:
        return {"action": "补齐前置", "recommendation": f"建议先学习前置基础，再回到「{segment_title}」做一次复答。"}

def _extract_video_qa_items(video_id):
    qa_path = os.path.join(VIDEO_DATA_DIR, str(video_id), 'knowledge_questions.json')
    if not os.path.exists(qa_path): return []
    try:
        with open(qa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = []
        for block in data if isinstance(data, list) else []:
            title = block.get('title', '未命名知识点')
            for qa in block.get('questions', []):
                q, a = (qa.get('question') or '').strip(), (qa.get('answer') or '').strip()
                if q and a: items.append({'title': title, 'question': q, 'answer': a})
        return items
    except:
        return []

def _load_video_segments(video_id):
    splits_path = os.path.join(VIDEO_DATA_DIR, str(video_id), "final_knowledge_splits.json")
    if os.path.exists(splits_path):
        try:
            with open(splits_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [{"title": s.get("knowledge_point", "知识点"), "start_time": s.get("start_time", 0), "end_time": s.get("end_time", 0)} for s in data]
        except: pass
    return []

# ==========================================
# 静态资源路由 (适配前端加载)
# ==========================================
@app.route('/uploads/<filename>')
def uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/video_data/<int:video_id>/<path:filename>')
def video_data(video_id, filename):
    per_video_dir = os.path.join(VIDEO_DATA_DIR, str(video_id))
    if os.path.exists(os.path.join(per_video_dir, filename)):
        return send_from_directory(per_video_dir, filename)
    return send_from_directory(VIDEO_DATA_DIR, filename)

# ==========================================
# 页面路由
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/workspace')
def workspace():
    demo_videos = [
        {'id': 1, 'filename': '高等数学-微积分基础.mp4', 'status': 'completed', 'title': '微积分入门', 'summary': '核心概念与应用'}
    ]
    user_id = _current_user_id()
    videos = list(reversed(USER_VIDEOS.get(user_id, []))) if USER_VIDEOS.get(user_id) else demo_videos
    student_overview = {
        'total_videos': len(videos),
        'completed': sum(1 for v in videos if v['status'] == 'completed'),
        'processing': sum(1 for v in videos if v['status'] == 'processing'),
        'learning_hours': 12.5
    }
    return render_template('workspace.html', videos=videos, student_overview=student_overview)

# ==========================================
# ASR 处理辅助函数
# ==========================================
def _extract_audio_for_asr(video_path, audio_path):
    """从视频中提取单声道16k wav，供 Whisper 使用"""
    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn',
        '-ac', '1',
        '-ar', '16000',
        '-f', 'wav',
        audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _generate_raw_asr_files(video_path, work_dir):
    """
    真实提取 ASR：
    1. 调 Whisper 生成分段字幕
    2. 保存 raw_asr_segments.json
    3. 拼接 raw_asr_text.txt
    """
    audio_path = os.path.join(work_dir, 'temp_asr.wav')
    raw_segments_path = os.path.join(work_dir, 'raw_asr_segments.json')
    raw_text_path = os.path.join(work_dir, 'raw_asr_text.txt')

    try:
        logger.info('开始提取音频供ASR识别')
        _extract_audio_for_asr(video_path, audio_path)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'开始Whisper识别（device={device}）')
        model = whisper.load_model('base', device=device)
        result = model.transcribe(audio_path, language='zh', verbose=False, fp16=(device == 'cuda'))

        segments = []
        raw_text_parts = []
        for seg in result.get('segments', []):
            text = (seg.get('text') or '').strip()
            if not text:
                continue
            start = round(float(seg.get('start', 0.0)), 2)
            end = round(float(seg.get('end', start)), 2)
            if end <= start:
                end = round(start + 0.5, 2)
            segments.append({'start': start, 'end': end, 'text': text})
            raw_text_parts.append(text)

        if not segments:
            fallback_text = (result.get('text') or '').strip()
            if not fallback_text:
                raise ValueError('Whisper 未识别到有效字幕内容')
            segments = [{'start': 0.0, 'end': 1.0, 'text': fallback_text}]
            raw_text_parts = [fallback_text]

        with open(raw_segments_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        raw_asr_text = '\n'.join(raw_text_parts).strip()
        with open(raw_text_path, 'w', encoding='utf-8') as f:
            f.write(raw_asr_text)

        logger.info(f'✅ ASR提取完成：共{len(segments)}段，已保存 {raw_segments_path}')
        return raw_asr_text, raw_segments_path

    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


def _prepare_real_asr(video_id, video_path, work_dir):
    """生成真实 ASR，并调用 correct_asr.py 输出 corrected_asr.txt"""
    raw_asr_text, raw_segments_path = _generate_raw_asr_files(video_path, work_dir)
    logger.info('开始执行 [ASR矫正]')
    corrected_asr = run_correct_asr(str(video_id), raw_asr_text, VIDEO_DATA_DIR)

    corrected_asr_path = os.path.join(work_dir, 'corrected_asr.txt')
    if not corrected_asr or not os.path.exists(corrected_asr_path):
        raise FileNotFoundError(f'ASR矫正结果未生成：{corrected_asr_path}')

    if not os.path.exists(raw_segments_path):
        raise FileNotFoundError(f'原始ASR分段文件未生成：{raw_segments_path}')

    return corrected_asr_path, raw_segments_path, corrected_asr

# ==========================================
# AI 流水线后台线程
# ==========================================
def process_video_pipeline(video_id, video_path, user_id):
    """
    真正的全栈 AI 流水线：
    上传视频 -> 提取真实ASR -> ASR矫正 -> 知识点拆分与物理切割 -> 摘要 -> 知识图谱 -> 关系网 -> 出题 -> 标题摘要
    """
    logger.info(f"开始后台处理视频 ID {video_id}，路径: {video_path}")

    vid_str = str(video_id)
    work_dir = os.path.join(VIDEO_DATA_DIR, vid_str)
    os.makedirs(work_dir, exist_ok=True)

    try:
        # [步骤0：真实ASR提取 + 矫正]
        corrected_asr_path, raw_asr_path, corrected_asr = _prepare_real_asr(vid_str, video_path, work_dir)
        logger.info(f"ASR文件准备完成：corrected={corrected_asr_path} | raw={raw_asr_path}")

        # [步骤1：知识点拆分与物理切割]
        logger.info("开始执行 [知识点拆分与物理切割]")
        final_splits = run_knowledge_split(video_path, corrected_asr_path, vid_str)
        logger.info(f"知识点拆分完成：{len(final_splits) if isinstance(final_splits, list) else '未知'}段")

        # [步骤2：子视频多模态摘要]
        logger.info("开始执行 [子视频多模态摘要]")
        split_dir = os.path.join(work_dir, 'split_videos')
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"拆分后的子视频目录不存在：{split_dir}")
        run_video_summary(split_dir, VIDEO_DATA_DIR, vid_str)

        # [步骤3：知识图谱]
        logger.info("开始执行 [知识图谱生成]")
        generate_video_kg(vid_str)

        # [步骤4：关系网生成]
        logger.info("开始执行 [视频关系网生成]")
        custom_kg_path = os.path.join(work_dir, f"custom_kg_{vid_str}.json")
        if os.path.exists(custom_kg_path):
            generate_relation_network(vid_str, work_dir, custom_kg_path)
        else:
            logger.warning(f"未找到知识图谱文件，跳过关系网生成：{custom_kg_path}")

        # [步骤5：生成问题]
        logger.info("开始执行 [随堂问题生成]")
        generate_questions_for_knowledge_points(vid_str)

        # [步骤6：生成网页标题摘要]
        logger.info("开始执行 [总结标题摘要]")
        web_title, web_summary = run_generate_web_title_summary(video_id=vid_str, corrected_asr=corrected_asr)

        for v in USER_VIDEOS.get(user_id, []):
            if v['id'] == video_id:
                v['status'] = 'completed'
                v['title'] = web_title
                v['summary'] = web_summary
                break

        logger.info(f"✅ 视频 {video_id} AI 流水线全链条处理完毕！")

    except Exception as e:
        logger.error(f"❌ 视频 {video_id} AI 处理崩溃: {str(e)}", exc_info=True)
        for v in USER_VIDEOS.get(user_id, []):
            if v['id'] == video_id:
                v['status'] = 'failed'
                break

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global NEXT_VIDEO_ID
    if request.method == 'POST':
        video_file = request.files.get('video_file')
        if not video_file or not video_file.filename:
            flash('未选择任何文件', 'danger')
            return redirect(url_for('upload'))

        original_name = video_file.filename
        safe_name = secure_filename(original_name)
        unique_name = f"{int(time.time() * 1000)}_{safe_name}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)
        video_file.save(save_path)

        user_id = _current_user_id()
        if user_id not in USER_VIDEOS:
            USER_VIDEOS[user_id] = []

        current_video_id = NEXT_VIDEO_ID
        USER_VIDEOS[user_id].append({
            'id': current_video_id,
            'filename': original_name,
            'status': 'processing', 
            'stored_filename': unique_name,
            'title': '解析中...',
            'summary': 'AI正在深度解析视频，提取知识图谱与考点...'
        })
        NEXT_VIDEO_ID += 1

        # 启动后台AI线程
        threading.Thread(target=process_video_pipeline, args=(current_video_id, save_path, user_id), daemon=True).start()

        flash(f'视频上传成功，后台 AI 流水线已启动：{original_name}', 'success')
        return redirect(url_for('workspace'))
    return render_template('upload.html')

@app.route('/delete/<int:video_id>', methods=['POST'])
def delete_video(video_id):
    user_id = _current_user_id()
    videos = USER_VIDEOS.get(user_id, [])
    target = next((v for v in videos if v['id'] == video_id), None)
    if target:
        videos.remove(target)
        flash('学习记录已彻底删除', 'success')
    return redirect(url_for('workspace'))

# ==========================================
# 沉浸学习详情页
# ==========================================
@app.route('/video_detail/<int:video_id>')
def video_detail(video_id):
    user_id = _current_user_id()
    selected_video = next((v for v in USER_VIDEOS.get(user_id, []) if v['id'] == video_id), None)

    # 加载真实的 AI 分段
    video_segments = _load_video_segments(video_id)
    if not video_segments:
        video_segments = [
            {"title": "第一章：什么是微积分", "start_time": 2, "end_time": 10},
            {"title": "第二章：导数的应用", "start_time": 15, "end_time": 25}
        ]

    title = selected_video['title'] if selected_video and selected_video.get('title') else f"知识点视频 {video_id}"
    filepath = selected_video['stored_filename'] if selected_video else "sample.mp4"
    summary = selected_video['summary'] if selected_video and selected_video.get('summary') else "AI功能环境已就绪。"
    
    # 提取推荐问题
    recommended_questions = []
    for item in _extract_video_qa_items(video_id)[:4]:
        recommended_questions.append(item['question'])

    if not recommended_questions:
        recommended_questions = ["这段视频最重要的结论是什么？", "有什么容易错的地方？", "帮我出一道考题"]

    return render_template('video_detail.html', 
                           title=title, video_id=video_id, filepath=filepath,
                           summary=summary, knowledge_segments=video_segments, 
                           recommended_questions=recommended_questions)

@app.route('/video_insights/<int:video_id>')
def video_insights(video_id):
    student_id = _current_user_id()
    memory_key = f"{student_id}:{video_id}"
    memory = LEARNING_MEMORY.get(memory_key, { "attempts": 0, "correct": 0, "total_chars": 0, "history": [], "topic_mistakes": {} })
    profile = _build_student_profile(memory)
    return render_template('video_insights.html', title=f"视频 {video_id} 分析", video_id=video_id, insights={"profile": profile, "memory": memory})

# ==========================================
# AJAX 异步接口 (AI 互动核心)
# ==========================================
@app.route('/api/analyze_answer', methods=['POST'])
def analyze_answer():
    data = request.json
    question, user_answer, standard_answer = data.get("question", ""), data.get("user_answer", ""), data.get("standard_answer", "")
    video_id, segment_title = int(data.get("video_id", 0)), data.get("segment_title", "当前知识点")

    # 本地判分逻辑
    eval_result = _evaluate_answer(user_answer, standard_answer)

    # 记录到画像
    student_id = _current_user_id()
    memory_key = f"{student_id}:{video_id}"
    if memory_key not in LEARNING_MEMORY:
        LEARNING_MEMORY[memory_key] = { "attempts": 0, "correct": 0, "total_chars": 0, "history": [], "topic_mistakes": {} }
    
    mem = LEARNING_MEMORY[memory_key]
    mem["attempts"] += 1
    mem["correct"] += 1 if eval_result["is_correct"] else 0
    mem["total_chars"] += len(user_answer.strip())
    mem["history"].append({"segment_title": segment_title, "is_correct": eval_result["is_correct"], "score": eval_result["score"]})

    profile = _build_student_profile(mem)
    path = _build_path_recommendation(profile, mem, segment_title)

    return jsonify({
        "is_correct": eval_result["is_correct"], "standard_answer": standard_answer,
        "analysis": f"🤖 AI 分析：{eval_result['comment']}",
        "evaluation": eval_result, "student_profile": profile, "learning_path": path
    })

@app.route('/api/ask', methods=['POST'])
def ask_ai():
    data = request.json or {}
    question = (data.get('question') or '').strip()
    video_id = int(data.get('video_id', 0) or 0)

    if not question:
        return jsonify({"answer": "请先输入你的问题。"})

    try:
        result = run_answer_question(video_id, question)
        return jsonify({
            "answer": result.get("answer", "解析遇到了点小问题。"),
            "matched_subvideo": result.get("matched_subvideo", "")
        })
    except Exception as e:
        logger.error(f"问答调用失败: {str(e)}")
        return jsonify({"answer": f"抱歉，你的问题难倒了我，请尝试换种问法。({str(e)[:20]})"})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))
    app.run(host=host, port=port, debug=True)