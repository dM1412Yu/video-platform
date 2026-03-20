from flask import Flask, render_template, request, redirect, url_for, flash, abort, jsonify, send_from_directory, session
import os
import re
import time
import json
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key_here' # 用于安全加密提示信息

# 轻量内存态学习档案（演示用）；生产环境建议迁移到数据库
LEARNING_MEMORY = {}
USER_VIDEOS = {}
NEXT_VIDEO_ID = 100

UPLOAD_DIR = os.path.join(app.root_path, 'static', 'uploads')
VIDEO_DATA_DIR = os.path.join(app.root_path, 'static', 'video_data')
ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


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

    # 兼容你原有的判定逻辑：包含标准答案前 3 个字符也算命中
    prefix_hit = bool(
        standard_answer and len(standard_answer) >= 3 and standard_answer[:3].lower() in (user_answer or "").lower()
    )

    is_correct = coverage >= 0.45 or prefix_hit
    score = min(100, int(coverage * 100) + (10 if prefix_hit else 0))

    if score >= 80:
        level = "优秀"
        comment = "回答结构清晰，核心概念覆盖完整。"
    elif score >= 55:
        level = "良好"
        comment = "关键点基本到位，再补充定义或应用场景会更好。"
    else:
        level = "待提升"
        comment = "已开始思考，但核心知识点覆盖不足，建议回看本节重点。"

    strengths = []
    weaknesses = []

    if score >= 60:
        strengths.append("抓住了问题主干")
    if len((user_answer or "").strip()) >= 25:
        strengths.append("回答较完整，表达有延展")
    if not strengths:
        strengths.append("有主动作答意识")

    if score < 55:
        weaknesses.append("关键术语覆盖不足")
    if len((user_answer or "").strip()) < 12:
        weaknesses.append("回答偏短，论证不充分")
    if not weaknesses:
        weaknesses.append("可增加例子让答案更有说服力")

    return {
        "is_correct": is_correct,
        "score": score,
        "level": level,
        "comment": comment,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }


def _build_student_profile(memory):
    attempts = memory["attempts"]
    correct = memory["correct"]
    total_chars = memory["total_chars"]
    accuracy = (correct / attempts) if attempts else 0.0
    avg_len = (total_chars / attempts) if attempts else 0.0

    if accuracy >= 0.8:
        mastery_level = "高"
        difficulty = "挑战"
    elif accuracy >= 0.55:
        mastery_level = "中"
        difficulty = "进阶"
    else:
        mastery_level = "基础"
        difficulty = "基础"

    if avg_len >= 35:
        learning_style = "解释型"
    elif avg_len >= 18:
        learning_style = "均衡型"
    else:
        learning_style = "速答型"

    confidence = min(100, int(accuracy * 100 * 0.7 + min(avg_len, 40) * 0.8))

    return {
        "mastery_level": mastery_level,
        "learning_style": learning_style,
        "confidence": confidence,
        "recommended_difficulty": difficulty,
        "accuracy": round(accuracy * 100, 1),
    }


def _build_path_recommendation(profile, memory, segment_title):
    segment_title = segment_title or "当前知识点"
    recent = memory["history"][-3:]
    recent_wrong = sum(1 for item in recent if not item["is_correct"])

    if recent_wrong >= 2:
        recommendation = f"建议先返回前面重看「{segment_title}」并完成1道基础题，再继续后续内容。"
        action = "回看复盘"
    elif profile["recommended_difficulty"] == "挑战":
        recommendation = f"你可以深入学习「{segment_title}」的拓展应用，并尝试跨章节综合题。"
        action = "深入学习"
    elif profile["recommended_difficulty"] == "进阶":
        recommendation = f"建议在当前章节继续进阶训练；若遇卡点，再回看「{segment_title}」关键定义。"
        action = "稳步进阶"
    else:
        recommendation = f"建议先学习前置基础（定义、符号和例题），再回到「{segment_title}」做一次复答。"
        action = "补齐前置"

    return {"action": action, "recommendation": recommendation}


def _extract_video_qa_items(video_id):
    qa_path = os.path.join(VIDEO_DATA_DIR, str(video_id), 'knowledge_questions.json')
    if not os.path.exists(qa_path):
        return []

    try:
        with open(qa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []

    items = []
    for block in data if isinstance(data, list) else []:
        title = block.get('title', '未命名知识点')
        for qa in block.get('questions', []):
            q = (qa.get('question') or '').strip()
            a = (qa.get('answer') or '').strip()
            if q and a:
                items.append({'title': title, 'question': q, 'answer': a})
    return items


def _load_video_segments(video_id):
    qa_path = os.path.join(VIDEO_DATA_DIR, str(video_id), 'knowledge_questions.json')
    if not os.path.exists(qa_path):
        return []

    try:
        with open(qa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []

    segments = []
    for idx, block in enumerate(data if isinstance(data, list) else []):
        title = (block.get('title') or '').strip()
        if not title:
            continue
        # 没有真实时间戳时，使用可视化演示的等距时间片
        start = 2 + idx * 60
        end = start + 45
        segments.append({"title": title, "start_time": start, "end_time": end})
    return segments


def _find_best_qa_answer(video_id, user_question):
    q_tokens = set(_tokenize_text(user_question))
    if not q_tokens:
        return None

    best_item = None
    best_score = 0.0
    for item in _extract_video_qa_items(video_id):
        item_tokens = set(_tokenize_text(item['question'] + ' ' + item['answer'] + ' ' + item['title']))
        if not item_tokens:
            continue
        overlap = len(q_tokens & item_tokens)
        score = overlap / max(1, len(q_tokens))
        if score > best_score:
            best_score = score
            best_item = item

    if best_item and best_score >= 0.2:
        return {
            'title': best_item['title'],
            'question': best_item['question'],
            'answer': best_item['answer'],
            'score': round(best_score, 3),
        }
    return None


def _build_recommended_questions(video_id, knowledge_segments, video_title=None):
    recommended = []
    seen = set()

    # 1) 优先使用当前视频题库中的真实问题
    for item in _extract_video_qa_items(video_id):
        q = (item.get('question') or '').strip()
        if not q:
            continue
        if q in seen:
            continue
        seen.add(q)
        recommended.append(q)
        if len(recommended) >= 4:
            return recommended

    # 2) 若题库不足，按当前视频知识点自动补齐
    templates = [
        "{title} 的核心概念是什么？",
        "{title} 常见易错点有哪些？",
        "如何用一道例题理解 {title}？",
        "{title} 和前置知识有什么关系？",
    ]

    for seg in knowledge_segments or []:
        title = (seg.get('title') or '').strip()
        if not title:
            continue
        for tpl in templates:
            q = tpl.format(title=title)
            if q in seen:
                continue
            seen.add(q)
            recommended.append(q)
            if len(recommended) >= 4:
                return recommended

    # 3) 最终兜底
    fallback = [
        f"这条视频《{video_title or '当前内容'}》最重要的三个知识点是什么？",
        "这一节最重要的三个知识点是什么？",
        "如果我要快速复习，这节课应该按什么顺序学？",
        "这节内容在考试中最常见的题型有哪些？",
        "我应该先巩固哪部分前置知识？",
    ]
    for q in fallback:
        if q not in seen:
            recommended.append(q)
        if len(recommended) >= 4:
            break

    return recommended[:4]


def _score_to_level(score):
    if score >= 80:
        return "优秀"
    if score >= 55:
        return "良好"
    return "待提升"


def _build_insight_view_data(video_id):
    student_id = _current_user_id()
    memory_key = f"{student_id}:{video_id}"
    memory = LEARNING_MEMORY.get(memory_key, {
        "attempts": 0,
        "correct": 0,
        "total_chars": 0,
        "history": [],
        "topic_mistakes": {}
    })

    profile = _build_student_profile(memory)
    recent = memory["history"][-1] if memory["history"] else None
    recent_score = recent["score"] if recent else 0

    if recent:
        evaluation_comment = "最近一次答题已纳入画像分析，可继续提问进行针对性提升。"
    else:
        evaluation_comment = "还没有答题记录，先完成一次随堂测验后这里会自动更新。"

    sorted_mistakes = sorted(memory["topic_mistakes"].items(), key=lambda x: x[1], reverse=True)
    weak_topics = [name for name, _ in sorted_mistakes[:3]]
    path = _build_path_recommendation(profile, memory, recent["segment_title"] if recent else "当前知识点")

    recent_records = []
    for item in memory["history"][-5:]:
        recent_records.append({
            "segment_title": item.get("segment_title", "知识点"),
            "score": item.get("score", 0),
            "status": "正确" if item.get("is_correct") else "待提升"
        })

    return {
        "evaluation": {
            "score": recent_score,
            "level": _score_to_level(recent_score),
            "comment": evaluation_comment,
            "strengths": ["主动完成答题反馈"] if recent else ["尚无记录"],
            "weaknesses": ["继续通过随堂测验积累数据"] if not recent else ["针对薄弱点复习"]
        },
        "profile": profile,
        "memory": {
            "attempts": memory["attempts"],
            "accuracy": profile["accuracy"],
            "weak_topics": weak_topics,
            "recent_records": recent_records
        },
        "path": path
    }


def _build_general_tutor_answer(question):
    q = (question or '').strip()
    if not q:
        return "请先输入一个具体问题，例如：什么是导数？导数在本节课中有什么作用？"

    if any(k in q for k in ['导数', '变化率', '斜率']):
        return (
            "导数可以理解为“瞬时变化率”，它描述了函数在某一点附近变化有多快。\n"
            "在本节内容里，你可以先把导数记成两件事：\n"
            "1. 物理意义：速度是位移对时间的导数。\n"
            "2. 几何意义：曲线在该点切线的斜率。\n"
            "如果你愿意，我可以继续给你一个 30 秒内能看懂的导数计算例子。"
        )

    if any(k in q for k in ['积分', '面积', '累积']):
        return (
            "积分可以理解为“累积量”，最直观的是曲线与坐标轴围成区域的面积。\n"
            "你可以先记住：导数解决“变化快慢”，积分解决“累计总量”。\n"
            "如果你想，我可以下一步帮你把“定积分与原函数”的关系画成一条学习链。"
        )

    if any(k in q for k in ['极限', '趋近']):
        return (
            "极限描述的是“无限接近时的结果”，它是导数和积分的基础语言。\n"
            "理解极限时可抓住一句话：变量可以无限逼近某点，但不一定真的取到该点。\n"
            "你可以继续问我一个具体极限式子，我按步骤带你算。"
        )

    return (
        f"我先直接回答你的问题：{q}\n"
        "当前页面是演示环境，我会基于本节微积分核心概念给你结构化解答。\n"
        "建议你把问题进一步具体化为“定义是什么 / 为什么成立 / 怎么计算 / 有什么应用”，"
        "我就能给你更精准的步骤答案。"
    )


def _build_video_context(video_id, max_items=6):
    items = _extract_video_qa_items(video_id)[:max_items]
    if not items:
        return "当前视频暂无结构化题库上下文。"

    lines = []
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. 知识点：{item['title']}\\n   问：{item['question']}\\n   答：{item['answer']}")
    return "\\n".join(lines)


def _call_remote_llm(question, video_id):
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        return None

    model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
    endpoint = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    context = _build_video_context(video_id)

    payload = {
        "model": model,
        "temperature": 0.3,
        "messages": [
            {
                "role": "system",
                "content": "你是视频学习助教。回答要求：简洁、准确、可执行；优先结合提供的视频上下文。"
            },
            {
                "role": "user",
                "content": f"视频上下文：\\n{context}\\n\\n学生问题：{question}\\n\\n请给出：1) 直接答案 2) 关键点 3) 下一步建议"
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip() or None
    except Exception:
        return None

# ==========================================
# 0. 静态资源路由 (适配前端的视频和JSON加载)
# ==========================================
@app.route('/uploads/<filename>')
def uploads(filename):
    """前端视频播放器获取视频文件的接口"""
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/video_data/<int:video_id>/<filename>')
def video_data(video_id, filename):
    """前端加载知识点题库 JSON 的接口"""
    per_video_dir = os.path.join(VIDEO_DATA_DIR, str(video_id))
    if os.path.exists(os.path.join(per_video_dir, filename)):
        return send_from_directory(per_video_dir, filename)
    return send_from_directory(VIDEO_DATA_DIR, filename)

# ==========================================
# 1. 官网首页 (Landing Page)
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

# ==========================================
# 2. 账号体系
# ==========================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password: 
            flash('登录成功，欢迎回来！', 'success')
            return redirect(url_for('workspace'))
        else:
            flash('账号或密码错误，请重试', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        flash('账号创建成功，请登录！', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

# ==========================================
# 3. 学习工作台
# ==========================================
@app.route('/workspace')
def workspace():
    # 无真实上传数据时，展示默认演示数据
    demo_videos = [
        {'id': 1, 'filename': '高等数学-微积分基础.mp4', 'status': 'completed'},
        {'id': 2, 'filename': 'Python从入门到精通.mp4', 'status': 'processing'},
        {'id': 3, 'filename': '损坏的视频文件.avi', 'status': 'failed'}
    ]
    user_id = _current_user_id()
    uploaded_videos = USER_VIDEOS.get(user_id, [])
    videos = list(reversed(uploaded_videos)) if uploaded_videos else demo_videos

    # 添加学生概览数据
    student_overview = {
        'total_videos': len(videos),
        'completed': sum(1 for v in videos if v['status'] == 'completed'),
        'processing': sum(1 for v in videos if v['status'] == 'processing'),
        'learning_hours': 12.5
    }
    return render_template('workspace.html', videos=videos, student_overview=student_overview)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global NEXT_VIDEO_ID

    if request.method == 'POST':
        video_file = request.files.get('video_file')
        if not video_file or not video_file.filename:
            flash('未选择任何文件，上传失败', 'danger')
            return redirect(url_for('upload'))

        original_name = video_file.filename
        if not _is_allowed_video(original_name):
            flash('仅支持 MP4 / AVI / MOV / MKV / WEBM 格式', 'danger')
            return redirect(url_for('upload'))

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        safe_name = secure_filename(original_name)
        unique_name = f"{int(time.time() * 1000)}_{safe_name}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)
        video_file.save(save_path)

        user_id = _current_user_id()
        if user_id not in USER_VIDEOS:
            USER_VIDEOS[user_id] = []

        USER_VIDEOS[user_id].append({
            'id': NEXT_VIDEO_ID,
            'filename': original_name,
            'status': 'completed',
            'stored_filename': unique_name,
        })
        NEXT_VIDEO_ID += 1

        flash(f'视频上传成功：{original_name}', 'success')
        return redirect(url_for('workspace'))
    return render_template('upload.html')

@app.route('/delete/<int:video_id>', methods=['POST'])
def delete_video(video_id):
    user_id = _current_user_id()
    videos = USER_VIDEOS.get(user_id, [])
    target = next((v for v in videos if v['id'] == video_id), None)

    if target:
        videos.remove(target)
        stored_name = target.get('stored_filename')
        if stored_name:
            file_path = os.path.join(UPLOAD_DIR, stored_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        flash('学习记录已彻底删除', 'success')
    else:
        flash('未找到对应学习记录', 'danger')
    return redirect(url_for('workspace'))

# ==========================================
# 4. 视频沉浸学习详情页
# ==========================================
@app.route('/video_detail/<int:video_id>')
def video_detail(video_id):
    user_id = _current_user_id()
    uploaded_videos = USER_VIDEOS.get(user_id, [])
    selected_video = next((v for v in uploaded_videos if v['id'] == video_id), None)

    # 默认演示知识点切片
    default_segments = [
        {"title": "第一章：什么是微积分", "start_time": 2, "end_time": 10},
        {"title": "第二章：导数的应用", "start_time": 15, "end_time": 25}
    ]
    video_segments = _load_video_segments(video_id)
    mock_segments = video_segments or default_segments

    if selected_video:
        title, _ = os.path.splitext(selected_video['filename'])
        filepath = selected_video.get('stored_filename', 'sample.mp4')
        summary = "这是你上传的视频。AI 功能演示环境已加载，后续可接入真实解析链路。"
    else:
        title = "高等数学-微积分基础"
        filepath = "sample.mp4"
        summary = "本节课主要讲解了微积分的核心概念，AI已为您提取出关键知识点，并在视频关键处为您准备了随堂测验。"
        if video_segments:
            title = f"视频 {video_id} 学习内容"
            summary = "已根据该视频的题库信息自动生成知识点与推荐问题。"

    recommended_questions = _build_recommended_questions(video_id, mock_segments, title)
    
    return render_template('video_detail.html', 
                           title=title,
                           video_id=video_id,
                           filepath=filepath,
                           summary=summary,
                           knowledge_segments=mock_segments,
                           recommended_questions=recommended_questions)


@app.route('/video_insights/<int:video_id>')
def video_insights(video_id):
    user_id = _current_user_id()
    uploaded_videos = USER_VIDEOS.get(user_id, [])
    selected_video = next((v for v in uploaded_videos if v['id'] == video_id), None)

    if selected_video:
        title, _ = os.path.splitext(selected_video['filename'])
    else:
        title = f"视频 {video_id} 学习分析"

    insights = _build_insight_view_data(video_id)
    return render_template('video_insights.html', title=title, video_id=video_id, insights=insights)


@app.route('/api/video_insights/<int:video_id>', methods=['GET'])
def api_video_insights(video_id):
    return jsonify(_build_insight_view_data(video_id))

# ==========================================
# 5. 前端 AJAX 异步接口 (AI 互动)
# ==========================================
@app.route('/api/analyze_answer', methods=['POST'])
def analyze_answer():
    data = request.json
    question = data.get("question", "")
    user_answer = data.get("user_answer", "")
    standard_answer = data.get("standard_answer", "未知标准答案")
    video_id = int(data.get("video_id", 0) or 0)
    segment_title = data.get("segment_title", "当前知识点")

    eval_result = _evaluate_answer(user_answer, standard_answer)

    student_id = session.get('user') or 'guest'
    memory_key = f"{student_id}:{video_id}"
    if memory_key not in LEARNING_MEMORY:
        LEARNING_MEMORY[memory_key] = {
            "attempts": 0,
            "correct": 0,
            "total_chars": 0,
            "history": [],
            "topic_mistakes": {}
        }

    memory = LEARNING_MEMORY[memory_key]
    memory["attempts"] += 1
    memory["correct"] += 1 if eval_result["is_correct"] else 0
    memory["total_chars"] += len(user_answer.strip())
    memory["history"].append({
        "question": question,
        "segment_title": segment_title,
        "is_correct": eval_result["is_correct"],
        "score": eval_result["score"]
    })
    if not eval_result["is_correct"]:
        memory["topic_mistakes"][segment_title] = memory["topic_mistakes"].get(segment_title, 0) + 1

    profile = _build_student_profile(memory)
    path = _build_path_recommendation(profile, memory, segment_title)

    sorted_mistakes = sorted(memory["topic_mistakes"].items(), key=lambda x: x[1], reverse=True)
    weak_topics = [name for name, _ in sorted_mistakes[:3]]
    recent_memory = memory["history"][-5:]

    return jsonify({
        "is_correct": eval_result["is_correct"],
        "standard_answer": standard_answer,
        "analysis": f"🤖 AI 分析：{eval_result['comment']}",
        "evaluation": {
            "score": eval_result["score"],
            "level": eval_result["level"],
            "comment": eval_result["comment"],
            "strengths": eval_result["strengths"],
            "weaknesses": eval_result["weaknesses"]
        },
        "student_profile": profile,
        "student_memory": {
            "attempts": memory["attempts"],
            "accuracy": profile["accuracy"],
            "weak_topics": weak_topics,
            "recent_records": recent_memory
        },
        "learning_path": path
    })

@app.route('/api/ask', methods=['POST'])
def ask_ai():
    data = request.json or {}
    question = (data.get('question') or '').strip()
    video_id = int(data.get('video_id', 0) or 0)

    if not question:
        return jsonify({"answer": "请先输入你的问题，我会结合当前视频内容回答你。"})

    matched = _find_best_qa_answer(video_id, question)
    if matched:
        answer = (
            f"基于当前视频知识点「{matched['title']}」，我先直接回答你：\n"
            f"{matched['answer']}\n\n"
            f"如果你还想延伸，我建议继续追问：\n"
            f"1. 这个概念为什么成立？\n"
            f"2. 在题目里怎么识别它？\n"
            f"3. 常见易错点是什么？"
        )
        return jsonify({"answer": answer})

    llm_answer = _call_remote_llm(question, video_id)
    if llm_answer:
        return jsonify({"answer": llm_answer})

    return jsonify({"answer": _build_general_tutor_answer(question)})

# ==========================================
# 6. 全局 404 错误捕获
# ==========================================
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    # 自动创建所需要的静态文件夹
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(VIDEO_DATA_DIR, exist_ok=True)

    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('FLASK_DEBUG', '0') in ('1', 'true', 'True')

    # 启动服务器
    app.run(host=host, port=port, debug=debug)