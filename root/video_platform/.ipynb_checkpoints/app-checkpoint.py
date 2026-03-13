# -*- coding: utf-8 -*-
import os
import sys
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
load_dotenv()

# 初始化Flask应用
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'video_platform_2026')  # 会话加密密钥
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB上传限制

# 创建必要目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'keyframes'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'split_videos'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'transcripts'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'knowledge_points'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'relation_networks'), exist_ok=True)

# 导入配置（创建config.py）
config_content = """# -*- coding: utf-8 -*-
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
"""
with open(os.path.join(os.path.dirname(__file__), 'config.py'), 'w', encoding='utf-8') as f:
    f.write(config_content)

# 模拟数据库（实际项目可替换为SQLite/MySQL）
user_db = {
    # 初始测试账号：admin / 123456
    "admin": "123456"
}

# 创建utils目录和db.py
os.makedirs(os.path.join(os.path.dirname(__file__), 'utils'), exist_ok=True)
db_content = """# -*- coding: utf-8 -*-
import json
import os
from datetime import datetime

def save_qa_record(video_id: str, username: str, question_type: str, question: str, answer: str, analysis: str = "", matched_sub_video: str = ""):
    \"\"\"保存问答记录\"\"\"
    record_dir = os.path.join(os.path.dirname(__file__), '../qa_records')
    os.makedirs(record_dir, exist_ok=True)
    record_path = os.path.join(record_dir, f"{video_id}_{username}.json")
    
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question_type": question_type,
        "question": question,
        "answer": answer,
        "analysis": analysis,
        "matched_sub_video": matched_sub_video
    }
    
    # 追加记录
    records = []
    if os.path.exists(record_path):
        with open(record_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    records.append(record)
    
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
"""
with open(os.path.join(os.path.dirname(__file__), 'utils/db.py'), 'w', encoding='utf-8') as f:
    f.write(db_content)

# 创建routes目录和video.py（核心接口）
os.makedirs(os.path.join(os.path.dirname(__file__), 'routes'), exist_ok=True)
video_route_content = """# -*- coding: utf-8 -*-
import os
import uuid
import json
from flask import Blueprint, request, jsonify, session, current_app
from pathlib import Path

# 导入核心工具类
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_process import VideoProcessor
from utils.relation_network import RelationNetworkAnalyzer
from utils.vlm_client import VLMClient

video_bp = Blueprint('video', __name__)

# 视频上传接口
@video_bp.route('/upload_video', methods=['POST'])
def upload_video():
    if "username" not in session:
        return jsonify({"status": "error", "msg": "请先登录"})
    
    if 'video' not in request.files:
        return jsonify({"status": "error", "msg": "未选择视频文件"})
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"status": "error", "msg": "视频文件名为空"})
    
    # 生成唯一视频ID
    video_id = str(uuid.uuid4())[:8]
    video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{video_id}_{video_file.filename}")
    
    # 保存视频文件
    try:
        video_file.save(video_path)
    except Exception as e:
        return jsonify({"status": "error", "msg": f"保存视频失败：{str(e)[:50]}"})
    
    # 处理视频（核心逻辑）
    try:
        processor = VideoProcessor(video_path, video_id)
        # 1. ASR转写
        asr_segments, full_asr = processor.transcribe_audio()
        # 2. 提取多模态特征
        frame_features, ocr_results = processor.extract_multi_modal_features(asr_segments)
        # 3. 提取知识点
        knowledge_points = processor.extract_knowledge_points(asr_segments, frame_features)
        # 4. 切割视频
        sub_videos = processor.split_video(knowledge_points)
        # 5. 生成关系网络
        rna = RelationNetworkAnalyzer(video_id)
        relation_graph = rna.build_knowledge_graph(knowledge_points, sub_videos)
        # 6. 生成自动提问
        vlm_client = VLMClient()
        auto_questions = vlm_client.generate_auto_questions(sub_videos, knowledge_points)
        
        return jsonify({
            "status": "success",
            "msg": "视频处理完成",
            "video_id": video_id,
            "sub_videos": sub_videos,
            "auto_questions": auto_questions,
            "knowledge_points": knowledge_points
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": f"处理视频失败：{str(e)[:80]}"})

# 智能提问接口
@video_bp.route('/ask_question', methods=['POST'])
def ask_question():
    if "username" not in session:
        return jsonify({"status": "error", "msg": "请先登录"})
    
    video_id = request.form.get('video_id')
    question = request.form.get('question')
    
    if not video_id or not question:
        return jsonify({"status": "error", "msg": "视频ID和问题不能为空"})
    
    # 调用VLM生成回答
    try:
        vlm_client = VLMClient()
        # 加载知识点和子视频
        knowledge_path = os.path.join(current_app.config['KNOWLEDGE_FOLDER'], video_id, 'knowledge_points.json')
        split_path = os.path.join(current_app.config['SPLIT_VIDEO_FOLDER'], video_id, 'split_videos.json')
        
        knowledge_points = []
        if os.path.exists(knowledge_path):
            with open(knowledge_path, "r", encoding="utf-8") as f:
                knowledge_points = json.load(f)
        
        sub_videos = []
        if os.path.exists(split_path):
            with open(split_path, "r", encoding="utf-8") as f:
                sub_videos = json.load(f)
        
        # 生成回答
        answer, matched_sub, related_knowledge = vlm_client.answer_question(question, knowledge_points, sub_videos)
        
        # 保存问答记录
        from utils.db import save_qa_record
        save_qa_record(
            video_id=video_id,
            username=session['username'],
            question_type="user_ask",
            question=question,
            answer=answer,
            matched_sub_video=matched_sub.get('name', '') if matched_sub else ''
        )
        
        return jsonify({
            "status": "success",
            "answer": answer,
            "matched_sub_video": matched_sub or {},
            "related_knowledge": related_knowledge
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": f"提问失败：{str(e)[:80]}"})

# 答案解析接口
@video_bp.route('/analyze_answer', methods=['POST'])
def analyze_answer():
    if "username" not in session:
        return jsonify({"status": "error", "msg": "请先登录"})
    
    video_id = request.form.get('video_id')
    sub_video_id = request.form.get('sub_video_id')
    question = request.form.get('question')
    user_answer = request.form.get('user_answer')
    
    if not all([video_id, sub_video_id, question, user_answer]):
        return jsonify({"status": "error", "msg": "参数不完整"})
    
    # 调用VLM解析答案
    try:
        vlm_client = VLMClient()
        analysis_result = vlm_client.analyze_user_answer(question, user_answer, video_id, sub_video_id)
        
        # 保存记录
        from utils.db import save_qa_record
        save_qa_record(
            video_id=video_id,
            username=session['username'],
            question_type="auto_question",
            question=question,
            answer=user_answer,
            analysis=analysis_result['analysis']
        )
        
        return jsonify({
            "status": "success",
            "correctness": analysis_result['correctness'],
            "analysis": analysis_result['analysis'],
            "true_answer": analysis_result['true_answer'],
            "suggestion": analysis_result['suggestion']
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": f"解析答案失败：{str(e)[:80]}"})
"""
with open(os.path.join(os.path.dirname(__file__), 'routes/video.py'), 'w', encoding='utf-8') as f:
    f.write(video_route_content)

# 注册蓝图
from routes.video import video_bp
app.register_blueprint(video_bp)

# 首页/登录/注册路由
@app.route('/')
def index():
    if "username" in session:
        return redirect(url_for('main'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in user_db and user_db[username] == password:
            session['username'] = username
            return redirect(url_for('main'))
        else:
            return render_template('login.html', alert_msg='用户名或密码错误', alert_type='error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_pwd = request.form.get('confirm_pwd')
        
        if username in user_db:
            return render_template('register.html', alert_msg='用户名已存在', alert_type='error')
        if password != confirm_pwd:
            return render_template('register.html', alert_msg='两次密码不一致', alert_type='error')
        
        # 注册成功
        user_db[username] = password
        return render_template('login.html', alert_msg='注册成功，请登录', alert_type='success')
    
    return render_template('register.html')

@app.route('/main')
def main():
    if "username" not in session:
        return redirect(url_for('login'))
    return render_template('main.html', username=session['username'])

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# 启动服务器
if __name__ == '__main__':
    # 监听所有IP，端口6006（云服务器需要开放该端口）
    app.run(host='0.0.0.0', port=6006, debug=True)
