# -*- coding: utf-8 -*-
from flask import Blueprint, render_template, request, jsonify, session, current_app
from utils.db import save_video_record
import uuid
import os

video_bp = Blueprint('video', __name__)

# 登录验证装饰器
def login_required(f):
    def wrap(*args, **kwargs):
        if 'username' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

# 主页面
@video_bp.route('/main')
@login_required
def main():
    return render_template('main.html', username=session['username'])
# 视频上传（核心：避免超时，模拟知识点）
@video_bp.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    if 'video' not in request.files:
        return jsonify({"status": "error", "msg": "未选择视频"})
    video_file = request.files['video']
    video_id = str(uuid.uuid4())[:8]
    video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{video_id}_{video_file.filename}")
    video_file.save(video_path)
    
    # 模拟知识点（避免真实处理超时）
    mock_kp = [{"knowledge_id": "kp1", "name": "核心知识点1", "start": 0, "end": 10}]
    save_video_record(video_id, video_file.filename, session['username'])
    
    return jsonify({
        "status": "success",
        "video_id": video_id,
        "video_url": f"/uploads/{video_id}_{video_file.filename}",
        "knowledge_points": mock_kp
    })

# 智能提问（模拟回答）
@video_bp.route('/ask_question', methods=['POST'])
@login_required
def ask_question():
    return jsonify({
        "status": "success",
        "answer": "这是智能回答内容",
        "related_knowledge": [{"name": "核心知识点1"}]
    })
