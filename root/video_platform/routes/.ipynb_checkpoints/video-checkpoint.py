# -*- coding: utf-8 -*-
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
