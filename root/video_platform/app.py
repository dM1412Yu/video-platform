from flask import Flask, render_template, request, redirect, url_for, flash, abort, jsonify, send_from_directory
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here' # 用于安全加密提示信息

# ==========================================
# 0. 静态资源路由 (适配前端的视频和JSON加载)
# ==========================================
@app.route('/uploads/<filename>')
def uploads(filename):
    """前端视频播放器获取视频文件的接口"""
    return send_from_directory('static/uploads', filename)

@app.route('/video_data/<int:video_id>/<filename>')
def video_data(video_id, filename):
    """前端加载知识点题库 JSON 的接口"""
    # 这里为了演示，我们统一从 static/video_data 文件夹读取
    return send_from_directory('static/video_data', filename)

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
    # 模拟数据库里的视频记录
    mock_videos = [
        {'id': 1, 'filename': '高等数学-微积分基础.mp4', 'status': 'completed'},
        {'id': 2, 'filename': 'Python从入门到精通.mp4', 'status': 'processing'},
        {'id': 3, 'filename': '损坏的视频文件.avi', 'status': 'failed'}
    ]
    return render_template('workspace.html', videos=mock_videos)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        video_file = request.files.get('video_file')
        if video_file:
            flash('视频上传成功，AI 正在为您全力解析中！', 'success')
            return redirect(url_for('workspace'))
        else:
            flash('未选择任何文件，上传失败', 'danger')
    return render_template('upload.html')

@app.route('/delete/<int:video_id>', methods=['POST'])
def delete_video(video_id):
    flash('学习记录已彻底删除', 'success')
    return redirect(url_for('workspace'))

# ==========================================
# 4. 视频沉浸学习详情页
# ==========================================
@app.route('/video_detail/<int:video_id>')
def video_detail(video_id):
    # 模拟知识点切片数据
    mock_segments = [
        {"title": "第一章：什么是微积分", "start_time": 2, "end_time": 10},
        {"title": "第二章：导数的应用", "start_time": 15, "end_time": 25}
    ]
    
    return render_template('video_detail.html', 
                           title="高等数学-微积分基础",
                           video_id=video_id,
                           filepath="sample.mp4", # 需在 static/uploads 放入该视频
                           summary="本节课主要讲解了微积分的核心概念，AI已为您提取出关键知识点，并在视频关键处为您准备了随堂测验。",
                           knowledge_segments=mock_segments)

# ==========================================
# 5. 前端 AJAX 异步接口 (AI 互动)
# ==========================================
@app.route('/api/analyze_answer', methods=['POST'])
def analyze_answer():
    data = request.json
    return jsonify({
        "is_correct": True,
        "standard_answer": data.get("standard_answer", "未知标准答案"),
        "analysis": "🤖 AI 分析：你的回答非常准确，完美命中了核心概念！继续保持。"
    })

@app.route('/api/ask', methods=['POST'])
def ask_ai():
    return jsonify({"answer": "🤖 AI伴学助教：这是一个很好的问题！结合当前视频内容，我的理解是：知识的本质在于不断的提问与思考。"})

# ==========================================
# 6. 全局 404 错误捕获
# ==========================================
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    # 自动创建所需要的静态文件夹
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/video_data', exist_ok=True)
    
    # 启动服务器
    app.run(debug=True, port=5000)