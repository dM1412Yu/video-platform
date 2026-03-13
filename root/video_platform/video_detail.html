<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - 视频详情</title>
    <style>
        /* 全局样式 */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: "Microsoft YaHei", sans-serif;
        }
        body {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            background-color: #f5f7fa;
        }
        .container {
            display: flex;
            flex: 1;
            padding: 20px;
            gap: 20px;
        }

        /* 左侧知识点列表 */
        .sidebar {
            width: 300px;
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            height: calc(100vh - 80px);
            overflow-y: auto;
        }
        .sidebar h3 {
            margin-bottom: 15px;
            color: #333;
            font-size: 18px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .segment {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #eee;
        }
        .segment:hover {
            background-color: #f0f7ff;
            border-color: #409eff;
        }
        .segment.active {
            background-color: #e6f7ff;
            border-color: #1890ff;
        }
        .segment p {
            color: #666;
            font-size: 12px;
            margin-top: 5px;
        }
        .segment h4 {
            color: #333;
            font-size: 14px;
            font-weight: normal;
        }

        /* 右侧主内容 */
        .main-content {
            flex: 1;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .video-container {
            width: 100%;
            aspect-ratio: 16/9;
            background-color: #000;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        #video-player {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .video-info h2 {
            font-size: 22px;
            color: #333;
            margin-bottom: 10px;
        }
        .video-info .summary {
            color: #666;
            line-height: 1.6;
            margin: 15px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 6px;
        }
        .asr-content {
            margin-top: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 6px;
            color: #333;
            line-height: 1.8;
            font-size: 14px;
            max-height: 300px;
            overflow-y: auto;
        }

        /* 弹窗样式 */
        .popup-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }
        .popup-content {
            width: 500px;
            background: white;
            border-radius: 8px;
            padding: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }
        .popup-title {
            font-size: 18px;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .popup-input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            margin: 15px 0;
            font-size: 14px;
            resize: none;
            min-height: 80px;
        }
        .popup-buttons {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-top: 20px;
        }
        .btn {
            padding: 8px 16px;
            border-radius: 6px;
            border: none;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .btn-primary {
            background-color: #409eff;
            color: white;
        }
        .btn-primary:hover {
            background-color: #66b1ff;
        }
        .btn-cancel {
            background-color: #f5f5f5;
            color: #666;
        }
        .btn-cancel:hover {
            background-color: #eee;
        }
        .feedback {
            margin: 10px 0;
            padding: 10px;
            border-radius: 6px;
            font-size: 14px;
        }
        .feedback.correct {
            background-color: #f0f9ff;
            color: #52c41a;
            border: 1px solid #b7eb8f;
        }
        .feedback.wrong {
            background-color: #fff2f0;
            color: #f5222d;
            border: 1px solid #ffccc7;
        }
        .answer-analysis {
            margin-top: 15px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 6px;
            display: none;
        }
        .answer-analysis h4 {
            font-size: 14px;
            color: #333;
            margin-bottom: 8px;
        }
        .answer-analysis p {
            font-size: 14px;
            color: #666;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <!-- 顶部导航（保留原有结构） -->
    <header style="background: #1890ff; color: white; padding: 10px 20px; font-size: 20px; font-weight: bold;">
        视频知识点学习系统
    </header>

    <div class="container">
        <!-- 左侧知识点列表 -->
        <div class="sidebar">
            <h3>知识点列表</h3>
            <div id="segments-container">
                <!-- 知识点会通过JS动态渲染 -->
                {% for seg in segments %}
                <div class="segment" 
                     data-start="{{ seg.start_time }}" 
                     data-end="{{ seg.end_time }}"
                     data-index="{{ loop.index0 }}">
                    <h4>{{ seg.title or seg.knowledge_point }}</h4>
                    <p>{{ "%d:%02d ~ %d:%02d"|format(seg.start_time|int//60, seg.start_time|int%60, seg.end_time|int//60, seg.end_time|int%60) }}</p>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- 右侧主内容 -->
        <div class="main-content">
            <div class="video-container">
                <video id="video-player" controls>
                    <source src="{{ url_for('uploads', filename=filepath) }}" type="video/mp4">
                    你的浏览器不支持HTML5视频播放
                </video>
            </div>

            <div class="video-info">
                <h2>{{ title }}</h2>
                <div class="summary">
                    <strong>视频摘要：</strong> {{ summary }}
                </div>
                <div class="asr-content">
                    <strong>音频文本：</strong> {{ corrected_asr }}
                </div>
            </div>
        </div>
    </div>

    <!-- 答题弹窗 -->
    <div class="popup-overlay" id="quiz-popup">
        <div class="popup-content">
            <h3 class="popup-title" id="quiz-title">请回答以下问题</h3>
            <textarea class="popup-input" id="user-answer" placeholder="请输入你的答案..."></textarea>
            <div class="feedback" id="feedback"></div>
            <div class="answer-analysis" id="answer-analysis">
                <h4>标准答案：</h4>
                <p id="standard-answer"></p>
                <h4 style="margin-top: 10px;">解析：</h4>
                <p id="answer-analysis-text"></p>
            </div>
            <div class="popup-buttons">
                <button class="btn btn-cancel" onclick="closePopup()">关闭</button>
                <button class="btn btn-primary" onclick="submitAnswer()">提交答案</button>
            </div>
        </div>
    </div>

    <script>
        // 全局变量
        const video = document.getElementById('video-player');
        const popup = document.getElementById('quiz-popup');
        const quizTitle = document.getElementById('quiz-title');
        const userAnswer = document.getElementById('user-answer');
        const feedback = document.getElementById('feedback');
        const standardAnswer = document.getElementById('standard-answer');
        const answerAnalysis = document.getElementById('answer-analysis');
        const answerAnalysisText = document.getElementById('answer-analysis-text');
        
        let currentSeg = null;       // 当前选中的知识点
        let currentQuestionIdx = 0; // 当前题目索引（0=引导题，1=考查题）
        let answered = false;        // 是否已提交答案

        // ========== 核心：修正时间显示 + 读取现成的knowledge_questions.json ==========
        window.onload = async function() {
            try {
                // 1. 读取现成的knowledge_questions.json（优先级最高）
                const res = await fetch(`/video_data/{{ video_id }}/knowledge_questions.json`);
                if (res.ok) {
                    const questionsData = await res.json();
                    console.log("✅ 读取现成的knowledge_questions.json成功：", questionsData);
                    
                    // 2. 重新渲染左侧知识点时间（覆盖模板渲染的时间）
                    const segmentElements = document.querySelectorAll('.segment');
                    questionsData.forEach((data, idx) => {
                        if (segmentElements[idx]) {
                            // 更新DOM的data属性
                            segmentElements[idx].dataset.start = data.start_time || 0;
                            segmentElements[idx].dataset.end = data.end_time || 0;
                            
                            // 格式化时间并更新显示
                            const startMin = Math.floor((data.start_time || 0) / 60);
                            const startSec = Math.floor((data.start_time || 0) % 60);
                            const endMin = Math.floor((data.end_time || 0) / 60);
                            const endSec = Math.floor((data.end_time || 0) % 60);
                            const timeText = `${startMin}:${startSec.toString().padStart(2, '0')} ~ ${endMin}:${endSec.toString().padStart(2, '0')}`;
                            
                            // 更新时间显示
                            segmentElements[idx].querySelector('p').textContent = timeText;
                        }
                    });
                } else {
                    console.warn("⚠️ 未找到knowledge_questions.json，使用模板默认时间");
                }
            } catch (e) {
                console.error("❌ 读取knowledge_questions.json出错：", e);
            }

            // 3. 知识点点击事件（跳转到对应时间 + 弹出题目）
            document.querySelectorAll('.segment').forEach(segEl => {
                segEl.addEventListener('click', function() {
                    // 移除其他active样式
                    document.querySelectorAll('.segment').forEach(el => el.classList.remove('active'));
                    this.classList.add('active');
                    
                    // 跳转到对应视频时间
                    const startTime = parseFloat(this.dataset.start) || 0;
                    video.currentTime = startTime;
                    video.play();
                    
                    // 记录当前知识点
                    const segIndex = parseInt(this.dataset.index) || 0;
                    currentSeg = {{ segments|tojson }}[segIndex];
                    
                    // 弹出引导题
                    showQuizPopup(currentSeg, 0);
                });
            });
        };

        // ========== 弹窗功能 ==========
        // 显示弹窗
        function showQuizPopup(seg, questionIdx) {
            if (!seg || !seg.questions || !seg.questions[questionIdx]) {
                alert("暂无对应题目数据");
                return;
            }
            
            currentSeg = seg;
            currentQuestionIdx = questionIdx;
            answered = false; // 重置答题状态
            
            // 设置弹窗标题和清空输入
            quizTitle.innerText = questionIdx === 0 ? `📌 引导题：${seg.questions[questionIdx].question}` : `📝 考查题：${seg.questions[questionIdx].question}`;
            userAnswer.value = '';
            feedback.innerText = '';
            feedback.className = 'feedback';
            answerAnalysis.style.display = 'none'; // 初始隐藏答案
            
            // 显示弹窗，暂停视频
            popup.style.display = 'flex';
            video.pause();
        }

        // 提交答案
        async function submitAnswer() {
            if (answered) return; // 防止重复提交
            
            const userAns = userAnswer.value.trim();
            if (!userAns) {
                feedback.innerText = '❌ 请输入答案后再提交';
                feedback.className = 'feedback wrong';
                return;
            }
            
            if (!currentSeg || !currentSeg.questions || !currentSeg.questions[currentQuestionIdx]) {
                feedback.innerText = '❌ 题目数据异常';
                feedback.className = 'feedback wrong';
                return;
            }

            // 获取标准答案
            const standardAns = currentSeg.questions[currentQuestionIdx].answer;
            feedback.innerText = '🔍 正在分析你的回答...';
            feedback.className = 'feedback';

            try {
                // 调用后端分析接口（备用，本地也有兜底逻辑）
                const res = await fetch('/api/analyze_answer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: currentSeg.questions[currentQuestionIdx].question,
                        user_answer: userAns,
                        standard_answer: standardAns,
                        video_content: currentSeg.description || ''
                    })
                });
                
                const data = await res.json();
                // 显示反馈
                feedback.innerText = data.is_correct ? '✅ 回答正确！' : '❌ 回答有误';
                feedback.className = `feedback ${data.is_correct ? 'correct' : 'wrong'}`;
                
                // 显示标准答案和解析（用户回答后才显示）
                standardAnswer.innerText = standardAns;
                answerAnalysisText.innerText = data.analysis || currentSeg.questions[currentQuestionIdx].analysis || '暂无解析';
                answerAnalysis.style.display = 'block';
                
                answered = true;
                
                // 正确则1.5秒后自动关闭
                if (data.is_correct) {
                    setTimeout(() => closePopup(), 1500);
                }
            } catch (e) {
                // 本地兜底逻辑
                console.error('分析失败:', e);
                const isCorrect = userAns.toLowerCase().includes(standardAns.toLowerCase().substring(0, 3));
                feedback.innerText = isCorrect ? '✅ 回答正确！' : '❌ 回答有误';
                feedback.className = `feedback ${isCorrect ? 'correct' : 'wrong'}`;
                
                // 显示标准答案和解析
                standardAnswer.innerText = standardAns;
                answerAnalysisText.innerText = isCorrect
                    ? '你的回答准确覆盖了核心知识点，表述清晰。'
                    : `正确答案参考：${standardAns}，建议理解核心要点后重新回答。`;
                answerAnalysis.style.display = 'block';
                
                answered = true;
            }
        }

        // 关闭弹窗
        function closePopup() {
            popup.style.display = 'none';
            userAnswer.value = '';
            feedback.innerText = '';
            feedback.className = 'feedback';
            answerAnalysis.style.display = 'none'; // 隐藏答案
            answered = false;
            currentQuestionIdx = 0;
            currentSeg = null;
            video.play(); // 关闭弹窗后继续播放视频
        }

        // ========== 视频进度监听（可选：知识点结束后弹出考查题） ==========
        let currentPlayingSeg = null;
        video.addEventListener('timeupdate', function() {
            const currentTime = video.currentTime;
            
            // 查找当前播放的知识点
            const segments = document.querySelectorAll('.segment');
            let matchedSeg = null;
            let matchedIdx = -1;
            
            segments.forEach((segEl, idx) => {
                const start = parseFloat(segEl.dataset.start) || 0;
                const end = parseFloat(segEl.dataset.end) || 0;
                if (currentTime >= start - 1 && currentTime <= end + 1) {
                    matchedSeg = {{ segments|tojson }}[idx];
                    matchedIdx = idx;
                }
            });
            
            // 知识点结束后弹出考查题
            if (currentPlayingSeg && matchedSeg !== currentPlayingSeg) {
                showQuizPopup(currentPlayingSeg, 1); // 1=考查题
                currentPlayingSeg = null;
            } else if (matchedSeg && !currentPlayingSeg) {
                currentPlayingSeg = matchedSeg;
            }
        });
    </script>
</body>
</html>
