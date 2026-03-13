<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{{ title }}</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box;font-family:Arial}
  body{background:#f4f4f4}
  .layout{display:flex;width:1600px;margin:0 auto;padding:20px;gap:20px}
  .sidebar{width:320px;background:#fff;border-radius:8px;padding:16px}
  .main{flex:1;background:#fff;border-radius:8px;padding:16px}
  .question-panel{width:320px;background:#fff;border-radius:8px;padding:16px}
  .video-box{position:relative;width:100%}
  video{width:100%;border-radius:8px}
  .quiz-popup{
    position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
    width:480px;background:#fff;border-radius:10px;box-shadow:0 0 20px rgba(0,0,0,0.3);
    padding:24px;z-index:999;display:none;
  }
  .quiz-popup h4{margin-bottom:16px;font-size:18px;color:#333}
  .quiz-popup input{
    width:100%;padding:12px;border:1px solid #ddd;border-radius:6px;margin:12px 0;
    font-size:14px;
  }
  .quiz-popup button{
    padding:10px 20px;background:#007bff;color:#fff;border:none;border-radius:6px;
    cursor:pointer;margin-right:8px;font-size:14px;
  }
  .quiz-popup .btn-secondary{background:#6c757d;}
  .feedback{margin-top:16px;line-height:1.6}
  .correct{color:#28a745;font-weight:bold}
  .wrong{color:#dc3545;font-weight:bold}
  .answer-analysis{margin-top:12px;padding:12px;background:#f8f9fa;border-radius:6px;display:none}
  .answer-analysis h5{margin-bottom:8px;color:#495057}
  .segment{
    padding:12px 16px;border:1px solid #eee;border-radius:6px;margin-bottom:12px;
    cursor:pointer;transition:all 0.2s;
  }
  .segment:hover{border-color:#007bff;background:#f8f9fa}
  .segment.active{border-color:#007bff;background:#e9f5ff}
  .segment h4{color:#212529;margin-bottom:4px}
  .segment p{color:#6c757d;font-size:14px;margin-bottom:8px}
  .view-questions-btn{
    padding:6px 12px;background:#007bff;color:#fff;border:none;border-radius:4px;
    font-size:13px;cursor:pointer;
  }
  .view-questions-btn:hover{background:#0056b3}
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 1000;
    display: none;
    justify-content: center;
    align-items: center;
  }
  .modal-content {
    background: #fff;
    padding: 24px;
    border-radius: 10px;
    width: 90%;
    max-width: 500px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
  }
  .modal-content h3 {
    margin-bottom: 16px;
    color: #333;
    border-bottom: 1px solid #eee;
    padding-bottom: 8px;
  }
  .modal-content .question-item {
    margin-bottom: 12px;
    padding: 8px;
    background: #f8f9fa;
    border-radius: 6px;
  }
  .modal-content .question-item strong {
    color: #007bff;
  }
  .modal-close {
    margin-top: 16px;
    padding: 8px 16px;
    background: #007bff;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    float: right;
  }
  .modal-close:hover {
    background: #0056b3;
  }
</style>
</head>
<body>
<div class="layout">
  <div class="sidebar">
    <h3>知识点列表</h3><br>
    {% for seg in knowledge_segments %}
    <div class="segment" data-index="{{ loop.index0 }}" data-start="{{ seg.start_time }}" data-end="{{ seg.end_time }}">
      <h4>{{ seg.title }}</h4>
      <p>{{ (seg.start_time|int)//60 }}:{{ '%02d' % ((seg.start_time|int)%60) }} ~ {{ (seg.end_time|int)//60 }}:{{ '%02d' % ((seg.end_time|int)%60) }}</p>
      <button class="view-questions-btn" onclick="event.stopPropagation(); showSegQuestionsModal({{ loop.index0 }})">
        查看知识点问题
      </button>
    </div>
    {% endfor %}
  </div>
  <div class="main">
    <h2>{{ title }}</h2><br>
    <div class="video-box">
      <video id="video" controls>
        <source src="{{ url_for('uploads',filename=filepath) }}" type="video/mp4">
      </video>
      <div class="quiz-popup" id="quizPopup">
        <h4 id="quizTitle">题目</h4>
        <input id="userAnswer" placeholder="请输入你的答案...">
        <div>
          <button onclick="submitAnswer()">提交答案</button>
          <button class="btn-secondary" onclick="closePopup()">关闭</button>
          <button class="btn-secondary" onclick="showAnswerAnalysis()" style="display:none" id="showAnalysisBtn">
            查看答案解析
          </button>
        </div>
        <div class="feedback" id="feedback"></div>
        <div class="answer-analysis" id="answerAnalysis">
          <h5>标准答案：</h5>
          <p id="standardAnswer"></p>
          <h5 style="margin-top:8px">回答分析：</h5>
          <p id="answerAnalysisText"></p>
        </div>
      </div>
    </div>
    <br>
    <h3>视频摘要</h3>
    <p>{{ summary }}</p>
  </div>
  <div class="question-panel">
    <h3>智能问答</h3>
    <input type="text" id="question" placeholder="输入问题" style="width:100%;padding:12px;margin:10px 0;border:1px solid #ddd;border-radius:6px">
    <button onclick="askAI()" style="padding:10px 20px;background:#007bff;color:#fff;border:none;border-radius:6px;cursor:pointer;width:100%">提问</button>
    <div id="answer" style="margin-top:16px;line-height:1.6;min-height:300px;"></div>
  </div>
</div>
<div class="modal-overlay" id="questionsModal">
  <div class="modal-content">
    <h3 id="modalTitle">知识点问题</h3>
    <div id="modalQuestions"></div>
    <button class="modal-close" onclick="closeModal()">确定</button>
  </div>
</div>
<script>
const video = document.getElementById('video');
const popup = document.getElementById('quizPopup');
const quizTitle = document.getElementById('quizTitle');
const userAnswer = document.getElementById('userAnswer');
const feedback = document.getElementById('feedback');
const showAnalysisBtn = document.getElementById('showAnalysisBtn');
const answerAnalysis = document.getElementById('answerAnalysis');
const standardAnswer = document.getElementById('standardAnswer');
const answerAnalysisText = document.getElementById('answerAnalysisText');

const segments = {{ knowledge_segments|tojson }};
let currentSeg = null;
let mode = '';
let currentQuestionIdx = 0;
let answered = false;

async function submitAnswer() {
  if (answered) return;
  const userAns = userAnswer.value.trim();
  if (!userAns) {
    feedback.innerText = '❌ 请输入答案后再提交';
    feedback.className = 'feedback wrong';
    return;
  }
  const standardAns = currentSeg.questions[currentQuestionIdx].answer;
  feedback.innerText = '🔍 正在分析你的回答...';
  feedback.className = 'feedback';
  try {
    const res = await fetch('/api/analyze_answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: currentSeg.questions[currentQuestionIdx].question,
        user_answer: userAns,
        standard_answer: standardAns,
        video_content: currentSeg.content
      })
    });
    const data = await res.json();
    feedback.innerText = data.is_correct ? '✅ 回答正确！' : '❌ 回答有误';
    feedback.className = `feedback ${data.is_correct ? 'correct' : 'wrong'}`;
    standardAnswer.innerText = standardAns;
    answerAnalysisText.innerText = data.analysis;
    showAnalysisBtn.style.display = 'inline-block';
    answered = true;
    if (data.is_correct) {
      setTimeout(() => closePopup(), 1500);
    }
  } catch (e) {
    const isCorrect = userAns.toLowerCase().includes(standardAns.toLowerCase().substring(0, 3));
    feedback.innerText = isCorrect ? '✅ 回答正确！' : '❌ 回答有误';
    feedback.className = `feedback ${isCorrect ? 'correct' : 'wrong'}`;
    standardAnswer.innerText = standardAns;
    answerAnalysisText.innerText = isCorrect
      ? '你的回答准确覆盖了核心知识点，表述清晰。'
      : '你的回答缺少核心要点，建议参考标准答案完善。';
    showAnalysisBtn.style.display = 'inline-block';
    answered = true;
  }
}
document.querySelectorAll('.segment').forEach(el => {
  el.onclick = () => {
    const s = parseFloat(el.dataset.start);
    video.currentTime = s;
    video.play();
    document.querySelectorAll('.segment').forEach(x => x.classList.remove('active'));
    el.classList.add('active');
  }
});

let questionsData = [];
window.onload = async function() {
  try {
    const res = await fetch(`/video_data/{{ video_id }}/knowledge_questions.json`);
    if (res.ok) {
      questionsData = await res.json();
      console.log("✅ 题目数据加载成功：", questionsData);
    } else {
      console.warn("⚠️ 题目文件加载失败，状态码：", res.status);
    }
  } catch (e) {
    console.error("❌ 加载题目数据出错：", e);
  }
};

function showSegQuestionsModal(idx) {
  const seg = segments[idx];
  const modalTitle = document.getElementById('modalTitle');
  const modalQuestions = document.getElementById('modalQuestions');
  const modal = document.getElementById('questionsModal');
  modalTitle.textContent = seg.title;
  const questionItem = questionsData.find(item => item.title === seg.title);
  let html = '';
  if (questionItem && questionItem.questions && questionItem.questions.length >= 2) {
    html = `
      <div class="question-item">
        <strong>引导题：</strong>${questionItem.questions[0].question}
        <br><small style="color:#6c757d">标准答案：${questionItem.questions[0].answer}</small>
      </div>
      <div class="question-item">
        <strong>考查题：</strong>${questionItem.questions[1].question}
        <br><small style="color:#6c757d">标准答案：${questionItem.questions[1].answer}</small>
      </div>
    `;
  } else {
    html = '<div style="color:#dc3545">暂无该知识点的题目数据</div>';
  }
  modalQuestions.innerHTML = html;
  modal.style.display = 'flex';
  video.currentTime = seg.start_time || seg.start || 0;
  video.play();
}

function closeModal() {
  const modal = document.getElementById('questionsModal');
  modal.style.display = 'none';
}

async function askAI() {
  const q = document.getElementById('question').value;
  if (!q) return;
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_id: {{ video_id }}, question: q })
  });
  const data = await res.json();
  document.getElementById('answer').innerText = data.answer || '暂无回答';
}

function closePopup() {
  popup.style.display = 'none';
  userAnswer.value = '';
  feedback.innerText = '';
  feedback.className = 'feedback';
  showAnalysisBtn.style.display = 'none';
  answerAnalysis.style.display = 'none';
  answered = false;
}

function showAnswerAnalysis() {
  answerAnalysis.style.display = 'block';
}
</script>
</body>
</html>
