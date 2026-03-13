// 播放进度保存
function saveProgress(videoId, currentTime) {
  fetch('/save_progress', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video_id: videoId, progress: currentTime})
  });
}
// 弹窗控制
function openModal(modalId) {document.getElementById(modalId).style.display='block';}
function closeModal(modalId) {document.getElementById(modalId).style.display='none';}
// 短视频点击联动
function playSubVideo(subVideoId, startTime) {
  const video = document.getElementById('main-video');
  video.currentTime = startTime;
  video.play();
  // 高亮选中项
  document.querySelectorAll('.sub-video-item').forEach(item => item.classList.remove('highlight'));
  document.getElementById(subVideoId).classList.add('highlight');
}
// 提问提交
function submitQuestion() {
  const question = document.getElementById('question-input').value;
  fetch('/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question: question, video_id: videoId})
  }).then(res => res.json()).then(data => {
    document.getElementById('qa-history').innerHTML += `<div>${data.question}: ${data.answer}</div>`;
  });
}
