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
import requests
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
from core.subject_utils import infer_subject_hint
import requests
from werkzeug.utils import secure_filename

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
    attempts = int(memory.get("attempts", 0) or 0)
    correct = int(memory.get("correct", 0) or 0)
    total_chars = int(memory.get("total_chars", 0) or 0)
    accuracy = (correct / attempts) if attempts else 0.0
    avg_len = (total_chars / attempts) if attempts else 0.0

    mastery_level = "高" if accuracy >= 0.8 else ("中" if accuracy >= 0.55 else "基础")
    difficulty = "挑战" if accuracy >= 0.8 else ("进阶" if accuracy >= 0.55 else "基础")
    learning_style = "解释型" if avg_len >= 35 else ("均衡型" if avg_len >= 18 else "速答型")
    confidence = min(100, int(accuracy * 100 * 0.7 + min(avg_len, 40) * 0.8))

    return {"mastery_level": mastery_level, "learning_style": learning_style, "confidence": confidence, "recommended_difficulty": difficulty, "accuracy": round(accuracy * 100, 1)}

def _build_path_recommendation(profile, memory, segment_title):
    segment_title = segment_title or "当前知识点"
    history = memory.get("history") or []
    recent = history[-3:]
    recent_wrong = sum(1 for item in recent if not item.get("is_correct"))

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

# 轻量内存态学习档案（演示用）；生产环境建议迁移到数据库
LEARNING_MEMORY = {}
USER_VIDEOS = {}
UPLOAD_DIR = os.path.join(app.root_path, 'static', 'uploads')
VIDEO_DATA_DIR = os.path.join(app.root_path, 'video_data')
ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
VIDEO_ID_LOCK = threading.Lock()
VIDEO_REGISTRY_LOCK = threading.Lock()
VIDEO_REGISTRY_PATH = os.path.join(VIDEO_DATA_DIR, '_user_videos.json')


def _existing_video_ids():
    ids = set()
    for base_dir in [VIDEO_DATA_DIR, os.path.join(app.root_path, 'static', 'video_data')]:
        if not os.path.isdir(base_dir):
            continue
        for name in os.listdir(base_dir):
            if name.isdigit():
                ids.add(int(name))
    return ids


def _get_next_video_id():
    existing_ids = _existing_video_ids()
    if not existing_ids:
        return 100
    return max(existing_ids) + 1


NEXT_VIDEO_ID = _get_next_video_id()


def _video_artifact_paths(video_id):
    video_id = str(video_id)
    work_dir = os.path.join(VIDEO_DATA_DIR, video_id)
    return {
        'work_dir': work_dir,
        'corrected_asr': os.path.join(work_dir, 'corrected_asr.txt'),
        'final_splits': os.path.join(work_dir, 'final_knowledge_splits.json'),
        'summaries': os.path.join(work_dir, 'subvideo_summaries_all.json'),
        'custom_kg': os.path.join(work_dir, f'custom_kg_{video_id}.json'),
        'relation_network': os.path.join(work_dir, 'relation_network.json'),
        'final_llm_input': os.path.join(work_dir, 'final_llm_input.json'),
        'knowledge_questions': os.path.join(work_dir, 'knowledge_questions.json'),
    }


def _safe_load_json_file(file_path, default):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def _save_json_file(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _segment_sort_key(name):
    match = re.search(r'(\d+)', str(name))
    return int(match.group(1)) if match else 10 ** 9


def _load_segment_records_from_artifacts(video_id):
    paths = _video_artifact_paths(video_id)
    splits = _safe_load_json_file(paths['final_splits'], [])
    summaries_doc = _safe_load_json_file(paths['summaries'], {})
    summary_map = summaries_doc.get('subvideo_summaries', {}) if isinstance(summaries_doc, dict) else {}
    records = []

    if isinstance(splits, list):
        for index, seg in enumerate(splits, start=1):
            summary_item = summary_map.get(f'segment_{index}.mp4', {}) if isinstance(summary_map, dict) else {}
            title = str(
                seg.get('knowledge_point') or
                seg.get('title') or
                (summary_item.get('title') if isinstance(summary_item, dict) else '') or
                f'知识点 {index}'
            ).strip()
            start = float(seg.get('start_time', 0) or 0)
            end = float(seg.get('end_time', start) or start)
            if end < start:
                end = start
            records.append({
                'segment_id': index,
                'segment_name': f'segment_{index}.mp4',
                'title': title,
                'start': start,
                'end': end,
            })

    if records:
        return records

    if isinstance(summary_map, dict):
        for index, segment_name in enumerate(sorted(summary_map.keys(), key=_segment_sort_key), start=1):
            item = summary_map.get(segment_name) or {}
            records.append({
                'segment_id': index,
                'segment_name': segment_name,
                'title': str(item.get('title') or f'知识点 {index}').strip(),
                'start': 0.0,
                'end': 0.0,
            })

    return records


def _build_completed_video_summary(video_id):
    segment_count = len(_load_segment_records_from_artifacts(video_id))
    if segment_count:
        return f"AI 已完成解析，共拆分 {segment_count} 个知识点，可开始学习。"
    return "AI 已完成解析，可开始学习。"


def _has_completed_video_artifacts(video_id):
    paths = _video_artifact_paths(video_id)
    required = ('corrected_asr', 'final_splits', 'summaries', 'knowledge_questions')
    return all(os.path.exists(paths[key]) for key in required)


def _write_relation_network_fallback(video_id):
    paths = _video_artifact_paths(video_id)
    summaries_doc = _safe_load_json_file(paths['summaries'], {})
    summary_map = summaries_doc.get('subvideo_summaries', {}) if isinstance(summaries_doc, dict) else {}
    custom_kg = _safe_load_json_file(paths['custom_kg'], {})

    nodes = []
    for segment_name in sorted(summary_map.keys(), key=_segment_sort_key):
        item = summary_map.get(segment_name) or {}
        nodes.append({
            'video_name': segment_name,
            'video_id': item.get('video_id', ''),
            'title': item.get('title') or segment_name,
            'key_concepts': item.get('key_concepts', []),
        })

    fallback_relation = {
        'nodes': nodes,
        'batch_edges': [],
        'cross_edges': [],
        'all_edges': [],
        'overall_structure': '关系网生成失败，已写入仅含节点的兜底结构。',
        'coverage_check': {
            'total_videos': len(nodes),
            'batch_coverage': [],
            'cross_coverage': [],
            'merged_node_names': [node['video_name'] for node in nodes],
        },
        'batch_info': {
            'fallback': True,
            'reason': 'relation_network_generation_failed',
        }
    }

    final_input = {
        'metadata': {
            'video_id': str(video_id),
            'video_count': len(nodes),
            'description': '关系网生成失败后的兜底输入',
        },
        'subvideo_summaries': summary_map,
        'custom_knowledge_graph': custom_kg,
        'subvideo_relation_network': fallback_relation,
    }

    _save_json_file(paths['relation_network'], fallback_relation)
    _save_json_file(paths['final_llm_input'], final_input)
    return fallback_relation


def _write_question_fallback(video_id, reason='题目生成阶段异常，已保留知识点结构，可稍后重试。'):
    paths = _video_artifact_paths(video_id)
    fallback_items = []
    for record in _load_segment_records_from_artifacts(video_id):
        segment_id = int(record.get('segment_id') or len(fallback_items) + 1)
        fallback_items.append({
            'should_generate_questions': False,
            'skip_reason': reason,
            'questions': [],
            'segment_id': segment_id,
            'segment_folder': f't{segment_id}',
            'title': record.get('title') or f'知识点 {segment_id}',
            'start': float(record.get('start', 0) or 0),
            'end': float(record.get('end', 0) or 0),
        })

    _save_json_file(paths['knowledge_questions'], fallback_items)
    return fallback_items


def _normalize_video_entry(entry):
    if not isinstance(entry, dict):
        return None

    try:
        video_id = int(entry.get('id'))
    except (TypeError, ValueError):
        return None

    status = entry.get('status')
    if status not in {'processing', 'completed', 'failed'}:
        status = 'processing'
    if _has_completed_video_artifacts(video_id):
        status = 'completed'

    stored_filename = str(entry.get('stored_filename') or '').strip()
    display_name = re.sub(r'^\d+_', '', stored_filename)
    display_name = display_name.replace('.cancelled', '')
    filename = str(entry.get('filename') or '').strip() or display_name or f"视频 {video_id}"
    title = str(entry.get('title') or '').strip() or f"视频 {video_id}"
    subject = str(entry.get('subject') or '').strip()
    if status == 'completed' and title == '解析中...':
        title = filename or f"视频 {video_id}"

    default_summary = {
        'processing': 'AI正在深度解析视频，提取知识图谱与考点...',
        'completed': 'AI 已完成解析，可开始学习。',
        'failed': 'AI 处理失败，可重新上传重试。'
    }
    summary = str(entry.get('summary') or '').strip() or default_summary[status]
    if status == 'completed' and summary in {default_summary['processing'], default_summary['failed']}:
        summary = _build_completed_video_summary(video_id)

    return {
        'id': video_id,
        'filename': filename,
        'status': status,
        'stored_filename': stored_filename,
        'title': title,
        'summary': summary,
        'subject': subject,
    }


def _build_video_registry_snapshot():
    snapshot = {}
    for user_id, videos in USER_VIDEOS.items():
        normalized = []
        for video in videos:
            item = _normalize_video_entry(video)
            if item:
                normalized.append(item)
        if normalized:
            snapshot[user_id] = sorted(normalized, key=lambda item: item['id'])
    return snapshot


def _write_video_registry(snapshot):
    os.makedirs(os.path.dirname(VIDEO_REGISTRY_PATH), exist_ok=True)
    temp_path = f"{VIDEO_REGISTRY_PATH}.tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, VIDEO_REGISTRY_PATH)


def _first_artifact_mtime(video_dir):
    timestamps = []
    for root_dir, _, filenames in os.walk(video_dir):
        for filename in filenames:
            file_path = os.path.join(root_dir, filename)
            try:
                timestamps.append(os.path.getmtime(file_path))
            except OSError:
                continue
    if timestamps:
        return min(timestamps)
    return os.path.getmtime(video_dir)


def _bootstrap_completed_guest_videos(existing_snapshot):
    existing_ids = {
        int(video.get('id'))
        for videos in existing_snapshot.values()
        for video in videos
        if isinstance(video, dict) and str(video.get('id', '')).isdigit()
    }

    upload_candidates = []
    if os.path.isdir(UPLOAD_DIR):
        for name in os.listdir(UPLOAD_DIR):
            full_path = os.path.join(UPLOAD_DIR, name)
            if not os.path.isfile(full_path) or name.lower().endswith('.cancelled'):
                continue
            _, ext = os.path.splitext(name.lower())
            if ext not in ALLOWED_VIDEO_EXTS:
                continue
            upload_candidates.append({
                'stored_filename': name,
                'filename': re.sub(r'^\d+_', '', name),
                'mtime': os.path.getmtime(full_path)
            })
    upload_candidates.sort(key=lambda item: item['mtime'])

    recovered = []
    pending_videos = []
    if os.path.isdir(VIDEO_DATA_DIR):
        for name in os.listdir(VIDEO_DATA_DIR):
            if not name.isdigit():
                continue
            video_id = int(name)
            if video_id in existing_ids:
                continue

            video_dir = os.path.join(VIDEO_DATA_DIR, name)
            if not os.path.isdir(video_dir):
                continue

            if not os.path.exists(os.path.join(video_dir, 'knowledge_questions.json')):
                continue

            segments = _load_video_segments(video_id)
            segment_count = len(segments)
            summary = f"AI 已完成解析，共拆分 {segment_count} 个知识点，可开始学习。" if segment_count else "AI 已完成解析，可开始学习。"
            pending_videos.append({
                'id': video_id,
                'title': f"视频 {video_id}",
                'summary': summary,
                'first_artifact_mtime': _first_artifact_mtime(video_dir)
            })

    pending_videos.sort(key=lambda item: item['first_artifact_mtime'])
    used_upload_indexes = set()

    for item in pending_videos:
        chosen_index = None
        for index, upload in enumerate(upload_candidates):
            if index in used_upload_indexes:
                continue
            if upload['mtime'] <= item['first_artifact_mtime']:
                chosen_index = index

        filename = f"视频 {item['id']}"
        stored_filename = ''
        if chosen_index is not None:
            used_upload_indexes.add(chosen_index)
            filename = upload_candidates[chosen_index]['filename']
            stored_filename = upload_candidates[chosen_index]['stored_filename']

        recovered.append({
            'id': item['id'],
            'filename': filename,
            'status': 'completed',
            'stored_filename': stored_filename,
            'title': item['title'],
            'summary': item['summary']
        })

    return recovered


def _load_video_registry():
    snapshot = {}

    if os.path.exists(VIDEO_REGISTRY_PATH):
        try:
            with open(VIDEO_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            if isinstance(raw_data, dict):
                for user_id, videos in raw_data.items():
                    normalized = []
                    for video in videos if isinstance(videos, list) else []:
                        item = _normalize_video_entry(video)
                        if item:
                            normalized.append(item)
                    if normalized:
                        snapshot[user_id] = sorted(normalized, key=lambda item: item['id'])
        except Exception as exc:
            logger.warning(f"加载视频注册表失败，准备回退到磁盘恢复：{exc}")

    recovered_videos = _bootstrap_completed_guest_videos(snapshot)
    if recovered_videos:
        snapshot.setdefault('guest', [])
        snapshot['guest'].extend(recovered_videos)
        snapshot['guest'] = sorted(snapshot['guest'], key=lambda item: item['id'])

    if snapshot:
        try:
            _write_video_registry(snapshot)
        except Exception as exc:
            logger.warning(f"写入视频注册表失败：{exc}")

    return snapshot


def _persist_user_videos_locked():
    _write_video_registry(_build_video_registry_snapshot())


def _get_user_videos(user_id):
    with VIDEO_REGISTRY_LOCK:
        return [dict(video) for video in USER_VIDEOS.get(user_id, [])]


def _get_user_video(user_id, video_id):
    with VIDEO_REGISTRY_LOCK:
        for video in USER_VIDEOS.get(user_id, []):
            if video.get('id') == video_id:
                return dict(video)
    return None


def _get_video_display_title(video, fallback):
    if video:
        filename = str(video.get('filename') or '').strip()
        if filename:
            return filename

        title = str(video.get('title') or '').strip()
        if title:
            return title

    return fallback


def _load_corrected_asr_excerpt(video_id, limit=2400):
    corrected_asr_path = os.path.join(VIDEO_DATA_DIR, str(video_id), 'corrected_asr.txt')
    if not os.path.exists(corrected_asr_path):
        return ''
    try:
        with open(corrected_asr_path, 'r', encoding='utf-8') as f:
            return f.read(limit).strip()
    except Exception:
        return ''


def _build_video_subject_hint(video_id=None, video=None, knowledge_segments=None, corrected_asr=None):
    segment_titles = [
        (seg.get('title') or seg.get('knowledge_point') or '').strip()
        for seg in (knowledge_segments or [])
        if isinstance(seg, dict)
    ]
    if corrected_asr is None and video_id is not None:
        corrected_asr = _load_corrected_asr_excerpt(video_id)

    return infer_subject_hint(
        (video or {}).get('subject'),
        (video or {}).get('title'),
        (video or {}).get('filename'),
        (video or {}).get('summary'),
        '\n'.join(segment_titles[:12]),
        corrected_asr,
    )


def _estimate_video_duration_seconds(video_id):
    try:
        segments = _load_video_segments(video_id)
    except Exception:
        return 0.0

    if not segments:
        return 0.0

    return max(float(seg.get('end_time', 0) or 0) for seg in segments)


def _require_user_video(video_id):
    user_id = _current_user_id()
    selected_video = _get_user_video(user_id, video_id)
    if not selected_video:
        abort(404)
    return selected_video


def _upsert_user_video(user_id, entry):
    normalized = _normalize_video_entry(entry)
    if not normalized:
        return None

    with VIDEO_REGISTRY_LOCK:
        videos = USER_VIDEOS.setdefault(user_id, [])
        for index, video in enumerate(videos):
            if video.get('id') == normalized['id']:
                videos[index] = normalized
                break
        else:
            videos.append(normalized)
            videos.sort(key=lambda item: item['id'])
        _persist_user_videos_locked()

    return dict(normalized)


def _patch_user_video(user_id, video_id, **fields):
    with VIDEO_REGISTRY_LOCK:
        videos = USER_VIDEOS.setdefault(user_id, [])
        target = next((video for video in videos if video.get('id') == video_id), None)
        if target is None:
            target = {'id': video_id}
            videos.append(target)
        target.update(fields)
        normalized = _normalize_video_entry(target)
        if not normalized:
            return None
        for index, video in enumerate(videos):
            if video.get('id') == video_id:
                videos[index] = normalized
                break
        videos.sort(key=lambda item: item['id'])
        _persist_user_videos_locked()
        return dict(normalized)


def _remove_user_video(user_id, video_id):
    with VIDEO_REGISTRY_LOCK:
        videos = USER_VIDEOS.get(user_id, [])
        target = next((video for video in videos if video.get('id') == video_id), None)
        if target:
            videos.remove(target)
            _persist_user_videos_locked()
            return dict(target)
    return None


USER_VIDEOS = _load_video_registry()


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
    splits_path = os.path.join(VIDEO_DATA_DIR, str(video_id), 'final_knowledge_splits.json')
    if os.path.exists(splits_path):
        try:
            with open(splits_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            segments = []
            for idx, block in enumerate(data if isinstance(data, list) else []):
                title = (
                    (block.get('knowledge_point') or '').strip()
                    or (block.get('title') or '').strip()
                    or f'知识点 {idx + 1}'
                )
                start_time = float(block.get('start_time', 0) or 0)
                end_time = float(block.get('end_time', start_time) or start_time)
                if end_time <= start_time:
                    end_time = start_time + 1
                segments.append({
                    "title": title,
                    "start_time": start_time,
                    "end_time": end_time,
                })
            if segments:
                return segments
        except Exception:
            pass

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
        start_time = float(block.get('start', 2 + idx * 60) or 0)
        end_time = float(block.get('end', start_time + 45) or (start_time + 45))
        if end_time <= start_time:
            end_time = start_time + 45
        segments.append({"title": title, "start_time": start_time, "end_time": end_time})
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


def _build_recommended_questions(video_id, knowledge_segments, video_title=None, subject_hint=None, video_summary=None):
    recommended = []
    seen = set()

    segment_titles = []
    for seg in knowledge_segments or []:
        title = (seg.get('title') or '').strip()
        if title:
            segment_titles.append(title)

    llm_questions = _generate_recommended_questions_with_llm(
        video_title=video_title,
        segment_titles=segment_titles,
        subject_hint=subject_hint,
        video_summary=video_summary,
    )
    for q in llm_questions:
        if q and q not in seen:
            seen.add(q)
            recommended.append(q)
        if len(recommended) >= 4:
            return recommended

    # 2) 若模型生成不足，按当前视频知识点自动补齐
    templates = [
        "这里的“{title}”到底在讲什么？",
        "学习 {title} 时最容易卡住的点是什么？",
        "为什么这里要先讲 {title}？",
        "{title} 和前面内容之间是什么关系？",
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
        f"这条视频《{video_title or '当前内容'}》主要想让我学会什么？",
        "这节课最容易让人听懂但不会做的是哪部分？",
        "如果我要快速复习，这节课应该按什么顺序抓重点？",
        "这节内容里有哪些地方最值得我停下来再看一遍？",
        "我应该先补哪部分前置知识，再回来听这一节？",
    ]
    for q in fallback:
        if q not in seen:
            recommended.append(q)
        if len(recommended) >= 4:
            break

    return recommended[:4]


def _generate_recommended_questions_with_llm(video_title=None, segment_titles=None, subject_hint=None, video_summary=None):
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        return []

    model = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
    endpoint = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    segment_titles = [title for title in (segment_titles or []) if title]
    subject_rule = (
        f"请严格限制在「{subject_hint}」学科范围内提问，不要跨到其他课程。"
        if subject_hint else
        "请先识别课程主学科，再保持问题始终在同一学科上下文中。"
    )
    prompt = f"""
你是教学视频学习助手。

请站在“正在看视频的学生”角度，针对这节课生成 4 个最自然、最可能出现的疑问句，作为界面里的“推荐问题”。

要求：
1. 问题要像学生会主动问出来的话，不要像老师命题，也不要像考试题。
2. 优先生成“听不懂、想确认、想串联、想抓重点”这类真实疑问。
3. 避免空泛大话，避免“帮我出一道题”“请总结一下”这类助教口吻。
4. 问题要尽量具体，但不要依赖当前播放器时间。
5. 每条问题一句话，长度自然，适合直接点击提问。
6. {subject_rule}
7. 问题不要照抄知识点标题，要更像学生在当前视频里会追问的说法。
8. 只输出 JSON，格式如下：
{{
  "questions": ["问题1", "问题2", "问题3", "问题4"]
}}

视频标题：
{video_title or "当前教学视频"}

课程摘要：
{video_summary or "无"}

本节涉及的知识点标题：
{json.dumps(segment_titles[:12], ensure_ascii=False)}
""".strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": "你擅长从学生视角生成自然、有启发性的教学视频提问建议。"},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
        if resp.status_code != 200:
            return []
        data = resp.json()
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        if not content:
            return []

        try:
            parsed = json.loads(content)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", content)
            parsed = json.loads(match.group(0)) if match else {}

        questions = parsed.get("questions") or []
        cleaned = []
        for item in questions:
            q = str(item or "").strip()
            if not q:
                continue
            cleaned.append(q)
            if len(cleaned) >= 4:
                break
        return cleaned
    except Exception:
        return []


def _score_to_level(score):
    if score >= 80:
        return "优秀"
    if score >= 55:
        return "良好"
    return "待提升"


def _default_insight_view_data():
    return {
        "evaluation": {
            "score": 0,
            "level": _score_to_level(0),
            "comment": "杩樻病鏈夌瓟棰樿褰曪紝鍏堝畬鎴愪竴娆￠殢鍫傛祴楠屽悗杩欓噷浼氳嚜鍔ㄦ洿鏂般€?",
            "strengths": ["灏氭棤璁板綍"],
            "weaknesses": ["缁х画閫氳繃闅忓爞娴嬮獙绉疮鏁版嵁"],
        },
        "profile": {
            "mastery_level": "鍩虹",
            "learning_style": "閫熺瓟鍨?",
            "confidence": 0,
            "recommended_difficulty": "鍩虹",
            "accuracy": 0.0,
        },
        "memory": {
            "attempts": 0,
            "accuracy": 0.0,
            "weak_topics": [],
            "recent_records": [],
        },
        "path": {
            "action": "琛ラ綈鍓嶇疆",
            "recommendation": "瀹屾垚绛旈鍚庝細鑷姩鎺ㄨ崘瀛︿範璺緞銆?",
        },
    }


def _normalize_insight_view_data(insights):
    normalized = _default_insight_view_data()
    if not isinstance(insights, dict):
        return normalized

    for key in ("evaluation", "profile", "memory", "path"):
        value = insights.get(key)
        if isinstance(value, dict):
            normalized[key].update(value)

    return normalized


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
    history = memory.get("history") or []
    recent = history[-1] if history else None
    recent_score = recent.get("score", 0) if recent else 0

    if recent:
        evaluation_comment = "最近一次答题已纳入画像分析，可继续提问进行针对性提升。"
    else:
        evaluation_comment = "还没有答题记录，先完成一次随堂测验后这里会自动更新。"

    sorted_mistakes = sorted((memory.get("topic_mistakes") or {}).items(), key=lambda x: x[1], reverse=True)
    weak_topics = [name for name, _ in sorted_mistakes[:3]]
    path = _build_path_recommendation(profile, memory, recent["segment_title"] if recent else "当前知识点")

    recent_records = []
    for item in history[-5:]:
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
            "attempts": int(memory.get("attempts", 0) or 0),
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
# 静态资源路由 (适配前端加载)
# ==========================================
@app.route('/uploads/<filename>')
def uploads(filename):
    """前端视频播放器获取视频文件的接口"""
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/video_data/<int:video_id>/<path:filename>')
def video_data(video_id, filename):
    """前端加载知识点题库 JSON 的接口"""
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
        {
            'id': 1,
            'filename': '高等数学-微积分基础.mp4',
            'status': 'completed',
            'title': '微积分入门',
            'summary': '本节围绕极限、导数和基本应用建立微积分入门框架。',
            'subject': '高等数学',
        }
    ]
    user_id = _current_user_id()
    user_videos = _get_user_videos(user_id)
    videos = list(reversed(user_videos)) if user_videos else demo_videos
    for video in videos:
        if video.get('subject'):
            continue
        video['subject'] = _build_video_subject_hint(video_id=video.get('id'), video=video)
    total_learning_hours = round(
        sum(_estimate_video_duration_seconds(video.get('id')) for video in videos) / 3600,
        1,
    )
    student_overview = {
        'total_videos': len(videos),
        'completed': sum(1 for v in videos if v['status'] == 'completed'),
        'processing': sum(1 for v in videos if v['status'] == 'processing'),
        'learning_hours': total_learning_hours
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
    current_video = _get_user_video(user_id, video_id)
    subject_hint = _build_video_subject_hint(video_id=video_id, video=current_video)
    web_title = _get_video_display_title(current_video, f"视频 {video_id}")
    web_summary = _build_completed_video_summary(vid_str)

    def _run_optional_stage(stage_name, stage_func, fallback=None):
        try:
            logger.info(f"开始执行 [{stage_name}]")
            return stage_func()
        except Exception as exc:
            logger.warning(f"⚠️ {stage_name} 失败，已降级继续：{exc}", exc_info=True)
            if fallback is None:
                return None
            try:
                return fallback(exc)
            except Exception as fallback_exc:
                logger.warning(f"⚠️ {stage_name} 兜底仍失败：{fallback_exc}", exc_info=True)
                return None

    try:
        # [步骤0：真实ASR提取 + 矫正]
        corrected_asr_path, raw_asr_path, corrected_asr = _prepare_real_asr(vid_str, video_path, work_dir)
        logger.info(f"ASR文件准备完成：corrected={corrected_asr_path} | raw={raw_asr_path}")
        subject_hint = _build_video_subject_hint(video_id=video_id, video=current_video, corrected_asr=corrected_asr)

        # [步骤1：知识点拆分与物理切割]
        logger.info("开始执行 [知识点拆分与物理切割]")
        final_splits = run_knowledge_split(video_path, corrected_asr_path, vid_str)
        logger.info(f"知识点拆分完成：{len(final_splits) if isinstance(final_splits, list) else '未知'}段")

        # [步骤2：子视频多模态摘要]
        logger.info("开始执行 [子视频多模态摘要]")
        split_dir = os.path.join(work_dir, 'split_videos')
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"拆分后的子视频目录不存在：{split_dir}")
        run_video_summary(split_dir, VIDEO_DATA_DIR, vid_str, subject_hint=subject_hint)

        # [步骤3：知识图谱]
        _run_optional_stage("知识图谱生成", lambda: generate_video_kg(vid_str))

        # [步骤4：关系网生成]
        custom_kg_path = _video_artifact_paths(vid_str)['custom_kg']
        if os.path.exists(custom_kg_path):
            _run_optional_stage(
                "视频关系网生成",
                lambda: generate_relation_network(vid_str, work_dir, custom_kg_path),
                fallback=lambda _exc: _write_relation_network_fallback(vid_str),
            )
        else:
            logger.warning(f"未找到知识图谱文件，跳过关系网生成：{custom_kg_path}")
            _write_relation_network_fallback(vid_str)

        # [步骤5：生成问题]
        _run_optional_stage(
            "随堂问题生成",
            lambda: generate_questions_for_knowledge_points(vid_str, subject_hint=subject_hint),
            fallback=lambda exc: _write_question_fallback(
                vid_str,
                reason=f"题目生成阶段异常：{str(exc)[:120]}",
            ),
        )
        question_path = _video_artifact_paths(vid_str)['knowledge_questions']
        if not os.path.exists(question_path):
            _write_question_fallback(vid_str, reason='题目生成阶段未产出结果，已保留知识点结构，可稍后重试。')

        # [步骤6：生成网页标题摘要]
        logger.info("开始执行 [总结标题摘要]")
        web_title, web_summary = run_generate_web_title_summary(
            video_id=vid_str,
            corrected_asr=corrected_asr,
            subject_hint=subject_hint,
        )
        if not web_title or web_title == '未命名视频':
            web_title = _get_video_display_title(current_video, f"视频 {video_id}")
        if not web_summary or web_summary == '暂无摘要':
            web_summary = _build_completed_video_summary(vid_str)

        if not _has_completed_video_artifacts(vid_str):
            raise RuntimeError('核心产物未生成完整，无法标记完成')

        _patch_user_video(
            user_id,
            video_id,
            status='completed',
            title=web_title,
            summary=web_summary,
            subject=subject_hint,
        )

        logger.info(f"✅ 视频 {video_id} AI 流水线全链条处理完毕！")

    except Exception as e:
        logger.error(f"❌ 视频 {video_id} AI 处理崩溃: {str(e)}", exc_info=True)
        if _has_completed_video_artifacts(vid_str):
            _patch_user_video(
                user_id,
                video_id,
                status='completed',
                title=web_title,
                summary=web_summary,
                subject=subject_hint,
            )
        else:
            _patch_user_video(user_id, video_id, status='failed')

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

        with VIDEO_ID_LOCK:
            NEXT_VIDEO_ID = max(NEXT_VIDEO_ID, _get_next_video_id())
            current_video_id = NEXT_VIDEO_ID
            NEXT_VIDEO_ID += 1

        _upsert_user_video(user_id, {
            'id': current_video_id,
            'filename': original_name,
            'status': 'processing', 
            'stored_filename': unique_name,
            'title': '解析中...',
            'summary': 'AI正在深度解析视频，提取知识图谱与考点...',
            'subject': infer_subject_hint(original_name),
        })

        # 启动后台AI线程
        threading.Thread(target=process_video_pipeline, args=(current_video_id, save_path, user_id), daemon=True).start()

        flash(f'视频上传成功，后台 AI 流水线已启动：{original_name}', 'success')
        return redirect(url_for('workspace'))
    return render_template('upload.html')

@app.route('/delete/<int:video_id>', methods=['POST'])
def delete_video(video_id):
    user_id = _current_user_id()
    target = _remove_user_video(user_id, video_id)
    if target:
        flash('学习记录已彻底删除', 'success')
    return redirect(url_for('workspace'))

# ==========================================
# 沉浸学习详情页
# ==========================================
@app.route('/video_detail/<int:video_id>')
def video_detail(video_id):
    selected_video = _require_user_video(video_id)

    # 加载真实的 AI 分段
    video_segments = _load_video_segments(video_id)
    if not video_segments:
        video_segments = [
            {"title": "第一章：什么是微积分", "start_time": 2, "end_time": 10},
            {"title": "第二章：导数的应用", "start_time": 15, "end_time": 25}
        ]

    title = _get_video_display_title(selected_video, f"知识点视频 {video_id}")
    filepath = selected_video['stored_filename'] if selected_video else "sample.mp4"
    summary = selected_video['summary'] if selected_video and selected_video.get('summary') else "AI功能环境已就绪。"
    subject_hint = _build_video_subject_hint(video_id=video_id, video=selected_video, knowledge_segments=video_segments)
    if subject_hint and not selected_video.get('subject'):
        selected_video = _patch_user_video(_current_user_id(), video_id, subject=subject_hint) or selected_video
    
    # 生成更贴近学生视角的推荐问题
    recommended_questions = _build_recommended_questions(
        video_id,
        video_segments,
        video_title=title,
        subject_hint=subject_hint,
        video_summary=summary,
    )

    return render_template('video_detail.html', 
                           title=title, video_id=video_id, filepath=filepath,
                           summary=summary, knowledge_segments=video_segments, 
                           recommended_questions=recommended_questions,
                           subject_hint=subject_hint)

@app.route('/video_insights/<int:video_id>')
def video_insights(video_id):
    selected_video = _require_user_video(video_id)
    title = _get_video_display_title(selected_video, f"视频 {video_id} 分析")
    insights = _normalize_insight_view_data(_build_insight_view_data(video_id))
    return render_template('video_insights.html', title=title, video_id=video_id, insights=insights)

@app.route('/knowledge_graph/<int:video_id>')
def knowledge_graph(video_id):
    selected_video = _require_user_video(video_id)
    title = _get_video_display_title(selected_video, f"视频 {video_id} 知识图谱")
    return render_template('knowledge_graph.html', title=title, video_id=video_id)

@app.route('/api/knowledge_graph/<int:video_id>')
def api_knowledge_graph(video_id):
    """返回视频的知识关系网络数据（relation_network.json）供前端 D3 渲染"""
    _require_user_video(video_id)
    rn_path = os.path.join(VIDEO_DATA_DIR, str(video_id), 'relation_network.json')
    if not os.path.exists(rn_path):
        return jsonify({'nodes': [], 'all_edges': [], 'error': '尚未生成知识关系网络'})
    try:
        with open(rn_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logger.error(f'加载知识图谱数据失败: {e}')
        return jsonify({'nodes': [], 'all_edges': [], 'error': str(e)})


@app.route('/api/video_insights/<int:video_id>')
def api_video_insights(video_id):
    _require_user_video(video_id)
    return jsonify(_normalize_insight_view_data(_build_insight_view_data(video_id)))

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
        "analysis": eval_result['comment'],
        "evaluation": eval_result, "student_profile": profile, "learning_path": path
    })

@app.route('/api/ask', methods=['POST'])
def ask_ai():
    data = request.json or {}
    question = (data.get('question') or '').strip()
    video_id = int(data.get('video_id', 0) or 0)
    current_time = data.get('current_time')

    if not question:
        return jsonify({"answer": "请先输入你的问题。"})

    try:
        result = run_answer_question(video_id, question, current_time=current_time)
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
    debug = os.getenv('FLASK_DEBUG', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
    app.run(host=host, port=port, debug=debug)
