# -*- coding: utf-8 -*-
import json
import uuid
from pathlib import Path
from config import DATA_DIR, UPLOAD_DIR

# 用户数据存储
USER_FILE = DATA_DIR / "users.json"

def init_db():
    """初始化数据文件"""
    if not USER_FILE.exists():
        with open(USER_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False)

# 用户相关操作
def add_user(username, password):
    """添加用户"""
    init_db()
    with open(USER_FILE, "r", encoding="utf-8") as f:
        users = json.load(f)
    if username in users:
        return False
    users[username] = {"password": password, "history": []}
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    return True

def verify_user(username, password):
    """验证用户"""
    init_db()
    with open(USER_FILE, "r", encoding="utf-8") as f:
        users = json.load(f)
    return username in users and users[username]["password"] == password

def add_video_to_history(username, video_id, video_name):
    """添加视频到用户历史"""
    init_db()
    with open(USER_FILE, "r", encoding="utf-8") as f:
        users = json.load(f)
    if username not in users:
        return False
    # 去重
    history = [v for v in users[username]["history"] if v["id"] != video_id]
    history.insert(0, {"id": video_id, "name": video_name})
    users[username]["history"] = history
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    return True

def get_user_history(username):
    """获取用户学习历史"""
    init_db()
    with open(USER_FILE, "r", encoding="utf-8") as f:
        users = json.load(f)
    return users.get(username, {}).get("history", [])

# 视频相关操作
def create_video_record(video_file):
    """创建视频记录"""
    video_id = str(uuid.uuid4())[:8]
    video_name = video_file.filename
    # 保存视频文件
    video_path = UPLOAD_DIR / video_name
    video_file.save(video_path)
    # 创建视频数据目录
    video_data_dir = DATA_DIR / video_id
    video_data_dir.mkdir(exist_ok=True)
    # 初始化视频记录
    record = {
        "id": video_id,
        "name": video_name,
        "file_path": str(video_path),
        "data_dir": str(video_data_dir),
        "progress": 0.0,
        "summary": "",
        "sub_videos": []
    }
    # 保存记录
    record_file = video_data_dir / "video_record.json"
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return video_id, record

def save_video_progress(video_id, progress):
    """保存视频播放进度"""
    video_data_dir = DATA_DIR / video_id
    record_file = video_data_dir / "video_record.json"
    if not record_file.exists():
        return False
    with open(record_file, "r", encoding="utf-8") as f:
        record = json.load(f)
    record["progress"] = progress
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return True

def get_video_record(video_id):
    """获取视频记录"""
    video_data_dir = DATA_DIR / video_id
    record_file = video_data_dir / "video_record.json"
    if not record_file.exists():
        return None
    with open(record_file, "r", encoding="utf-8") as f:
        return json.load(f)

def save_sub_videos(video_id, sub_videos):
    """保存短视频列表"""
    video_data_dir = DATA_DIR / video_id
    record_file = video_data_dir / "video_record.json"
    if not record_file.exists():
        return False
    with open(record_file, "r", encoding="utf-8") as f:
        record = json.load(f)
    record["sub_videos"] = sub_videos
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return True

def save_video_summary(video_id, summary):
    """保存视频摘要"""
    video_data_dir = DATA_DIR / video_id
    record_file = video_data_dir / "video_record.json"
    if not record_file.exists():
        return False
    with open(record_file, "r", encoding="utf-8") as f:
        record = json.load(f)
    record["summary"] = summary
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    # 同时保存单独的摘要文件
    summary_file = video_data_dir / "global_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    return True