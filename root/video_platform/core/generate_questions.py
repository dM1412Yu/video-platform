import os
import json
import logging
import cv2
import pytesseract
import numpy as np
from PIL import Image
from pathlib import Path
import dashscope
from dashscope import Generation

# ========================== 基础配置 ==========================
pytesseract.pytesseract_cmd = '/usr/bin/tesseract'
DASHSCOPE_API_KEY = "sk-24feda02d5524ed89a3ff3f5e0cee735"
dashscope.api_key = DASHSCOPE_API_KEY

SEGMENT_FOLDER_PREFIX = "t"
VIDEO_DATA_DIR = "/root/video_platform/video_data"
PROJECT_ROOT = "/root/video_platform"

# OCR 常量（核心修改：MIN_INTERVAL改为2.0，实现最多2秒一帧）
MIN_INTERVAL = 2.0  # 仅修改这一行，从1.0→2.0
FRAME_SKIP = 2
MAX_EXTRACT_FRAMES = 20
CHANGE_THRESHOLD = 1000

# ========================== 日志 ==========================
def setup_logging(video_id):
    log_dir = Path(PROJECT_ROOT) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"generate_questions_{video_id}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    file_handler = logging.FileHandler(log_dir / f"generate_questions_{video_id}.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# ========================== 工具函数 ==========================
def safe_json_load(file_path_or_content, is_content=False):
    try:
        if is_content:
            content = file_path_or_content.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(content) if content else {}
        else:
            if not os.path.exists(file_path_or_content):
                logging.warning(f"JSON不存在：{file_path_or_content}")
                return {}
            with open(file_path_or_content, encoding="utf-8-sig") as f:
                content = f.read().replace('\x00', '').strip()
                return json.loads(content) if content else {}
    except Exception as e:
        logging.error(f"JSON解析失败：{str(e)[:80]}")
        return {}

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 60.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames / fps

def fast_frame_diff(frame1, frame2):
    f1 = cv2.cvtColor(cv2.resize(frame1, (320, 180)), cv2.COLOR_BGR2GRAY)
    f2 = cv2.cvtColor(cv2.resize(frame2, (320, 180)), cv2.COLOR_BGR2GRAY)
    return np.sum(cv2.absdiff(f1, f2))

# ========================== OCR 提取（2秒一帧 + 相似帧过滤） ==========================
def extract_t_ocr_with_pytesseract(video_id, video_data_dir=VIDEO_DATA_DIR):
    logger = setup_logging(video_id)
    base_dir = Path(video_data_dir) / str(video_id)
    ocr_root_dir = base_dir / "ocr_segments"
    ocr_root_dir.mkdir(parents=True, exist_ok=True)

    # 查找视频文件
    video_path = None
    upload_dir = Path(PROJECT_ROOT) / "uploads"
    if upload_dir.exists():
        candidates = list(upload_dir.glob(f"*{video_id}*.mp4"))
        if candidates:
            video_path = str(candidates[0])
    if video_path is None:
        split_dir = base_dir / "split_videos"
        if split_dir.exists():
            candidates = list(split_dir.glob("segment_*.mp4"))
            if candidates:
                video_path = str(candidates[0])
    if video_path is None or not os.path.exists(video_path):
        logger.error(f"❌ 未找到视频 {video_id}")
        return {}

    # 加载知识点分段数据（提取knowledge_point）
    splits_path = base_dir / "final_knowledge_splits.json"
    knowledge_segments = safe_json_load(splits_path)
    if not isinstance(knowledge_segments, list) or not knowledge_segments:
        duration = get_video_duration(video_path)
        knowledge_segments = [{"start_time": 0.0, "end_time": duration, "knowledge_point": "全视频知识点"}]
        logger.warning("⚠️ 无分段文件，使用全视频分段")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ 无法打开视频：{video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ocr_results = {}

    for seg_idx, seg in enumerate(knowledge_segments):
        seg_folder = f"{SEGMENT_FOLDER_PREFIX}{seg_idx+1}"
        seg_ocr_dir = ocr_root_dir / seg_folder
        seg_ocr_dir.mkdir(parents=True, exist_ok=True)

        seg_start = float(seg["start_time"])
        seg_end = float(seg["end_time"])
        # 优先提取knowledge_point作为知识点名称
        seg_title = seg.get("knowledge_point", seg.get("title", f"知识点{seg_idx+1}"))

        start_frame = int(seg_start * fps)
        end_frame = min(int(seg_end * fps), total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        seg_ocr = []
        prev_frame = None
        last_extract = seg_start - MIN_INTERVAL  # 使用2秒间隔
        extracted = 0

        for frame_idx in range(start_frame, end_frame, FRAME_SKIP):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, curr = cap.read()
            if not ret or extracted >= MAX_EXTRACT_FRAMES:
                break

            curr_time = frame_idx / fps
            # 确保至少间隔2秒才提取下一帧
            if curr_time - last_extract < MIN_INTERVAL:
                continue

            # 相似帧过滤：仅帧差大于阈值才提取
            need = False
            if prev_frame is None:
                need = True
            else:
                if fast_frame_diff(prev_frame, curr) > CHANGE_THRESHOLD:
                    need = True

            if need:
                try:
                    gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(
                        Image.fromarray(gray), lang='chi_sim+eng', config='--psm 6 --oem 3'
                    ).strip()

                    ocr_file = seg_ocr_dir / f"ocr_{round(curr_time,1)}s.txt"
                    with open(ocr_file, "w", encoding="utf-8") as f:
                        f.write(text)

                    seg_ocr.append({
                        "time": curr_time,
                        "ocr_text": text,
                        "file_path": str(ocr_file),
                        "segment_folder": seg_folder
                    })
                    extracted += 1
                    last_extract = curr_time
                except Exception as e:
                    logger.warning(f"{seg_folder} 帧{frame_idx} OCR失败：{str(e)[:50]}")
            prev_frame = curr

        # 兜底提取（5秒间隔）
        if extracted == 0:
            logger.warning(f"{seg_folder} 无变化帧，5秒兜底提取")
            extracted = 0
            last_extract = seg_start - 5
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_idx in range(start_frame, end_frame, int(fps * 5)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, curr = cap.read()
                if not ret or extracted >= MAX_EXTRACT_FRAMES:
                    break
                curr_time = frame_idx / fps
                if curr_time - last_extract < 5:
                    continue
                try:
                    gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(
                        Image.fromarray(gray), lang='chi_sim+eng', config='--psm 6 --oem 3'
                    ).strip()
                    ocr_file = seg_ocr_dir / f"ocr_{round(curr_time,1)}s.txt"
                    with open(ocr_file, "w", encoding="utf-8") as f:
                        f.write(text)
                    seg_ocr.append({
                        "time": curr_time,
                        "ocr_text": text,
                        "file_path": str(ocr_file),
                        "segment_folder": seg_folder
                    })
                    extracted += 1
                    last_extract = curr_time
                except Exception as e:
                    logger.warning(f"{seg_folder} 兜底帧{frame_idx}失败：{str(e)[:50]}")

        # 保存分段OCR结果
        seg_sum = {
            "segment_folder": seg_folder,
            "title": seg_title,
            "start_time": seg_start,
            "end_time": seg_end,
            "ocr_frames": seg_ocr
        }
        with open(seg_ocr_dir / "ocr_summary.json", "w", encoding="utf-8") as f:
            json.dump(seg_sum, f, ensure_ascii=False, indent=2)
        ocr_results[seg_folder] = seg_sum
        logger.info(f"{seg_folder} {seg_title} | 提取帧：{extracted}")

    cap.release()
    # 保存全局OCR摘要
    global_sum = {
        "video_id": video_id,
        "segment_folders": list(ocr_results.keys()),
        "ocr_results": ocr_results
    }
    with open(ocr_root_dir / "global_ocr_summary.json", "w", encoding="utf-8") as f:
        json.dump(global_sum, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ OCR 完成（2秒一帧 + 相似帧过滤）")
    return global_sum

# ========================== 千问调用 ==========================
def call_qwen_safely(messages):
    logger = logging.getLogger("call_qwen_safely")
    try:
        response = Generation.call(
            model="qwen-plus", 
            messages=messages,
            temperature=0.1,
            result_format="text",
            max_tokens=2048,
            timeout=120
        )
        return response.output.text.strip() if response.status_code == 200 else ""
    except Exception as e:
        logger.error(f"千问调用失败：{str(e)[:100]}")
        return ""

# ========================== 上下文初始化 ==========================
def init_base_context(video_id):
    logger = setup_logging(video_id)
    system_prompt = """你是出题老师，需结合以下视频核心数据为每个知识点分段生成题目：
1. 每段生成2题：1道引导题（启发思考）+ 1道考查题（检验掌握）；
2. 答案字数≤50字，必须准确贴合视频内容；
3. 仅返回JSON格式，无任何多余文字、注释或说明；
4. 优先参考矫正后的语音内容，其次参考结构化摘要/关系网，最后参考OCR文字。"""
    
    messages = [{"role": "system", "content": system_prompt}]
    base_dir = Path(VIDEO_DATA_DIR) / str(video_id)

    # 传入final_llm_input.json
    llm_input_path = base_dir / "final_llm_input.json"
    if os.path.exists(llm_input_path):
        llm_input_data = safe_json_load(llm_input_path)
        messages.append({
            "role": "user",
            "content": f"【视频结构化摘要+关系网数据】：{json.dumps(llm_input_data, ensure_ascii=False)}"
        })

    # 传入矫正ASR
    asr_path = base_dir / "corrected_asr.txt"
    if os.path.exists(asr_path):
        with open(asr_path, "r", encoding="utf-8") as f:
            asr_text = f.read().strip()
        messages.append({
            "role": "user",
            "content": f"【全局矫正视频语音内容】：{asr_text}"
        })

    return messages

# ========================== 读取分段（提取knowledge_point） ==========================
def load_all_t_segments_ocr(video_id):
    logger = setup_logging(video_id)
    base_dir = Path(VIDEO_DATA_DIR) / str(video_id)
    ocr_path = base_dir / "ocr_segments" / "global_ocr_summary.json"

    ocr_global = safe_json_load(ocr_path)
    if not isinstance(ocr_global, dict) or not ocr_global.get("ocr_results"):
        logger.warning("⚠️ OCR不存在，自动提取")
        ocr_global = extract_t_ocr_with_pytesseract(video_id)
        if not isinstance(ocr_global, dict) or not ocr_global.get("ocr_results"):
            logger.error("❌ OCR提取失败")
            return []

    # 构建knowledge_point映射表
    splits_path = base_dir / "final_knowledge_splits.json"
    knowledge_splits = safe_json_load(splits_path)
    title_map = {}
    for seg_idx, seg in enumerate(knowledge_splits):
        seg_folder = f"{SEGMENT_FOLDER_PREFIX}{seg_idx+1}"
        title_map[seg_folder] = seg.get("knowledge_point", seg.get("title", seg_folder))

    segments = []
    ocr_results = ocr_global.get("ocr_results", {})
    for seg_folder, info in ocr_results.items():
        if seg_folder.startswith(SEGMENT_FOLDER_PREFIX):
            ocr_frames = info.get("ocr_frames", [])
            all_text = "\n".join([
                f.get("ocr_text", "").strip() for f in ocr_frames
                if f.get("ocr_text","").strip()
            ])

            specific_title = title_map.get(seg_folder, info.get("title", f"知识点{seg_folder[1:]}"))

            segments.append({
                "segment_folder": seg_folder,
                "title": specific_title,
                "start_time": float(info.get("start_time", 0.0)),
                "end_time": float(info.get("end_time", 0.0)),
                "all_ocr_text": all_text
            })
    logger.info(f"✅ 读取分段：{len(segments)} 个（已使用knowledge_point名称）")
    return segments

# ========================== 生成题目 ==========================
def generate_questions_for_knowledge_points(video_id, video_data_dir=VIDEO_DATA_DIR):
    logger = setup_logging(video_id)
    logger.info("===== 开始生成题目 =====")

    ctx = init_base_context(video_id)
    segments = load_all_t_segments_ocr(video_id)

    if not segments:
        logger.error("❌ 无有效分段，终止")
        return []

    questions = []
    for seg in segments:
        sf = seg["segment_folder"]
        logger.info(f"➡️ 处理 {sf} | 标题：{seg['title']}")

        prompt = f"""
### 当前知识点分段信息
标题：{seg['title']}
时间范围：{seg['start_time']}s - {seg['end_time']}s
画面识别文字：
{seg['all_ocr_text'] if seg['all_ocr_text'] else '【无画面文字】'}

### 出题要求
1. 结合已传入的「视频结构化摘要/关系网」和「矫正ASR语音」，为该分段生成2道题；
2. 引导题：启发对本知识点的思考（如“本知识点解决了什么问题？”）；
3. 考查题：检验对本知识点的掌握（如“本知识点的核心步骤是什么？”）；
4. 答案必须≤50字，准确贴合视频内容，不编造未提及的信息；
5. 严格按以下JSON格式输出，无任何多余内容：
{{
  "title":"{seg['title']}",
  "start":{seg['start_time']},
  "end":{seg['end_time']},
  "segment_folder":"{sf}",
  "segment_id":{int(sf[1:]) if sf[1:].isdigit() else 1},
  "questions":[
    {{"type":"引导题","question":"","answer":""}},
    {{"type":"考查题","question":"","answer":""}}
  ]
}}
""".strip()

        ctx.append({"role": "user", "content": prompt})
        res = call_qwen_safely(ctx)

        if res:
            obj = safe_json_load(res, is_content=True)
            obj.update({
                "start": seg['start_time'],
                "end": seg['end_time'],
                "segment_folder": sf,
                "segment_id": int(sf[1:]) if sf[1:].isdigit() else 1,
                "title": seg['title']
            })
            questions.append(obj)
            logger.info(f"✅ {sf} AI出题成功")
            ctx.append({"role": "assistant", "content": res})
        else:
            fallback = {
                "title": seg['title'],
                "start": seg['start_time'],
                "end": seg['end_time'],
                "segment_folder": sf,
                "segment_id": int(sf[1:]) if sf[1:].isdigit() else 1,
                "questions": [
                    {"type": "引导题", "question": f"{seg['title']}在计算机网络中的作用是什么？", "answer": "掌握本段核心概念，理解其在网络中的应用场景"},
                    {"type": "考查题", "question": f"{seg['title']}的关键实现步骤有哪些？", "answer": "结合视频内容，列举该知识点的核心实现步骤"}
                ]
            }
            questions.append(fallback)
            logger.warning(f"⚠️ {sf} 使用兜底题（模型返回为空）")

    # 保存题目
    out_path = Path(video_data_dir) / str(video_id) / "knowledge_questions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    logger.info(f"===== 生成完成！题目已保存到：{out_path} =====")
    return questions

# ========================== 运行 ==========================
if __name__ == "__main__":
    TARGET_VIDEO_ID = 65
    generate_questions_for_knowledge_points(TARGET_VIDEO_ID)