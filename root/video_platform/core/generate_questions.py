import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import dashscope
    from dashscope import Generation
except Exception:  # pragma: no cover - allow local static analysis without dashscope
    dashscope = None
    Generation = None


if pytesseract is not None:
    pytesseract.pytesseract_cmd = "/usr/bin/tesseract"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or "sk-24feda02d5524ed89a3ff3f5e0cee735"
if dashscope is not None:
    dashscope.api_key = DASHSCOPE_API_KEY

SEGMENT_FOLDER_PREFIX = "t"
_DEFAULT_PROJECT_ROOT = Path("/root/video_platform")
_LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = os.getenv(
    "VIDEO_PLATFORM_BASE_DIR",
    str(_DEFAULT_PROJECT_ROOT if _DEFAULT_PROJECT_ROOT.exists() else _LOCAL_PROJECT_ROOT),
)
VIDEO_DATA_DIR = os.getenv(
    "VIDEO_DATA_DIR",
    str((Path(PROJECT_ROOT) / "video_data")),
)

MIN_INTERVAL = 2.0
FRAME_SKIP = 2
MAX_EXTRACT_FRAMES = 20
CHANGE_THRESHOLD = 1000

SKIP_KEYWORDS = [
    "课程介绍",
    "本章导入",
    "学习目标",
    "章节导入",
    "章节过渡",
    "过渡",
    "复习衔接",
    "课程总结",
    "本章总结",
    "总结",
    "总结性",
    "提醒",
    "鼓励",
    "闲聊",
    "作业",
    "考试要求",
    "要求说明",
    "课堂组织",
    "导学",
    "学习建议",
    "预告",
]

NON_KNOWLEDGE_ASR_HINTS = [
    "大家好",
    "同学",
    "上午好",
    "我们今天",
    "这门课",
    "这部课程",
    "课程",
    "课时",
    "努力",
    "平时",
    "学习建议",
    "作业",
    "考试",
    "要求",
    "导入",
    "过渡",
    "总结",
]

KNOWLEDGE_KEYWORDS = [
    "定义",
    "概念",
    "原理",
    "公式",
    "定理",
    "步骤",
    "方法",
    "例题",
    "分析",
    "算法",
    "过程",
    "结论",
    "对比",
    "应用",
    "性质",
    "判定",
    "推导",
    "证明",
    "条件",
]


def setup_logging(video_id: Any) -> logging.Logger:
    log_dir = Path(PROJECT_ROOT) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"generate_questions_{video_id}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    file_handler = logging.FileHandler(log_dir / f"generate_questions_{video_id}.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s")
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def safe_json_load(file_path_or_content: Any, is_content: bool = False) -> Any:
    try:
        if is_content:
            content = str(file_path_or_content or "").strip().replace("```json", "").replace("```", "").strip()
            return json.loads(content) if content else {}

        file_path = Path(file_path_or_content)
        if not file_path.exists():
            return {}

        with file_path.open("r", encoding="utf-8-sig") as f:
            content = f.read().replace("\x00", "").strip()
        return json.loads(content) if content else {}
    except Exception:
        return {}


def get_video_duration(video_path: str) -> float:
    if cv2 is None:
        return 60.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 60.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames / fps


def fast_frame_diff(frame1: Any, frame2: Any) -> float:
    if cv2 is None or np is None:
        return 0.0
    f1 = cv2.cvtColor(cv2.resize(frame1, (320, 180)), cv2.COLOR_BGR2GRAY)
    f2 = cv2.cvtColor(cv2.resize(frame2, (320, 180)), cv2.COLOR_BGR2GRAY)
    return float(np.sum(cv2.absdiff(f1, f2)))


def extract_t_ocr_with_pytesseract(video_id: Any, video_data_dir: str = VIDEO_DATA_DIR) -> Dict[str, Any]:
    logger = setup_logging(video_id)
    if cv2 is None or pytesseract is None or Image is None:
        logger.warning("OCR 依赖缺失，跳过自动 OCR 抽取")
        return {}
    base_dir = Path(video_data_dir) / str(video_id)
    ocr_root_dir = base_dir / "ocr_segments"
    ocr_root_dir.mkdir(parents=True, exist_ok=True)

    video_path = None
    split_dir = base_dir / "split_videos"
    if split_dir.exists():
        candidates = sorted(split_dir.glob("segment_*.mp4"))
        if candidates:
            video_path = str(candidates[0])

    upload_dir = Path(PROJECT_ROOT) / "uploads"
    if video_path is None and upload_dir.exists():
        candidates = list(upload_dir.glob(f"*{video_id}*.mp4"))
        if candidates:
            video_path = str(candidates[0])

    if video_path is None or not os.path.exists(video_path):
        logger.error("未找到可用于 OCR 的视频文件: %s", video_id)
        return {}

    splits_path = base_dir / "final_knowledge_splits.json"
    knowledge_segments = safe_json_load(splits_path)
    if not isinstance(knowledge_segments, list) or not knowledge_segments:
        duration = get_video_duration(video_path)
        knowledge_segments = [
            {"start_time": 0.0, "end_time": duration, "knowledge_point": "整段视频"}
        ]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("无法打开视频进行 OCR: %s", video_path)
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ocr_results: Dict[str, Any] = {}

    for seg_idx, seg in enumerate(knowledge_segments, 1):
        seg_folder = f"{SEGMENT_FOLDER_PREFIX}{seg_idx}"
        seg_ocr_dir = ocr_root_dir / seg_folder
        seg_ocr_dir.mkdir(parents=True, exist_ok=True)

        seg_start = float(seg.get("start_time", 0.0) or 0.0)
        seg_end = float(seg.get("end_time", seg_start) or seg_start)
        seg_title = seg.get("knowledge_point") or seg.get("title") or f"知识点{seg_idx}"

        start_frame = int(seg_start * fps)
        end_frame = min(int(seg_end * fps), total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        seg_ocr = []
        prev_frame = None
        last_extract = seg_start - MIN_INTERVAL
        extracted = 0

        for frame_idx in range(start_frame, end_frame, FRAME_SKIP):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, curr = cap.read()
            if not ret or extracted >= MAX_EXTRACT_FRAMES:
                break

            curr_time = frame_idx / fps
            if curr_time - last_extract < MIN_INTERVAL:
                continue

            need = prev_frame is None or fast_frame_diff(prev_frame, curr) > CHANGE_THRESHOLD
            if not need:
                prev_frame = curr
                continue

            try:
                gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(
                    Image.fromarray(gray), lang="chi_sim+eng", config="--psm 6 --oem 3"
                ).strip()
                ocr_file = seg_ocr_dir / f"ocr_{round(curr_time, 1)}s.txt"
                with ocr_file.open("w", encoding="utf-8") as f:
                    f.write(text)
                seg_ocr.append(
                    {
                        "time": curr_time,
                        "ocr_text": text,
                        "file_path": str(ocr_file),
                        "segment_folder": seg_folder,
                    }
                )
                extracted += 1
                last_extract = curr_time
            except Exception as exc:
                logger.warning("%s OCR 失败: %s", seg_folder, str(exc)[:80])
            prev_frame = curr

        seg_sum = {
            "segment_folder": seg_folder,
            "title": seg_title,
            "start_time": seg_start,
            "end_time": seg_end,
            "ocr_frames": seg_ocr,
        }
        with (seg_ocr_dir / "ocr_summary.json").open("w", encoding="utf-8") as f:
            json.dump(seg_sum, f, ensure_ascii=False, indent=2)
        ocr_results[seg_folder] = seg_sum

    cap.release()
    global_sum = {
        "video_id": str(video_id),
        "segment_folders": list(ocr_results.keys()),
        "ocr_results": ocr_results,
    }
    with (ocr_root_dir / "global_ocr_summary.json").open("w", encoding="utf-8") as f:
        json.dump(global_sum, f, ensure_ascii=False, indent=2)
    return global_sum


def call_qwen_safely(messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
    logger = logging.getLogger("call_qwen_safely")
    if Generation is None:
        logger.warning("dashscope.Generation unavailable, skip model call")
        return ""
    try:
        response = Generation.call(
            model="qwen-plus",
            messages=messages,
            temperature=0.1,
            result_format="text",
            max_tokens=max_tokens,
            timeout=120,
        )
        if getattr(response, "status_code", None) != 200:
            return ""
        return ((response.output or {}).text or "").strip()
    except Exception as exc:
        logger.error("千问调用失败: %s", str(exc)[:120])
        return ""


def _truncate_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _extract_json_block(text: str) -> Dict[str, Any]:
    data = safe_json_load(text, is_content=True)
    if isinstance(data, dict):
        return data

    if not text:
        return {}

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return {}
    return safe_json_load(text[start : end + 1], is_content=True)


def _load_subvideo_summaries(base_dir: Path) -> Dict[str, Any]:
    data = safe_json_load(base_dir / "subvideo_summaries_all.json")
    if not isinstance(data, dict):
        return {}
    summaries = data.get("subvideo_summaries")
    return summaries if isinstance(summaries, dict) else {}


def _load_relation_network(base_dir: Path) -> Dict[str, Any]:
    data = safe_json_load(base_dir / "relation_network.json")
    return data if isinstance(data, dict) else {}


def _load_global_ocr(base_dir: Path) -> Dict[str, Any]:
    data = safe_json_load(base_dir / "ocr_segments" / "global_ocr_summary.json")
    return data if isinstance(data, dict) else {}


def _load_raw_asr_segments(base_dir: Path) -> List[Dict[str, Any]]:
    data = safe_json_load(base_dir / "raw_asr_segments.json")
    return data if isinstance(data, list) else []


def _guess_segment_name(segment_id: int, split: Dict[str, Any], split_dir: Path) -> str:
    explicit = split.get("segment_file") or split.get("subvideo_file")
    if explicit:
        return str(explicit)
    default_name = f"segment_{segment_id}.mp4"
    return default_name if (split_dir / default_name).exists() else default_name


def _slice_asr_by_time(asr_segments: List[Dict[str, Any]], start_time: float, end_time: float) -> str:
    texts: List[str] = []
    for segment in asr_segments:
        seg_start = float(segment.get("start", 0.0) or 0.0)
        seg_end = float(segment.get("end", seg_start) or seg_start)
        if seg_end < start_time or seg_start > end_time:
            continue
        text = str(segment.get("text") or "").strip()
        if text:
            texts.append(text)
    return " ".join(texts)


def _extract_ocr_text(ocr_data: Dict[str, Any], segment_folder: str) -> str:
    segment_block = (ocr_data.get("ocr_results") or {}).get(segment_folder, {})
    frames = segment_block.get("ocr_frames") or []
    texts = [str(frame.get("ocr_text") or "").strip() for frame in frames if str(frame.get("ocr_text") or "").strip()]
    return " ".join(texts)


def _collect_relation_text(relation_data: Dict[str, Any], segment_name: str) -> str:
    texts: List[str] = []
    for node in relation_data.get("nodes") or []:
        if str(node.get("video_name") or "").strip() != segment_name:
            continue
        texts.extend(
            [
                str(node.get("title") or "").strip(),
                " ".join(node.get("key_concepts") or []),
            ]
        )

    for edge in relation_data.get("all_edges") or []:
        edge_text = " ".join(
            str(edge.get(field) or "").strip()
            for field in ("source", "target", "relation", "relation_type", "reason", "description", "kg_concept_relation")
            if str(edge.get(field) or "").strip()
        )
        if segment_name in edge_text:
            texts.append(edge_text)

    return " ".join(item for item in texts if item)


def _load_segment_contexts(video_id: Any, video_data_dir: str = VIDEO_DATA_DIR) -> List[Dict[str, Any]]:
    logger = setup_logging(video_id)
    base_dir = Path(video_data_dir) / str(video_id)
    split_dir = base_dir / "split_videos"

    splits = safe_json_load(base_dir / "final_knowledge_splits.json")
    if not isinstance(splits, list):
        logger.warning("未找到 final_knowledge_splits.json，无法生成题目")
        return []

    summaries = _load_subvideo_summaries(base_dir)
    relation_data = _load_relation_network(base_dir)
    ocr_data = _load_global_ocr(base_dir)
    if not (ocr_data.get("ocr_results") or {}):
        logger.info("未找到全局 OCR 摘要，尝试自动抽取")
        ocr_data = extract_t_ocr_with_pytesseract(video_id, video_data_dir=video_data_dir)
    raw_asr_segments = _load_raw_asr_segments(base_dir)

    segment_contexts: List[Dict[str, Any]] = []
    for idx, split in enumerate(splits, 1):
        segment_folder = f"{SEGMENT_FOLDER_PREFIX}{idx}"
        segment_name = _guess_segment_name(idx, split, split_dir)
        summary_item = summaries.get(segment_name, {}) if isinstance(summaries, dict) else {}

        start_time = float(split.get("start_time", 0.0) or 0.0)
        end_time = float(split.get("end_time", start_time) or start_time)
        title = (
            str(split.get("knowledge_point") or "").strip()
            or str(summary_item.get("title") or "").strip()
            or f"知识点{idx}"
        )
        description = str(split.get("description") or "").strip()
        summary_text = " ".join(
            str(summary_item.get(key) or "").strip()
            for key in ("summary", "logic_flow", "details", "emphasis")
            if str(summary_item.get(key) or "").strip()
        )
        relation_text = _collect_relation_text(relation_data, segment_name)
        asr_text = _slice_asr_by_time(raw_asr_segments, start_time, end_time)
        ocr_text = _extract_ocr_text(ocr_data, segment_folder)

        segment_contexts.append(
            {
                "segment_id": idx,
                "segment_folder": segment_folder,
                "segment_name": segment_name,
                "segment_path": str(split_dir / segment_name),
                "title": title,
                "description": description,
                "start": start_time,
                "end": end_time,
                "summary": summary_text,
                "relation_text": relation_text,
                "key_concepts": summary_item.get("key_concepts") or [],
                "ocr_text": ocr_text,
                "asr_text": asr_text,
                "split_item": split,
                "summary_item": summary_item,
            }
        )

    return segment_contexts


def _normalize_question_items(questions: Any) -> List[Dict[str, str]]:
    if not isinstance(questions, list):
        return []

    normalized = []
    seen = set()
    for item in questions:
        if not isinstance(item, dict):
            continue
        q_type = str(item.get("type") or "").strip() or "考查题"
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question or not answer:
            continue
        dedupe_key = (q_type, question, answer)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append({"type": q_type, "question": question, "answer": answer})
    return normalized[:2]


def _empty_segment_result(segment: Dict[str, Any], skip_reason: str) -> Dict[str, Any]:
    return {
        "should_generate_questions": False,
        "skip_reason": skip_reason,
        "questions": [],
        "segment_id": segment["segment_id"],
        "segment_folder": segment["segment_folder"],
        "title": segment["title"],
        "start": segment["start"],
        "end": segment["end"],
    }


def _count_keyword_hits(text: str, keywords: List[str]) -> int:
    normalized = str(text or "").replace(" ", "")
    return sum(1 for kw in keywords if kw and kw in normalized)


def _heuristic_should_skip(segment: Dict[str, Any]) -> str:
    text_pool = " ".join(
        [
            segment["title"],
            segment["description"],
            segment["summary"],
            segment["relation_text"],
            segment["asr_text"],
            segment["ocr_text"],
        ]
    )
    title_text = str(segment["title"] or "")
    title_summary_text = " ".join(
        [
            segment["title"],
            segment["description"],
            segment["summary"],
        ]
    )

    skip_hits = _count_keyword_hits(text_pool, SKIP_KEYWORDS)
    title_only_skip_hits = _count_keyword_hits(title_text, SKIP_KEYWORDS)
    title_skip_hits = _count_keyword_hits(title_summary_text, SKIP_KEYWORDS)
    title_only_knowledge_hits = _count_keyword_hits(title_text, KNOWLEDGE_KEYWORDS)
    knowledge_hits = _count_keyword_hits(text_pool, KNOWLEDGE_KEYWORDS)
    title_knowledge_hits = _count_keyword_hits(title_summary_text, KNOWLEDGE_KEYWORDS)
    asr_skip_hits = _count_keyword_hits(segment["asr_text"], NON_KNOWLEDGE_ASR_HINTS)
    asr_len = len((segment["asr_text"] or "").strip())
    ocr_len = len((segment["ocr_text"] or "").strip())

    if title_only_skip_hits >= 1 and title_only_knowledge_hits == 0:
        return "课程介绍/导入过渡内容"
    if title_skip_hits >= 1 and title_knowledge_hits == 0:
        return "课程介绍/导入过渡内容"
    if title_skip_hits >= 2 and title_knowledge_hits <= 1:
        return "非知识性组织内容"
    if asr_skip_hits >= 3 and title_knowledge_hits == 0:
        return "课堂组织语言/学习提示"
    if skip_hits >= 2 and knowledge_hits == 0:
        return "课程介绍/过渡内容"
    if skip_hits >= 1 and asr_len < 180 and ocr_len < 40:
        return "非知识性组织语言"
    if knowledge_hits == 0 and asr_len < 120 and ocr_len < 30:
        return "知识密度不足"
    return ""


def _fallback_questions(segment: Dict[str, Any]) -> List[Dict[str, str]]:
    if _heuristic_should_skip(segment):
        return []

    title = segment["title"]
    description = segment["description"] or segment["summary"]
    if not description:
        return []

    guide = {
        "type": "引导题",
        "question": f"本片段围绕“{title}”主要讲清了什么核心内容？",
        "answer": _truncate_text(description, 48),
    }

    inspect = {
        "type": "考查题",
        "question": f"根据本片段，“{title}”最需要掌握的关键点是什么？",
        "answer": _truncate_text(segment["summary"] or description, 48),
    }
    return _normalize_question_items([guide, inspect])


def _build_messages_for_segment(video_id: Any, segment: Dict[str, Any], video_data_dir: str = VIDEO_DATA_DIR) -> List[Dict[str, str]]:
    base_dir = Path(video_data_dir) / str(video_id)
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "你是教学视频出题专家。"
                "请先判断当前片段是否包含适合教学提问的实质性知识内容。"
                "若该片段主要是课程介绍、导入、过渡、总结、提醒、作业说明、课堂组织语言或其他非知识性内容，请不要硬生成题目，而是返回不适合出题的判断结果。"
                "只有当片段中存在明确、可考察、可教学提问的知识点时，才生成题目。"
                "摘要、关系网、description、OCR、ASR 都只作为定位与辅助理解材料，不要机械地直接把摘要句子改写成题目。"
                "如果适合出题，题目必须围绕该片段本身，难度适中，不空泛、不重复、不脱离片段。"
                "只输出 JSON。"
            ),
        }
    ]

    final_llm_input = safe_json_load(base_dir / "final_llm_input.json")
    if final_llm_input:
        messages.append(
            {
                "role": "user",
                "content": f"全局摘要与关系网索引（仅作导航，不可机械照抄出题）:\n{json.dumps(final_llm_input, ensure_ascii=False)}",
            }
        )

    messages.append(
        {
            "role": "user",
            "content": (
                "当前片段信息如下：\n"
                f"- segment_id: {segment['segment_id']}\n"
                f"- segment_folder: {segment['segment_folder']}\n"
                f"- segment_file: {segment['segment_name']}\n"
                f"- title: {segment['title']}\n"
                f"- start: {segment['start']}\n"
                f"- end: {segment['end']}\n"
                f"- description: {segment['description'] or '无'}\n"
                f"- subvideo_summary: {_truncate_text(segment['summary'], 1800) or '无'}\n"
                f"- relation_graph_hint: {_truncate_text(segment['relation_text'], 1400) or '无'}\n"
                f"- key_concepts: {json.dumps(segment['key_concepts'], ensure_ascii=False)}\n"
                f"- ASR: {_truncate_text(segment['asr_text'], 3000) or '无'}\n"
                f"- OCR: {_truncate_text(segment['ocr_text'], 1200) or '无'}\n\n"
                "请先判断它属于哪一类：\n"
                "A. 适合出题的知识性内容，例如概念定义、原理讲解、公式/定理、方法步骤、例题分析、算法过程、关键结论、对比分析、应用场景。\n"
                "B. 不适合硬出题的非知识性内容，例如课程介绍、本章导入、学习目标说明、课堂过渡语、复习衔接语、作业/考试/要求说明、鼓励提醒闲聊、总结性套话、没有实质知识点的信息。\n\n"
                "如果判断为 B，请不要为了凑数量硬生成两道题，可以返回空题目列表。\n"
                "请严格输出以下 JSON 结构：\n"
                "{\n"
                '  "should_generate_questions": true,\n'
                '  "skip_reason": "",\n'
                '  "segment_id": 1,\n'
                '  "segment_folder": "t1",\n'
                '  "title": "片段标题",\n'
                '  "start": 0.0,\n'
                '  "end": 0.0,\n'
                '  "questions": [\n'
                '    {"type": "引导题", "question": "", "answer": ""},\n'
                '    {"type": "考查题", "question": "", "answer": ""}\n'
                "  ]\n"
                "}\n"
                "如果不适合出题，则 should_generate_questions=false，skip_reason 写明原因，questions 返回空数组。"
            ),
        }
    )
    return messages


def _generate_for_segment(video_id: Any, segment: Dict[str, Any], video_data_dir: str = VIDEO_DATA_DIR) -> Dict[str, Any]:
    logger = setup_logging(video_id)

    heuristic_skip = _heuristic_should_skip(segment)
    if heuristic_skip:
        logger.info("%s 命中启发式跳过: %s", segment["segment_folder"], heuristic_skip)
        return _empty_segment_result(segment, heuristic_skip)

    messages = _build_messages_for_segment(video_id, segment, video_data_dir=video_data_dir)
    response = call_qwen_safely(messages, max_tokens=2400)
    payload = _extract_json_block(response)

    if not payload:
        logger.warning("%s 模型返回不可解析，回退到启发式结果", segment["segment_folder"])
        fallback_questions = _fallback_questions(segment)
        if fallback_questions:
            return {
                "should_generate_questions": True,
                "skip_reason": "",
                "questions": fallback_questions,
                "segment_id": segment["segment_id"],
                "segment_folder": segment["segment_folder"],
                "title": segment["title"],
                "start": segment["start"],
                "end": segment["end"],
            }
        return _empty_segment_result(segment, "信息不足，模型返回异常")

    should_generate = bool(payload.get("should_generate_questions", True))
    skip_reason = str(payload.get("skip_reason") or "").strip()
    questions = _normalize_question_items(payload.get("questions"))

    result = {
        "should_generate_questions": should_generate,
        "skip_reason": skip_reason,
        "questions": questions if should_generate else [],
        "segment_id": segment["segment_id"],
        "segment_folder": segment["segment_folder"],
        "title": str(payload.get("title") or segment["title"]).strip() or segment["title"],
        "start": float(payload.get("start", segment["start"]) or segment["start"]),
        "end": float(payload.get("end", segment["end"]) or segment["end"]),
    }

    if not result["should_generate_questions"]:
        if not result["skip_reason"]:
            result["skip_reason"] = "模型判断为非知识性内容"
        result["questions"] = []
        return result

    if not result["questions"]:
        fallback_questions = _fallback_questions(segment)
        if fallback_questions:
            result["questions"] = fallback_questions
        else:
            result["should_generate_questions"] = False
            result["skip_reason"] = skip_reason or "知识点不够清晰，降级跳过"
            result["questions"] = []

    return result


def generate_questions_for_knowledge_points(video_id: Any, video_data_dir: str = VIDEO_DATA_DIR) -> List[Dict[str, Any]]:
    logger = setup_logging(video_id)
    logger.info("===== 开始生成题目（先判断是否适合出题） =====")

    segments = _load_segment_contexts(video_id, video_data_dir=video_data_dir)
    if not segments:
        logger.error("没有可用片段，终止生成")
        return []

    results: List[Dict[str, Any]] = []
    for segment in segments:
        logger.info("处理 %s | %s", segment["segment_folder"], segment["title"])
        result = _generate_for_segment(video_id, segment, video_data_dir=video_data_dir)
        results.append(result)
        if result["should_generate_questions"]:
            logger.info("%s 生成题目 %s 道", segment["segment_folder"], len(result["questions"]))
        else:
            logger.info("%s 跳过出题: %s", segment["segment_folder"], result["skip_reason"])

    out_path = Path(video_data_dir) / str(video_id) / "knowledge_questions.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    generated_count = sum(1 for item in results if item.get("should_generate_questions"))
    skipped_count = len(results) - generated_count
    logger.info("===== 生成完成：输出到 %s | 可出题=%s | 跳过=%s =====", out_path, generated_count, skipped_count)
    return results


if __name__ == "__main__":
    TARGET_VIDEO_ID = 65
    generate_questions_for_knowledge_points(TARGET_VIDEO_ID)
