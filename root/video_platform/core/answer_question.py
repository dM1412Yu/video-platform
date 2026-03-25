import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import dashscope
    from dashscope import Generation, MultiModalConversation
except Exception:  # pragma: no cover - allow local analysis without deps
    dashscope = None
    Generation = None
    MultiModalConversation = None


DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or "sk-24feda02d5524ed89a3ff3f5e0cee735"
if dashscope is not None and DASHSCOPE_API_KEY:
    dashscope.api_key = DASHSCOPE_API_KEY

_DEFAULT_VIDEO_DATA_DIR = Path("/root/video_platform/video_data")
_LOCAL_VIDEO_DATA_DIR = Path(__file__).resolve().parents[1] / "video_data"
VIDEO_DATA_DIR = os.getenv(
    "VIDEO_DATA_DIR",
    str(_DEFAULT_VIDEO_DATA_DIR if _DEFAULT_VIDEO_DATA_DIR.exists() else _LOCAL_VIDEO_DATA_DIR),
)
DEFAULT_TEXT_MODEL = os.getenv("QA_TEXT_MODEL", "qwen-plus")
DEFAULT_MM_MODEL = os.getenv("QA_MULTIMODAL_MODEL", "qwen-vl-max")
MAX_CANDIDATES = int(os.getenv("QA_MAX_CANDIDATES", "3"))

QUESTION_TYPE_LOCAL = "local"
QUESTION_TYPE_GLOBAL = "global"
QUESTION_TYPE_VISUAL = "visual_detail"

ANSWER_TYPE_DIRECT = "direct"
ANSWER_TYPE_INFER = "infer"
ANSWER_TYPE_INSUFFICIENT = "insufficient"
VALID_ANSWER_TYPES = {ANSWER_TYPE_DIRECT, ANSWER_TYPE_INFER, ANSWER_TYPE_INSUFFICIENT}

QUESTION_CLASSIFICATION_RULE = "请先判断用户问题属于：局部知识点问题、整体视频问题、画面细节问题。"
TIME_ANCHOR_RULES = (
    "如果用户问题包含明确时间点，请优先依据时间点定位相关片段，而不是先进行语义匹配。"
    "如果用户使用“现在、这一页、这个画面、当前”等表达，只有在已知 current_time 或其他明确时间锚点时才能定位；否则不要硬回答，应提示用户提供当前播放时间。"
)
LOCAL_PROMPT_RULE = (
    "如果是局部知识点问题，请把摘要、关系网、知识点描述视为导航索引，用于定位最相关的子视频；"
    "不要仅依据这些摘要性信息直接作答。最终答案应优先基于相关子视频内容本身。"
)
VISUAL_PROMPT_RULE = (
    "如果是画面细节问题，请优先定位与该画面内容最相关的子视频、时间段、OCR文本或关键帧附近内容，"
    "再基于画面、OCR、ASR和相关片段内容回答。不要只根据摘要猜测画面内容。"
    "如果无法可靠定位到具体画面证据，请明确说明证据不足。"
)
DEFAULT_PROMPT_RULE = "请把摘要性信息视为导航索引，最终答案优先基于当前候选片段的直接证据。"
GLOBAL_PROMPT_RULE = (
    "如果是整体视频问题，请不要只依据单个子视频作答，也不要把某一个局部片段当作整段视频的全部内容。"
    "你必须整合所有子视频摘要、知识点标题、知识点关系网以及必要的全局信息，对完整视频进行总体总结，"
    "并尽量体现视频的整体结构和主线脉络。"
)

INSUFFICIENT_ANSWER_MM = "当前未能稳定完成对子视频的多模态分析，且现有 ASR/OCR 证据不足以给出可靠答案。"
INSUFFICIENT_EVIDENCE = ["未获得足够稳定的视频直接证据"]
INSUFFICIENT_REASON_MM = "多模态调用失败后退化到文本证据合成"
INSUFFICIENT_ANSWER_BY_TIME = "已按时间点定位到相关片段，但当前没有足够稳定的直接证据生成可靠回答。"
INSUFFICIENT_ANSWER_BY_CURRENT = "已按当前播放时间定位到相关片段，但现有 OCR/ASR/画面证据仍不足以严谨确定答案。"
INSUFFICIENT_ANSWER_BY_CANDIDATE = "已定位到相关子视频，但当前没有足够稳定的直接证据生成可靠回答。"

GLOBAL_QUESTION_HINTS = [
    "这个视频讲了什么",
    "这节课讲了什么",
    "整节课",
    "整个视频",
    "整体视频",
    "整体结构",
    "主要内容",
    "主要讲了哪些",
    "主要讲了什么",
    "总体总结",
    "整体总结",
    "知识框架",
    "核心脉络",
    "主线",
    "整体内容",
    "总体内容",
    "主要讲哪些部分",
    "讲了哪些部分",
    "这节课主要",
    "这个视频主要",
    "整体上讲了什么",
]

VISUAL_QUESTION_HINTS = [
    "画面里",
    "画面中的",
    "板书上",
    "板书上的",
    "黑板上",
    "这一页",
    "这一屏",
    "这个图",
    "这张图",
    "这个公式",
    "这个式子",
    "这个式子是",
    "这个式子什么意思",
    "板书",
    "ppt",
    "PPT",
    "屏幕上",
    "图里",
    "页上",
    "这一页ppt",
    "这一页ppt上",
]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("answer_question")


logger = setup_logging()


def _looks_like_mojibake(text: str) -> bool:
    if not text:
        return False
    markers = ("锛", "銆", "鈥", "鈫", "鍦", "鐨", "鏄", "鍙", "鍚", "寮", "鎴", "澶", "璇")
    hit_count = sum(text.count(marker) for marker in markers)
    return hit_count >= 2


def _repair_mojibake_text(text: str) -> str:
    if not isinstance(text, str) or not _looks_like_mojibake(text):
        return text

    candidates = [text]
    for src_enc, dst_enc in (("gbk", "utf-8"), ("gb18030", "utf-8")):
        try:
            candidates.append(text.encode(src_enc).decode(dst_enc))
        except Exception:
            continue

    def _score(candidate_text: str) -> tuple:
        markers = sum(candidate_text.count(marker) for marker in ("锛", "銆", "鈥", "鈫", "鍦", "鐨", "鏄", "鍙"))
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", candidate_text))
        return (-markers, chinese_chars)

    candidates = sorted(set(candidates), key=_score, reverse=True)
    return candidates[0]


def _repair_loaded_data(data: Any) -> Any:
    if isinstance(data, str):
        return _repair_mojibake_text(data)
    if isinstance(data, list):
        return [_repair_loaded_data(item) for item in data]
    if isinstance(data, dict):
        return {
            _repair_loaded_data(key): _repair_loaded_data(value)
            for key, value in data.items()
        }
    return data


def load_file_content(file_path, is_json=False):
    file_path = Path(file_path)
    if not file_path.exists():
        return {} if is_json else ""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read().replace("\x00", "").strip()
        parsed = json.loads(content) if is_json else content
        return _repair_loaded_data(parsed)
    except Exception as exc:
        logger.error("加载失败 %s: %s", file_path, str(exc)[:120])
        return {} if is_json else ""


def _tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(list(part))
            if len(part) >= 2:
                tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def _normalize_text(text: str) -> str:
    return " ".join(_tokenize_text(text))


def _truncate_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _clamp_confidence(value: Any, default: float = 0.0) -> float:
    try:
        confidence = float(value if value is not None and value != "" else default)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def _normalize_evidence_list(evidence: Any) -> List[str]:
    if not isinstance(evidence, list):
        evidence = [str(evidence).strip()] if evidence else []
    return [str(item).strip() for item in evidence if str(item).strip()]


def _get_type_specific_rule(question_type: str, for_global: bool = False) -> str:
    if for_global:
        return GLOBAL_PROMPT_RULE
    mapping = {
        QUESTION_TYPE_LOCAL: LOCAL_PROMPT_RULE,
        QUESTION_TYPE_VISUAL: VISUAL_PROMPT_RULE,
    }
    return mapping.get(question_type, DEFAULT_PROMPT_RULE)


def _build_answer_json_schema(
    matched_subvideo: str = "segment_x.mp4",
    answer_label: str = "回答内容",
    evidence_labels: Optional[List[str]] = None,
    reason_label: str = "简短说明",
) -> str:
    evidence_labels = evidence_labels or ["证据1", "证据2"]
    schema = {
        "is_video_related": True,
        "matched_subvideo": matched_subvideo,
        "answer": answer_label,
        "evidence": evidence_labels,
        "answer_type": f"{ANSWER_TYPE_DIRECT}|{ANSWER_TYPE_INFER}|{ANSWER_TYPE_INSUFFICIENT}",
        "confidence": 0.0,
        "reason": reason_label,
    }
    return json.dumps(schema, ensure_ascii=False, indent=2)


def _apply_insufficient_defaults(
    payload: Dict[str, Any],
    *,
    answer_text: str,
    confidence_cap: float = 0.35,
    evidence: Optional[List[str]] = None,
    reason: str = "",
) -> Dict[str, Any]:
    if not payload.get("answer"):
        payload["answer"] = answer_text
    if not payload.get("evidence"):
        payload["evidence"] = list(evidence or INSUFFICIENT_EVIDENCE)
    if not payload.get("reason") and reason:
        payload["reason"] = reason
    if payload.get("answer_type") == ANSWER_TYPE_INSUFFICIENT:
        payload["confidence"] = min(_clamp_confidence(payload.get("confidence")), confidence_cap)
    return payload


def _build_single_candidate_summary(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "segment_name": candidate["segment_name"],
        "title": candidate["title"],
        "recall_score": None,
        "score_breakdown": {},
    }


def _answer_with_anchored_candidate(
    result: Dict[str, Any],
    question: str,
    anchored_candidate: Dict[str, Any],
    question_type: str,
    insufficient_answer_text: str,
) -> Dict[str, Any]:
    result["question_type"] = question_type
    result["candidates"] = [_build_single_candidate_summary(anchored_candidate)]

    answer_payload = _analyze_candidate_with_multimodal(
        question,
        anchored_candidate,
        [anchored_candidate],
        question_type=question_type,
    )
    if not answer_payload.get("answer"):
        answer_payload = _apply_insufficient_defaults(
            answer_payload,
            answer_text=insufficient_answer_text,
            confidence_cap=0.3,
        )

    result.update(answer_payload)
    result["is_video_related"] = bool(answer_payload.get("is_video_related", True))
    return result


def _call_text_model(messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 1800) -> str:
    if Generation is None:
        logger.warning("dashscope.Generation unavailable, skip text model call")
        return ""
    try:
        response = Generation.call(
            model=DEFAULT_TEXT_MODEL,
            messages=messages,
            temperature=temperature,
            result_format="text",
            max_tokens=max_tokens,
        )
        if getattr(response, "status_code", None) != 200:
            logger.warning("text model status != 200: %s", getattr(response, "code", "unknown"))
            return ""
        return ((response.output or {}).text or "").strip()
    except Exception as exc:
        logger.error("文本模型调用失败: %s", str(exc)[:200])
        return ""


def _call_multimodal_model(video_path: Path, prompt: str) -> str:
    if MultiModalConversation is None:
        logger.warning("dashscope.MultiModalConversation unavailable, skip multimodal call")
        return ""
    if not video_path.exists():
        logger.warning("candidate video does not exist: %s", video_path)
        return ""

    file_uri = video_path.as_uri() if video_path.is_absolute() else str(video_path)
    try:
        response = MultiModalConversation.call(
            model=DEFAULT_MM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": [{"text": "你是严谨的视频问答助手。"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"video": file_uri},
                        {"text": prompt},
                    ],
                },
            ],
        )
        if getattr(response, "status_code", None) != 200:
            logger.warning("multimodal status != 200: %s", getattr(response, "code", "unknown"))
            return ""

        output = getattr(response, "output", None)
        if hasattr(output, "choices"):
            choices = output.choices or []
        else:
            choices = (output or {}).get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else getattr(choices[0], "message", {})
        content = message.get("content", []) if isinstance(message, dict) else getattr(message, "content", [])
        text_parts = []
        for item in content or []:
            if isinstance(item, dict) and item.get("text"):
                text_parts.append(item["text"])
        return "\n".join(text_parts).strip()
    except Exception as exc:
        logger.error("多模态模型调用失败: %s", str(exc)[:200])
        return ""


def _load_knowledge_splits(base_dir: Path) -> List[Dict[str, Any]]:
    splits = load_file_content(base_dir / "final_knowledge_splits.json", is_json=True)
    return splits if isinstance(splits, list) else []


def _load_subvideo_summaries(base_dir: Path) -> Dict[str, Any]:
    data = load_file_content(base_dir / "subvideo_summaries_all.json", is_json=True)
    if not data:
        return {}
    if isinstance(data, dict):
        return data

    # legacy format
    if isinstance(data, list):
        mapping = {}
        for index, item in enumerate(data, 1):
            segment_name = item.get("subvideo_id") or f"segment_{index}.mp4"
            mapping[segment_name] = item
        return {"subvideo_summaries": mapping}
    return {}


def _load_relation_network(base_dir: Path) -> Dict[str, Any]:
    data = load_file_content(base_dir / "relation_network.json", is_json=True)
    return data if isinstance(data, dict) else {}


def _load_ocr_summary(base_dir: Path) -> Dict[str, Any]:
    data = load_file_content(base_dir / "ocr_segments" / "global_ocr_summary.json", is_json=True)
    return data if isinstance(data, dict) else {}


def _load_raw_asr(base_dir: Path) -> List[Dict[str, Any]]:
    data = load_file_content(base_dir / "raw_asr_segments.json", is_json=True)
    return data if isinstance(data, list) else []


def _load_corrected_asr(base_dir: Path) -> str:
    data = load_file_content(base_dir / "corrected_asr.txt", is_json=False)
    return data if isinstance(data, str) else ""


def _load_final_llm_input(base_dir: Path) -> Dict[str, Any]:
    data = load_file_content(base_dir / "final_llm_input.json", is_json=True)
    return data if isinstance(data, dict) else {}


def _guess_segment_name(index: int, split: Dict[str, Any], split_dir: Path) -> str:
    explicit = split.get("segment_file") or split.get("subvideo_file")
    if explicit:
        return explicit
    expected = f"segment_{index}.mp4"
    if (split_dir / expected).exists():
        return expected
    return expected


def _slice_asr_by_time(asr_segments: List[Dict[str, Any]], start_time: float, end_time: float) -> str:
    texts = []
    for segment in asr_segments:
        seg_start = float(segment.get("start", 0.0) or 0.0)
        seg_end = float(segment.get("end", seg_start) or seg_start)
        if seg_end < start_time or seg_start > end_time:
            continue
        text = (segment.get("text") or "").strip()
        if text:
            texts.append(text)
    return " ".join(texts)


def _collect_asr_segments_by_time(asr_segments: List[Dict[str, Any]], start_time: float, end_time: float) -> List[Dict[str, Any]]:
    matched = []
    for segment in asr_segments:
        seg_start = float(segment.get("start", 0.0) or 0.0)
        seg_end = float(segment.get("end", seg_start) or seg_start)
        if seg_end < start_time or seg_start > end_time:
            continue
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        matched.append(
            {
                "start": seg_start,
                "end": seg_end,
                "text": text,
            }
        )
    return matched


def _extract_ocr_frames(ocr_data: Dict[str, Any], segment_index: int) -> List[Dict[str, Any]]:
    folder_key = f"t{segment_index}"
    segment_block = (ocr_data.get("ocr_results") or {}).get(folder_key, {})
    frames = segment_block.get("ocr_frames") or []
    normalized = []
    for frame in frames:
        text = (frame.get("ocr_text") or "").strip()
        normalized.append(
            {
                "time": float(frame.get("time", 0.0) or 0.0),
                "ocr_text": text,
                "file_path": str(frame.get("file_path") or "").strip(),
            }
        )
    return normalized


def _extract_ocr_text(ocr_data: Dict[str, Any], segment_index: int) -> str:
    texts = []
    for frame in _extract_ocr_frames(ocr_data, segment_index):
        text = (frame.get("ocr_text") or "").strip()
        if text:
            texts.append(text)
    return " ".join(texts)


def _collect_relation_text(relation_data: Dict[str, Any], segment_name: str) -> str:
    texts = []
    summary = relation_data.get("summary")
    if isinstance(summary, str):
        texts.append(summary)

    for node in relation_data.get("nodes") or []:
        if str(node.get("video_name") or "").strip() != segment_name:
            continue
        node_text = " ".join(
            [
                str(node.get("title") or "").strip(),
                " ".join(node.get("key_concepts") or []),
                str(node.get("video_id") or "").strip(),
            ]
        ).strip()
        if node_text:
            texts.append(node_text)

    for edge in relation_data.get("all_edges") or []:
        edge_text = " ".join(
            str(edge.get(field, "")).strip()
            for field in ("source", "target", "relation", "relation_type", "reason", "description", "kg_concept_relation")
            if str(edge.get(field, "")).strip()
        )
        if segment_name in edge_text:
            texts.append(edge_text)
    return " ".join(texts)


def _build_segment_records(base_dir: Path) -> List[Dict[str, Any]]:
    split_dir = base_dir / "split_videos"
    splits = _load_knowledge_splits(base_dir)
    summaries = _load_subvideo_summaries(base_dir)
    summary_map = summaries.get("subvideo_summaries") or {}
    relation_data = _load_relation_network(base_dir)
    ocr_data = _load_ocr_summary(base_dir)
    raw_asr = _load_raw_asr(base_dir)

    records = []
    for index, split in enumerate(splits, 1):
        segment_name = _guess_segment_name(index, split, split_dir)
        summary_item = summary_map.get(segment_name, {}) if isinstance(summary_map, dict) else {}
        start_time = float(split.get("start_time", 0.0) or 0.0)
        end_time = float(split.get("end_time", start_time) or start_time)
        segment_path = split_dir / segment_name

        title = (
            split.get("knowledge_point")
            or summary_item.get("title")
            or summary_item.get("subvideo_title")
            or f"知识点 {index}"
        )
        description = split.get("description") or ""
        summary_text = " ".join(
            str(summary_item.get(key, "")).strip()
            for key in ("summary", "logic_flow", "details", "emphasis")
            if str(summary_item.get(key, "")).strip()
        )
        key_concepts = " ".join(summary_item.get("key_concepts") or [])
        ocr_frames = _extract_ocr_frames(ocr_data, index)
        ocr_text = _extract_ocr_text(ocr_data, index)
        asr_segments = _collect_asr_segments_by_time(raw_asr, start_time, end_time)
        asr_text = _slice_asr_by_time(raw_asr, start_time, end_time)
        relation_text = _collect_relation_text(relation_data, segment_name)

        retrieval_text = "\n".join(
            part
            for part in [
                f"标题: {title}",
                f"描述: {description}",
                f"摘要: {summary_text}",
                f"概念: {key_concepts}",
                f"关系: {relation_text}",
                f"OCR: {_truncate_text(ocr_text, 600)}",
                f"ASR: {_truncate_text(asr_text, 600)}",
            ]
            if part and part.split(":", 1)[1].strip()
        )

        records.append(
            {
                "segment_index": index,
                "segment_name": segment_name,
                "segment_path": str(segment_path),
                "title": title,
                "description": description,
                "summary": summary_text,
                "key_concepts": summary_item.get("key_concepts") or [],
                "relation_text": relation_text,
                "ocr_frames": ocr_frames,
                "ocr_text": ocr_text,
                "asr_segments": asr_segments,
                "asr_text": asr_text,
                "start_time": start_time,
                "end_time": end_time,
                "retrieval_text": retrieval_text,
                "summary_item": summary_item,
                "split_item": split,
            }
        )

    return records


def _keyword_overlap_score(question: str, text: str) -> float:
    q_tokens = _tokenize_text(question)
    t_tokens = _tokenize_text(text)
    if not q_tokens or not t_tokens:
        return 0.0

    q_set = set(q_tokens)
    t_set = set(t_tokens)
    overlap = q_set & t_set
    return len(overlap) / max(1, len(q_set))


def _contains_any(text: str, keywords: List[str]) -> bool:
    source = str(text or "").lower()
    return any(keyword.lower() in source for keyword in keywords)


def _classify_question(question: str) -> str:
    normalized = str(question or "").lower()

    visual_pattern = re.compile(
        "(\u753b\u9762\u91cc|\u753b\u9762\u4e2d\u7684|\u677f\u4e66\u4e0a|\u677f\u4e66\u4e0a\u7684|"
        "\u9ed1\u677f\u4e0a|\u8fd9\u4e00\u9875|\u8fd9\u4e00\u5c4f|\u8fd9\u4e2a\u56fe|"
        "\u8fd9\u5f20\u56fe|\u8fd9\u4e2a\u516c\u5f0f|\u8fd9\u4e2a\u5f0f\u5b50|"
        "\u5c4f\u5e55\u4e0a|\u56fe\u91cc|\u9875\u4e0a|ppt)"
    )
    global_pattern = re.compile(
        "(\u8fd9\u4e2a\u89c6\u9891|\u8fd9\u8282\u8bfe|"
        "\u6574\u8282\u8bfe|\u6574\u4e2a\u89c6\u9891|\u6574\u4f53|\u5168\u7247|\u5168\u7a0b|"
        "\u6982\u89c8|\u6846\u67b6|\u603b\u7ed3|\u8109\u7edc|\u7ed3\u6784|\u4e3b\u7ebf|"
        "\u4e3b\u8981\u5185\u5bb9|\u4e3b\u8981\u8bb2|\u54ea\u4e9b\u90e8\u5206|\u77e5\u8bc6\u6846\u67b6)"
    )

    if visual_pattern.search(normalized):
        return QUESTION_TYPE_VISUAL
    if global_pattern.search(normalized):
        return QUESTION_TYPE_GLOBAL
    return QUESTION_TYPE_LOCAL


def _parse_explicit_time_to_seconds(question: str) -> Optional[float]:
    text = str(question or "").strip()
    if not text:
        return None

    mmss_match = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", text)
    if mmss_match:
        minutes = int(mmss_match.group(1))
        seconds = int(mmss_match.group(2))
        return float(minutes * 60 + seconds)

    chinese_match = re.search(r"(?:(\d{1,2})\s*分(?:钟)?)?\s*(\d{1,2})\s*秒", text)
    if chinese_match:
        minutes = int(chinese_match.group(1) or 0)
        seconds = int(chinese_match.group(2) or 0)
        return float(minutes * 60 + seconds)

    plain_seconds_match = re.search(r"(?<![:\d])(\d{1,5}(?:\.\d+)?)\s*秒(?!\d)", text)
    if plain_seconds_match:
        return float(plain_seconds_match.group(1))

    return None


def _needs_current_time_anchor(question: str) -> bool:
    text = str(question or "")
    return bool(
        re.search(
            "(\u73b0\u5728|\u5f53\u524d|\u8fd9\u4e00\u9875|\u8fd9\u4e2a\u753b\u9762|\u73b0\u5728\u7684\u753b\u9762|\u5f53\u524d\u8fd9\u4e2a\u516c\u5f0f)",
            text,
        )
    )


def _normalize_current_time(current_time: Any) -> Optional[float]:
    if current_time is None or current_time == "":
        return None
    if isinstance(current_time, (int, float)):
        return float(current_time)
    text = str(current_time).strip()
    if not text:
        return None
    mmss_match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if mmss_match:
        return float(int(mmss_match.group(1)) * 60 + int(mmss_match.group(2)))
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _find_record_by_time(records: List[Dict[str, Any]], time_seconds: float) -> Optional[Dict[str, Any]]:
    for record in records:
        start_time = float(record.get("start_time", 0.0) or 0.0)
        end_time = float(record.get("end_time", start_time) or start_time)
        if start_time <= time_seconds <= end_time:
            return record
    return None


def _char_ngram_score(question: str, text: str) -> float:
    q_text = _normalize_text(question)
    t_text = _normalize_text(text)
    if not q_text or not t_text:
        return 0.0

    ngrams = {q_text[i : i + 2] for i in range(max(0, len(q_text) - 1))}
    if not ngrams:
        return 0.0
    hits = sum(1 for ng in ngrams if ng and ng in t_text)
    return hits / max(1, len(ngrams))


def _score_candidate(question: str, record: Dict[str, Any], question_type: str = QUESTION_TYPE_LOCAL) -> Dict[str, Any]:
    title_text = f"{record['title']} {record['description']}"
    summary_text = f"{record['summary']} {' '.join(record['key_concepts'])}"
    relation_text = record["relation_text"]
    ocr_asr_text = f"{record['ocr_text']} {record['asr_text']}"

    title_score = _keyword_overlap_score(question, title_text)
    summary_score = _keyword_overlap_score(question, summary_text)
    relation_score = _keyword_overlap_score(question, relation_text)
    evidence_score = _keyword_overlap_score(question, ocr_asr_text)
    fuzzy_score = _char_ngram_score(question, record["retrieval_text"])

    if question_type == QUESTION_TYPE_VISUAL:
        score = (
            title_score * 0.18
            + summary_score * 0.16
            + relation_score * 0.08
            + evidence_score * 0.42
            + fuzzy_score * 0.16
        )
    else:
        score = (
            title_score * 0.32
            + summary_score * 0.24
            + relation_score * 0.14
            + evidence_score * 0.2
            + fuzzy_score * 0.1
        )

    return {
        **record,
        "recall_score": round(score, 4),
        "score_breakdown": {
            "title": round(title_score, 4),
            "summary": round(summary_score, 4),
            "relation": round(relation_score, 4),
            "evidence": round(evidence_score, 4),
            "fuzzy": round(fuzzy_score, 4),
        },
    }


def _recall_candidates(
    question: str,
    records: List[Dict[str, Any]],
    top_k: int = MAX_CANDIDATES,
    question_type: str = QUESTION_TYPE_LOCAL,
) -> List[Dict[str, Any]]:
    scored = [_score_candidate(question, record, question_type=question_type) for record in records]
    scored.sort(key=lambda item: item["recall_score"], reverse=True)
    return scored[:top_k]


def _is_video_related_by_recall(
    question: str,
    candidates: List[Dict[str, Any]],
    question_type: str = QUESTION_TYPE_LOCAL,
) -> bool:
    if not candidates:
        return False
    if question_type == QUESTION_TYPE_VISUAL:
        return True
    top = candidates[0]
    if top["recall_score"] >= 0.12:
        return True

    # allow evidence-heavy match
    breakdown = top.get("score_breakdown") or {}
    return (breakdown.get("title", 0.0) + breakdown.get("evidence", 0.0)) >= 0.18


def _build_candidate_catalog(candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for item in candidates:
        lines.append(
            json.dumps(
                {
                    "segment_name": item["segment_name"],
                    "title": item["title"],
                    "description": item["description"],
                    "summary": _truncate_text(item["summary"], 400),
                    "relation_text": _truncate_text(item["relation_text"], 300),
                    "ocr_hint": _truncate_text(item["ocr_text"], 240),
                    "asr_hint": _truncate_text(item["asr_text"], 240),
                    "recall_score": item.get("recall_score"),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _find_relevant_ocr_frames(question: str, candidate: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    frames = candidate.get("ocr_frames") or []
    scored = []
    for frame in frames:
        text = str(frame.get("ocr_text") or "").strip()
        if not text:
            continue
        score = _keyword_overlap_score(question, text) + _char_ngram_score(question, text)
        scored.append(
            {
                "time": float(frame.get("time", 0.0) or 0.0),
                "ocr_text": text,
                "score": round(score, 4),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return [item for item in scored[:limit] if item["score"] > 0]


def _find_relevant_asr_segments(question: str, candidate: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    segments = candidate.get("asr_segments") or []
    scored = []
    for item in segments:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        score = _keyword_overlap_score(question, text) + _char_ngram_score(question, text)
        scored.append(
            {
                "start": float(item.get("start", 0.0) or 0.0),
                "end": float(item.get("end", 0.0) or 0.0),
                "text": text,
                "score": round(score, 4),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return [item for item in scored[:limit] if item["score"] > 0]


def _build_visual_evidence_hint(question: str, candidate: Dict[str, Any]) -> str:
    lines = [
        f"- 子视频: {candidate['segment_name']}",
        f"- 标题: {candidate['title']}",
        f"- 时间范围: {candidate['start_time']} - {candidate['end_time']}",
        f"- 描述: {candidate['description'] or '无'}",
        f"- 摘要: {_truncate_text(candidate['summary'], 500) or '无'}",
    ]

    ocr_hits = _find_relevant_ocr_frames(question, candidate)
    asr_hits = _find_relevant_asr_segments(question, candidate)

    if ocr_hits:
        lines.append("- OCR 相关帧:")
        lines.extend(f"  - {item['time']}s: {_truncate_text(item['ocr_text'], 180)}" for item in ocr_hits)
    else:
        lines.append("- OCR 相关帧: 无明显高相关命中")

    if asr_hits:
        lines.append("- ASR 相关片段:")
        lines.extend(
            f"  - {item['start']}-{item['end']}s: {_truncate_text(item['text'], 180)}"
            for item in asr_hits
        )
    else:
        lines.append("- ASR 相关片段: 无明显高相关命中")

    return "\n".join(lines)


def _build_global_context(base_dir: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    relation_data = _load_relation_network(base_dir)
    final_llm_input = _load_final_llm_input(base_dir)
    corrected_asr = _load_corrected_asr(base_dir)

    segment_outline = []
    for item in records:
        segment_outline.append(
            {
                "segment_name": item["segment_name"],
                "title": item["title"],
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "description": item["description"],
                "summary": _truncate_text(item["summary"], 320),
                "key_concepts": item["key_concepts"],
            }
        )

    relation_edges = []
    for edge in relation_data.get("all_edges") or []:
        relation_edges.append(
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "relation_type": edge.get("relation_type"),
                "reason": _truncate_text(str(edge.get("reason") or "").strip(), 160),
            }
        )

    return {
        "segment_outline": segment_outline,
        "relation_edges": relation_edges,
        "relation_summary": relation_data.get("summary") or "",
        "final_llm_input": final_llm_input,
        "corrected_asr_hint": _truncate_text(corrected_asr, 2400),
    }


def _rerank_candidates_with_index_model(
    question: str,
    candidates: List[Dict[str, Any]],
    question_type: str = QUESTION_TYPE_LOCAL,
) -> List[Dict[str, Any]]:
    if len(candidates) <= 1:
        return candidates

    type_hint = {
        QUESTION_TYPE_LOCAL: "当前问题是局部知识点问题，请优先选最能直接回答该局部知识点的子视频。",
        QUESTION_TYPE_VISUAL: "当前问题是画面细节问题，请优先选 OCR、ASR、板书/PPT 线索最强的子视频。",
        QUESTION_TYPE_GLOBAL: "当前问题是整体视频问题，这里只做候选排序，最终回答不能只依赖单个子视频。",
    }.get(question_type, "")

    prompt = f"""
你要根据“知识点标题 / description / 子视频摘要 / 关系网 / OCR / ASR 提示词”这些导航索引，判断哪个候选子视频最值得优先进入下一步视频分析。

注意：
1. 这些信息只能用于定位，不是最终作答证据。
2. 请优先选“最可能直接回答用户问题”的子视频，而不是仅仅相关的扩展章节。
3. 只输出 JSON。
4. {type_hint}

用户问题：
{question}

候选列表：
{_build_candidate_catalog(candidates)}

输出：
{{
  "matched_subvideo": "segment_x.mp4",
  "reason": "简短说明"
}}
""".strip()

    response = _call_text_model([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=500)
    payload = _extract_json_block(response) or {}
    matched_name = str(payload.get("matched_subvideo") or "").strip()
    if not matched_name:
        return candidates

    index = {item["segment_name"]: item for item in candidates}
    if matched_name not in index:
        return candidates

    reordered = [index[matched_name]]
    reordered.extend(item for item in candidates if item["segment_name"] != matched_name)
    return reordered


def _build_multimodal_prompt(
    question: str,
    candidates: List[Dict[str, Any]],
    question_type: str = QUESTION_TYPE_LOCAL,
    visual_hint: str = "",
) -> str:
    catalog = _build_candidate_catalog(candidates)
    type_specific = _get_type_specific_rule(question_type)
    return f"""
你将收到一个子视频文件，以及该子视频对应的一些“导航索引”信息。

请严格遵守以下原则：
1. {QUESTION_CLASSIFICATION_RULE}{type_specific}
2. 回答时优先依据子视频画面、讲解、ASR、OCR等直接证据。
3. 若视频直接证据不足，请明确说证据不足，不要编造。
4. 请区分“视频直接说明”和“根据视频内容推断”。
5. {TIME_ANCHOR_RULES}

用户问题：
{question}

候选子视频导航索引：
{catalog}

与当前问题最相关的局部证据提示：
{visual_hint or "无额外局部证据提示"}

当前你正在看的，就是其中一个候选子视频本身。请结合视频内容，输出 JSON：
{_build_answer_json_schema(reason_label="为什么这样判断")}

要求：
1. 只输出 JSON。
2. 若答案主要来自视频直接讲解、画面、字幕或板书，answer_type 用 direct。
3. 若需要结合视频内容做谨慎推断，answer_type 用 infer。
4. 若视频没有足够信息支持回答，answer_type 用 insufficient。
""".strip()


def _build_text_fallback_prompt(
    question: str,
    candidate: Dict[str, Any],
    all_candidates: List[Dict[str, Any]],
    question_type: str = QUESTION_TYPE_LOCAL,
    visual_hint: str = "",
) -> str:
    catalog = _build_candidate_catalog(all_candidates)
    type_specific = _get_type_specific_rule(question_type)
    return f"""
你是一名严谨的视频问答助手。

{QUESTION_CLASSIFICATION_RULE}
{type_specific}
{TIME_ANCHOR_RULES}

当前环境未成功完成视频多模态分析，因此你只能基于候选子视频的 ASR / OCR / 知识点定位线索做保守回答。
如果直接证据不足，请明确说明证据不足，不要编造。
请区分视频直接说明与推断。

用户问题：
{question}

候选导航索引：
{catalog}

当前主候选的直接证据：
- 子视频: {candidate["segment_name"]}
- 标题: {candidate["title"]}
- 描述: {candidate["description"]}
- ASR: {_truncate_text(candidate["asr_text"], 1200)}
- OCR: {_truncate_text(candidate["ocr_text"], 900)}
- 补充证据提示:
{visual_hint or "无"}

请输出 JSON：
{_build_answer_json_schema()}

只输出 JSON。
""".strip()


def _build_unrelated_answer(question: str) -> str:
    prompt = f"""
当前问题与视频内容关联较弱，请你独立思考并直接回答用户问题。
要求：
1. 优先给出正确、简洁、可理解的通用答案。
2. 不要假装答案来自视频。
3. 如果问题本身信息不足，请明确指出信息不足。
4. 最后用一句话提醒：这部分回答不是基于当前视频内容。

用户问题：{question}
"""
    answer = _call_text_model([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=300)
    return answer or "暂时无法可靠回答这个通用问题。这部分回答不是基于当前视频内容。"


def _build_global_answer_prompt(question: str, global_context: Dict[str, Any]) -> str:
    return f"""
你是一名严谨的教学视频问答助手。

{QUESTION_CLASSIFICATION_RULE}
{_get_type_specific_rule(QUESTION_TYPE_GLOBAL, for_global=True)}
摘要、关系网、知识点描述在这里可以用于全局总结，但仍要避免把某个片段机械扩写成整段视频结论。

用户问题：
{question}

整段视频的知识点切分与子视频摘要：
{json.dumps(global_context.get("segment_outline") or [], ensure_ascii=False)}

知识点关系网：
{json.dumps(global_context.get("relation_edges") or [], ensure_ascii=False)}

关系网摘要与全局索引：
{json.dumps({
    "relation_summary": global_context.get("relation_summary") or "",
    "final_llm_input": global_context.get("final_llm_input") or {},
    "corrected_asr_hint": global_context.get("corrected_asr_hint") or "",
}, ensure_ascii=False)}

请输出 JSON：
{_build_answer_json_schema(matched_subvideo="GLOBAL", answer_label="总体回答", evidence_labels=["整体证据1", "整体证据2"])}

只输出 JSON。
""".strip()


def _build_global_fallback_answer(question: str, records: List[Dict[str, Any]]) -> str:
    lines = []
    for item in records:
        lines.append(
            f"{item['segment_index']}. {item['title']}（{item['start_time']}-{item['end_time']}s）："
            f"{_truncate_text(item['description'] or item['summary'], 80)}"
        )
    if not lines:
        return "当前缺少足够的全片结构数据，暂时无法可靠总结整段视频。"
    return "这段视频整体上主要包含以下部分：" + "；".join(lines)


def _normalize_answer_payload(payload: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    evidence = _normalize_evidence_list(payload.get("evidence"))

    answer_type = str(payload.get("answer_type") or ANSWER_TYPE_INSUFFICIENT).strip().lower()
    if answer_type not in VALID_ANSWER_TYPES:
        answer_type = ANSWER_TYPE_INSUFFICIENT

    confidence = _clamp_confidence(payload.get("confidence", 0.0))

    matched_subvideo = str(payload.get("matched_subvideo") or candidate["segment_name"]).strip()
    if not matched_subvideo:
        matched_subvideo = candidate["segment_name"]

    return {
        "is_video_related": bool(payload.get("is_video_related", True)),
        "matched_subvideo": matched_subvideo,
        "answer": str(payload.get("answer") or "").strip(),
        "evidence": evidence,
        "answer_type": answer_type,
        "confidence": confidence,
        "reason": str(payload.get("reason") or "").strip(),
    }


def _answer_global_question(question: str, base_dir: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    global_context = _build_global_context(base_dir, records)
    prompt = _build_global_answer_prompt(question, global_context)
    response = _call_text_model([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=1800)
    payload = _extract_json_block(response) or {}
    normalized = {
        "is_video_related": True,
        "matched_subvideo": str(payload.get("matched_subvideo") or "GLOBAL").strip() or "GLOBAL",
        "answer": str(payload.get("answer") or "").strip(),
        "evidence": _normalize_evidence_list(payload.get("evidence")),
        "answer_type": str(payload.get("answer_type") or ANSWER_TYPE_INFER).strip().lower(),
        "confidence": _clamp_confidence(payload.get("confidence", 0.0)),
        "reason": str(payload.get("reason") or "").strip(),
    }

    if not normalized["answer"]:
        normalized["answer"] = _build_global_fallback_answer(question, records)
        normalized["answer_type"] = ANSWER_TYPE_INFER
        normalized["confidence"] = min(normalized["confidence"] or 0.55, 0.65)
        normalized["reason"] = normalized["reason"] or "模型未返回稳定 JSON，降级为基于全片结构信息的总结"
    if not normalized["evidence"]:
        normalized["evidence"] = [
            f"{item['segment_index']}. {item['title']}" for item in records[: min(4, len(records))]
        ]
    if normalized["answer_type"] not in VALID_ANSWER_TYPES:
        normalized["answer_type"] = ANSWER_TYPE_INFER
    normalized["confidence"] = _clamp_confidence(normalized["confidence"])
    return normalized


def _analyze_candidate_with_multimodal(
    question: str,
    candidate: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    question_type: str = QUESTION_TYPE_LOCAL,
) -> Dict[str, Any]:
    visual_hint = _build_visual_evidence_hint(question, candidate) if question_type == QUESTION_TYPE_VISUAL else ""
    prompt = _build_multimodal_prompt(question, candidates, question_type=question_type, visual_hint=visual_hint)
    mm_response = _call_multimodal_model(Path(candidate["segment_path"]), prompt)
    payload = _extract_json_block(mm_response)
    if payload:
        return _normalize_answer_payload(payload, candidate)

    logger.warning("multimodal response not parseable, fallback to text evidence synthesis")
    fallback_prompt = _build_text_fallback_prompt(
        question,
        candidate,
        candidates,
        question_type=question_type,
        visual_hint=visual_hint,
    )
    fallback_response = _call_text_model([{"role": "user", "content": fallback_prompt}], temperature=0.1, max_tokens=1200)
    payload = _extract_json_block(fallback_response) or {}
    normalized = _normalize_answer_payload(payload, candidate)
    return _apply_insufficient_defaults(
        normalized,
        answer_text=INSUFFICIENT_ANSWER_MM,
        confidence_cap=0.35,
        evidence=INSUFFICIENT_EVIDENCE,
        reason=INSUFFICIENT_REASON_MM,
    )


def run_answer_question(video_id, question, current_time=None):
    result = {
        "answer": "暂无有效回答",
        "is_video_related": False,
        "matched_subvideo": "",
        "evidence": [],
        "answer_type": "insufficient",
        "confidence": 0.0,
        "error": "",
        "code": 200,
        "candidates": [],
        "question_type": QUESTION_TYPE_LOCAL,
        "time_anchor": None,
    }

    question = (question or "").strip()
    if not question:
        result["answer"] = "请输入具体问题后再试。"
        result["error"] = "empty_question"
        result["code"] = 400
        return result

    try:
        logger.info("处理问答 | video_id=%s | question=%s", video_id, question)
        base_dir = Path(VIDEO_DATA_DIR) / str(video_id)
        if not base_dir.exists():
            result["answer"] = "未找到该视频的处理结果目录。"
            result["error"] = "video_data_not_found"
            result["code"] = 404
            return result

        records = _build_segment_records(base_dir)
        if not records:
            result["answer"] = "未找到可用于问答的知识点切分或子视频数据。"
            result["error"] = "segment_records_not_found"
            result["code"] = 404
            return result

        explicit_time = _parse_explicit_time_to_seconds(question)
        normalized_current_time = _normalize_current_time(current_time)
        if explicit_time is not None:
            result["time_anchor"] = explicit_time
            anchored_candidate = _find_record_by_time(records, explicit_time)
            if not anchored_candidate:
                result["answer"] = "该时间点不在任何已知片段范围内。"
                result["error"] = "time_not_in_known_segments"
                result["code"] = 404
                return result

            question_type = _classify_question(question)
            if question_type == QUESTION_TYPE_GLOBAL:
                question_type = QUESTION_TYPE_LOCAL
            return _answer_with_anchored_candidate(
                result,
                question,
                anchored_candidate,
                question_type,
                INSUFFICIENT_ANSWER_BY_TIME,
            )

        if _needs_current_time_anchor(question):
            if normalized_current_time is None:
                result["answer"] = "无法确定“现在/这一页/当前画面”对应的视频时间点，请提供当前播放时间。"
                result["error"] = "current_time_required"
                result["code"] = 400
                result["question_type"] = QUESTION_TYPE_VISUAL
                return result

            result["time_anchor"] = normalized_current_time
            anchored_candidate = _find_record_by_time(records, normalized_current_time)
            if not anchored_candidate:
                result["answer"] = "当前播放时间不在任何已知片段范围内。"
                result["error"] = "current_time_not_in_known_segments"
                result["code"] = 404
                result["question_type"] = QUESTION_TYPE_VISUAL
                return result

            return _answer_with_anchored_candidate(
                result,
                question,
                anchored_candidate,
                QUESTION_TYPE_VISUAL,
                INSUFFICIENT_ANSWER_BY_CURRENT,
            )

        question_type = _classify_question(question)
        result["question_type"] = question_type

        if question_type == QUESTION_TYPE_GLOBAL:
            answer_payload = _answer_global_question(question, base_dir, records)
            result.update(answer_payload)
            result["is_video_related"] = True
            result["candidates"] = [
                {
                    "segment_name": item["segment_name"],
                    "title": item["title"],
                    "recall_score": None,
                    "score_breakdown": {},
                }
                for item in records[: min(len(records), MAX_CANDIDATES)]
            ]
            return result

        candidates = _recall_candidates(question, records, top_k=MAX_CANDIDATES, question_type=question_type)
        candidates = _rerank_candidates_with_index_model(question, candidates, question_type=question_type)
        result["candidates"] = [
            {
                "segment_name": item["segment_name"],
                "title": item["title"],
                "recall_score": item["recall_score"],
                "score_breakdown": item["score_breakdown"],
            }
            for item in candidates
        ]

        logger.info("候选召回结果: %s", json.dumps(result["candidates"], ensure_ascii=False))

        if not _is_video_related_by_recall(question, candidates, question_type=question_type):
            result["answer"] = _build_unrelated_answer(question)
            result["is_video_related"] = False
            result["answer_type"] = "insufficient"
            result["confidence"] = 0.2
            return result

        primary_candidate = candidates[0]
        answer_payload = _analyze_candidate_with_multimodal(
            question,
            primary_candidate,
            candidates,
            question_type=question_type,
        )

        if not answer_payload.get("is_video_related", True):
            result["answer"] = _build_unrelated_answer(question)
            result["is_video_related"] = False
            result["matched_subvideo"] = ""
            result["evidence"] = []
            result["answer_type"] = "infer"
            result["confidence"] = max(0.3, min(float(answer_payload.get("confidence", 0.0) or 0.0), 0.8))
            return result

        if not answer_payload.get("answer"):
            answer_payload = _apply_insufficient_defaults(
                answer_payload,
                answer_text=INSUFFICIENT_ANSWER_BY_CANDIDATE,
                confidence_cap=0.3,
            )

        result.update(answer_payload)
        result["is_video_related"] = bool(answer_payload.get("is_video_related", True))
        return result

    except Exception as exc:
        logger.error("问答异常: %s", str(exc), exc_info=True)
        result["answer"] = f"回答生成失败：{str(exc)[:120]}"
        result["error"] = str(exc)[:240]
        result["code"] = 500
        return result


if __name__ == "__main__":
    test_result = run_answer_question(66, "这个视频里对 FTP 数据端口是怎么讲的？")
    print(json.dumps(test_result, ensure_ascii=False, indent=2))
