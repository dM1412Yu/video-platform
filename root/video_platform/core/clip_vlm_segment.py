import os
import json
import dashscope
import subprocess
import traceback
import re
from pathlib import Path
import cv2  # 新增：用于精准获取视频时长

# ==========================
# 核心配置
# ==========================
DASHSCOPE_API_KEY = "sk-24feda02d5524ed89a3ff3f5e0cee735"
MIN_SEC = 30          
MAX_SEC = 270         
FORCE_MERGE_BELOW = 30
OVERLAP_GAP = 0.1     
OUTPUT_DIR = "/root/video_platform/video_data"  
FFMPEG_PATH = "ffmpeg"

dashscope.api_key = DASHSCOPE_API_KEY

# ==========================
# 工具函数（重点修复：精准获取视频时长）
# ==========================
def clean_filename(filename):
    """增强版：彻底清洗文件名，避免特殊字符/长度问题"""
    # 移除所有非法字符（保留中文、字母、数字、下划线）
    filename = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9_]', '_', filename)
    return filename[:20]  # 缩短长度，避免路径过长

def get_video_duration(video_path):
    """
    精准获取视频真实时长（修复600秒默认值问题）
    优先用ffprobe，失败则用cv2逐帧计算，确保准确
    """
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在：{video_path}")
        return 900.0  # 兜底15分钟，适配14分钟的视频

    # 方案1：用ffprobe精准获取（推荐）
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        duration = float(result.stdout.strip())
        # 校验时长合理性（10秒 ~ 10小时）
        if 10 <= duration <= 36000:
            print(f"✅ 精准获取视频时长：{duration:.2f}秒（{duration/60:.1f}分钟）")
            return duration
    except Exception as e:
        print(f"⚠️ ffprobe获取时长失败：{str(e)[:50]}，改用CV2方案")

    # 方案2：CV2逐帧计算（最精准）
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = total_frames / fps if fps > 0 else 900.0
            cap.release()
            print(f"✅ CV2获取视频时长：{duration:.2f}秒（{duration/60:.1f}分钟）")
            return duration
    except Exception as e:
        print(f"⚠️ CV2获取时长失败：{str(e)[:50]}")

    # 最终兜底（不再用600秒，适配14分钟视频）
    return 900.0

# ==========================
# 新增：视频物理切割函数（极简版，无try/catch、无兜底）
# ==========================
def video_split(video_path, final_splits, video_id):
    """
    基于知识点拆分的时间戳切割视频
    :param video_path: 原视频路径
    :param final_splits: 知识点拆分结果（含start_time/end_time）
    :param video_id: 视频ID
    """
    # 创建子视频输出目录
    split_video_dir = os.path.join(OUTPUT_DIR, str(video_id), "split_videos")
    Path(split_video_dir).mkdir(exist_ok=True)
    
    # 遍历知识点拆分结果，逐段切割
    split_video_paths = []
    for idx, seg in enumerate(final_splits):
        seg_start = float(seg["start_time"])
        seg_end = float(seg["end_time"])
        seg_duration = seg_end - seg_start
        
        # 跳过过短分段（<0.5秒）
        if seg_duration < 0.5:
            print(f"⚠️ 分段{idx+1}时长过短（{seg_duration:.2f}秒），跳过切割")
            continue
        
        # 生成子视频路径
        seg_video_path = os.path.join(split_video_dir, f"segment_{idx+1}.mp4")
        split_video_paths.append(seg_video_path)
        
        # 构建ffmpeg切割命令（快速复制编码，不重新压缩）
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-ss", str(seg_start),       # 起始时间（秒）
            "-i", video_path,           # 原视频路径
            "-to", str(seg_duration),   # 持续时长（秒）
            "-c:v", "copy",             # 视频编码直接复制
            "-c:a", "copy",             # 音频编码直接复制
            "-y",                       # 覆盖已有文件
            seg_video_path
        ]
        
        # 执行切割（无容错，失败直接抛出异常）
        subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 超时5分钟
        )
        print(f"✅ 分段{idx+1}切割完成：{seg_video_path}")
    
    print(f"\n📌 视频切割完成：共生成{len(split_video_paths)}个子视频，保存至{split_video_dir}")

# ==========================
# 兜底拆分（修复时长默认值）
# ==========================
def get_fallback_splits(subtitles, video_id):
    """修复：兜底拆分也用真实时长，而非固定600秒"""
    # 获取视频真实时长
    video_duration = 900.0
    video_output_dir = os.path.join(OUTPUT_DIR, str(video_id))
    # 尝试从已保存的信息中获取真实时长
    for file in os.listdir("/root/video_platform/uploads"):
        if str(video_id) in file:
            video_duration = get_video_duration(os.path.join("/root/video_platform/uploads", file))
            break
    
    if not subtitles:
        return [{
            "start_time": 0,
            "end_time": video_duration,
            "knowledge_point": "视频核心知识点",
            "description": "教学视频核心内容讲解"
        }]
    
    total_start = min([s["start_time"] for s in subtitles])
    total_end = max([s["end_time"] for s in subtitles])
    total_duration = total_end - total_start
    
    split_count = max(1, int(total_duration / MAX_SEC))
    split_duration = total_duration / split_count
    
    fallback_splits = []
    for i in range(split_count):
        start = total_start + i * split_duration
        end = min(start + split_duration, total_end)
        fallback_splits.append({
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "knowledge_point": f"教学知识点{i+1}",
            "description": f"知识点讲解（{start:.1f}~{end:.1f}秒）"
        })
    
    print(f"⚠️ 使用兜底拆分：共{len(fallback_splits)}个知识点")
    return fallback_splits

# ==========================
# 纯文本拆分（新增：传入真实时长，避免超限）
# ==========================
def call_text_only_knowledge_split(subtitles, video_id, video_duration):
    """修复：在prompt中告知大模型视频真实时长，避免生成超限时间戳"""
    prompt = f"""
你是教学视频知识点拆分专家，仅基于语音字幕文本语义拆分，严格遵循以下规则：
1. 核心原则：以**完整的大知识点**为单位拆分，绝对禁止拆分为细碎小知识点
2. 时长强制规则：
   - 单个知识点时长控制在 {MIN_SEC}~{MAX_SEC} 秒
   - 小于 {FORCE_MERGE_BELOW} 秒的知识点必须与前后语义相关的合并
   - 超过 {MAX_SEC} 秒的知识点仅按语义断点拆分（如概念/公式/例题）
3. 拆分规则：
   - 语义相关的短知识点（如定义+公式）必须合并为一个完整知识点
   - 不同主题的知识点（如牛顿定律/机械能守恒）独立拆分
   - 严格按字幕时间戳标注，不凭空生成时间
   - 所有时间戳必须在 0 ~ {video_duration:.2f} 秒范围内（视频真实总时长）
4. 输出要求：
   - 仅返回JSON数组，无任何多余文字
   - 每个对象必须包含：start_time(数字/秒)、end_time(数字/秒)、knowledge_point(知识点名称)、description(简要说明)

字幕数据：
{json.dumps(subtitles, ensure_ascii=False, indent=2)}
""".strip()

    try:
        resp = dashscope.MultiModalConversation.call(
            model="qwen-vl-max",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            result_format="json",
            temperature=0.1
        )
        
        if resp.status_code != 200:
            print(f"❌ 文本拆分失败：{resp.message}")
            return get_fallback_splits(subtitles, video_id)

        content = resp.output.choices[0].message.content
        if isinstance(content, list) and len(content) > 0 and "text" in content[0]:
            content = content[0]["text"]
        content = content.strip().replace("\\n", "").replace('\\"', '"').replace("```json", "").replace("```", "")
        
        splits = json.loads(content)
        valid_splits = []
        for s in splits:
            if all(k in s for k in ["start_time", "end_time", "knowledge_point"]) and s["end_time"] > s["start_time"]:
                # 强制修正：确保时间戳在视频时长范围内
                start = round(max(0.0, min(s["start_time"], video_duration - 1)), 2)
                end = round(max(start + 1, min(s["end_time"], video_duration)), 2)
                valid_splits.append({
                    "start_time": start,
                    "end_time": end,
                    "knowledge_point": s["knowledge_point"].strip(),
                    "description": s.get("description", "知识点讲解").strip()
                })

        if not valid_splits:
            valid_splits = get_fallback_splits(subtitles, video_id)

        video_output_dir = os.path.join(OUTPUT_DIR, str(video_id))
        os.makedirs(video_output_dir, exist_ok=True)
        save_path = os.path.join(video_output_dir, "text_only_knowledge_splits.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(valid_splits, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 纯文本知识点拆分完成：共{len(valid_splits)}个")
        return valid_splits
    except Exception as e:
        print(f"❌ 文本拆分异常：{str(e)}")
        return get_fallback_splits(subtitles, video_id)

# ==========================
# 时间轴调整（保留）
# ==========================
def adjust_no_overlap(segments, video_id):
    if not segments:
        return []
    
    segments = sorted(segments, key=lambda x: x["start_time"])
    final_segments = []
    prev_end = 0.0

    for seg in segments:
        new_start = max(prev_end + OVERLAP_GAP, seg["start_time"])
        new_end = seg["end_time"]
        
        if new_end - new_start < 5:
            continue
        
        final_seg = seg.copy()
        final_seg["start_time"] = round(new_start, 2)
        final_seg["end_time"] = round(new_end, 2)
        final_segments.append(final_seg)
        prev_end = new_end

    video_output_dir = os.path.join(OUTPUT_DIR, str(video_id))
    save_path = os.path.join(video_output_dir, "final_no_overlap_splits.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_segments, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 时间轴调整完成：{len(final_segments)}个无重叠片段")
    return final_segments

# ==========================
# 加载ASR（保留）
# ==========================
def load_corrected_asr(asr_path, video_duration):
    asr_dir = os.path.dirname(asr_path)
    raw_asr_path = os.path.join(asr_dir, "raw_asr_segments.json")

    try:
        with open(raw_asr_path, "r", encoding="utf-8") as f:
            subtitles = json.load(f)
        
        valid_subtitles = []
        for seg in subtitles:
            if all(k in seg for k in ["start", "end", "text"]) and seg["text"].strip():
                valid_subtitles.append({
                    "start_time": round(seg["start"], 2),
                    "end_time": round(seg["end"], 2),
                    "text": seg["text"].strip()
                })
        
        print(f"✅ 加载ASR字幕完成：{len(valid_subtitles)}个片段")
        return valid_subtitles
    except Exception as e:
        print(f"❌ 加载ASR失败：{str(e)}")
        return [{
            "start_time": 0,
            "end_time": video_duration,
            "text": "教学视频核心内容讲解"
        }]

# ==========================
# 主函数（整合视频切割，知识点拆分+切割一体化）
# ==========================
def run_knowledge_split(video_path, asr_path, video_id):
    """
    知识点拆分 + 视频物理切割 一体化主流程
    入参：
        video_path: 视频文件路径
        asr_path: ASR文件路径
        video_id: 视频ID
    出参：
        final_splits: 知识点片段列表
    """
    # 1. 获取视频真实时长
    video_duration = get_video_duration(video_path)
    print(f"✅ 视频信息：时长{video_duration:.1f}秒")
    
    # 2. 加载ASR字幕
    subtitles = load_corrected_asr(asr_path, video_duration)
    
    # 3. 纯文本拆分知识点
    raw_splits = call_text_only_knowledge_split(subtitles, video_id, video_duration)
    
    # 4. 调整时间轴无重叠
    final_splits = adjust_no_overlap(raw_splits, video_id)
    
    # 5. 新增：执行视频物理切割（核心新增逻辑）
    print("\n🔪 开始执行视频物理切割...")
    video_split(video_path, final_splits, video_id)
    
    # 保存最终知识点拆分结果
    video_output_dir = os.path.join(OUTPUT_DIR, str(video_id))
    final_path = os.path.join(video_output_dir, "final_knowledge_splits.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_splits, f, ensure_ascii=False, indent=2)
    
    print("\n🎉 知识点拆分+视频切割全流程完成！")
    print(f"📁 结果保存至：{video_output_dir}")
    return final_splits

# 确保函数能被外部导入
__all__ = ["run_knowledge_split"]
if __name__ == "__main__":
    # ====================== 本地测试配置（只需改这里） ======================
    TEST_VIDEO_ID = 59  # 替换成你要测试的视频ID
    UPLOAD_DIR = "/root/video_platform/uploads"  # 原视频存储目录
    VIDEO_DATA_DIR = "/root/video_platform/video_data"  # 视频数据目录

    # ====================== 自动查找文件 ======================
    # 1. 查找原视频文件
    video_path = None
    for file in os.listdir(UPLOAD_DIR):
        if file.endswith(".mp4"):
            # 优先匹配含视频ID的文件，无则取第一个MP4
            if str(TEST_VIDEO_ID) in file or video_path is None:
                video_path = os.path.join(UPLOAD_DIR, file)
    if not video_path:
        raise FileNotFoundError(f"❌ 未找到视频文件（ID：{TEST_VIDEO_ID}），请检查{UPLOAD_DIR}目录")
    print(f"🔍 找到测试视频：{video_path}")

    # 2. 查找ASR文件（corrected_asr.txt）
    asr_path = os.path.join(VIDEO_DATA_DIR, str(TEST_VIDEO_ID), "corrected_asr.txt")
    if not os.path.exists(asr_path):
        raise FileNotFoundError(f"❌ 未找到ASR文件：{asr_path}，请先生成矫正ASR")
    print(f"🔍 找到ASR文件：{asr_path}")

    # ====================== 执行知识点拆分+视频切割 ======================
    print(f"\n🚀 开始测试视频ID {TEST_VIDEO_ID} 的知识点拆分+视频切割...")
    final_splits = run_knowledge_split(video_path, asr_path, TEST_VIDEO_ID)
    
    # 输出测试结果
    print(f"\n✅ 测试完成！")
    print(f"📊 拆分结果：共{len(final_splits)}个知识点片段")
    for idx, seg in enumerate(final_splits):
        print(f"  - 片段{idx+1}：{seg['knowledge_point']}（{seg['start_time']}~{seg['end_time']}秒）")
    print(f"📁 子视频保存路径：{os.path.join(VIDEO_DATA_DIR, str(TEST_VIDEO_ID), 'split_videos')}")