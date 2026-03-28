import os
import json
import logging
import sqlite3
from pathlib import Path
import dashscope
from dashscope import Generation

try:
    from .subject_utils import build_subject_constraint_text, infer_subject_hint
except ImportError:  # pragma: no cover
    from subject_utils import build_subject_constraint_text, infer_subject_hint

# ====================== 日志配置（增强报错） ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ====================== 全局配置（改为绝对路径，适配跨目录运行） ======================
DASHSCOPE_API_KEY = "sk-24feda02d5524ed89a3ff3f5e0cee735"
# 基于当前文件目录生成绝对路径，避免工作目录问题
BASE_DIR = Path(__file__).resolve().parent.parent  # 对应项目根目录
VIDEO_DATA_DIR = str(BASE_DIR / "video_data")  # 绝对路径
UPLOAD_FOLDER = str(BASE_DIR / "uploads")      # 绝对路径
DATABASE_PATH = str(BASE_DIR / "videos.db")    # 数据库绝对路径

# ====================== 工具函数：安全路径处理（新增，修复文件名过长） ======================
def sanitize_path(path):
    """清理路径非法字符+限制长度，避免Errno 36"""
    # 移除路径中的特殊字符（中文标点/非法符号）
    illegal_chars = r'\/:*?"<>|()（）【】、，。！？；：￥%&*'
    for char in illegal_chars:
        path = path.replace(char, "_")
    # 限制路径长度（Linux最大255字符，这里留冗余）
    if len(path) > 200:
        path = path[:200]
    return path

# ====================== 通义千问调用（优化导入+精准报错） ======================
def call_qianwen(prompt, model="qwen-plus"):
    """完全复用你原代码的call_qianwen函数，变量名1:1对齐"""
    try:
        dashscope.api_key = DASHSCOPE_API_KEY
        
        response = Generation.call(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            result_format="text",
            temperature=0.7,
            top_p=0.8
        )
        if response.status_code == 200:
            return response.output.text.strip(), True
        else:
            err_msg = f"调用失败：状态码{response.status_code}，信息{response.message}"
            logger.error(err_msg)
            return err_msg, False
    except ImportError as e:
        err_msg = f"模块导入失败：请安装dashscope（pip install dashscope），错误：{e}"
        logger.error(err_msg)
        return err_msg, False
    except Exception as e:
        err_msg = f"调用异常：{str(e)[:100]}"
        logger.error(err_msg)
        return err_msg, False

# ====================== 核心函数（删除OCR+优化Prompt） ======================
def generate_video_title_summary(corrected_asr, subject_hint=None):
    """
    生成网站展示用标题+摘要（彻底删除OCR相关内容）
    :param corrected_asr: 矫正后ASR文本（字符串）
    :return: (title, summary) 与原返回值完全一致
    """
    # 1. 边界值兜底：确保corrected_asr是字符串
    corrected_asr = corrected_asr.strip() if isinstance(corrected_asr, str) else ""
    if not corrected_asr:
        logger.warning("⚠️ corrected_asr为空，使用兜底Prompt")
        return "未命名视频", "暂无摘要"

    detected_subject = subject_hint or infer_subject_hint(corrected_asr[:4000])
    subject_rule = build_subject_constraint_text(detected_subject)

    # 2. 优化Prompt：删除OCR相关内容，指令更清晰
    prompt = f"""
基于以下网课视频的音频文本，严格按格式生成：
1. 标题：简洁概括视频核心内容（≤20字，仅返回标题文本）
2. 摘要：简要说明视频的知识点和讲解逻辑（≤150字，仅返回摘要文本）

学科限制：
{subject_rule}

额外要求：
1. 标题和摘要都要体现课程的主学科，不要写成跨学科泛化描述。
2. 只基于材料中明确出现的内容总结，不要补充课外知识。
3. 摘要优先写“这节课讲了什么 + 讲解顺序/重点”，避免空泛表述。

矫正后音频文本：{corrected_asr[:800]}  # 限制长度避免Prompt过长

输出格式要求：
标题：[你的标题]
摘要：[你的摘要]
    """.strip()

    # 3. 调用通义千问+鲁棒解析（兼容多种格式）
    result, success = call_qianwen(prompt)
    if success:
        # 解析逻辑：兼容"标题：""摘要：" / "1.标题""2.摘要" / "1、标题"等格式
        title = ""
        summary = ""
        # 优先按"标题：""摘要："解析（最稳定）
        if "标题：" in result and "摘要：" in result:
            title = result.split("摘要：")[0].split("标题：")[-1].strip()[:30]
            summary = result.split("摘要：")[-1].strip()[:200]
        # 兼容"1.""2."格式
        elif "1." in result and "2." in result:
            title = result.split("2.")[0].replace("1.", "").strip()[:30]
            summary = result.split("2.")[1].strip()[:200]
        # 兜底：取前20字当标题，后面当摘要
        else:
            title = result.strip()[:20]
            summary = result.strip()[20:170] if len(result) > 20 else "暂无详细摘要"
        
        if title and summary:
            logger.info(f"✅ 标题+摘要生成成功：{title}")
            return title, summary

    # 兜底逻辑
    logger.warning("⚠️ 标题+摘要生成失败，使用兜底值")
    return "未命名视频", "暂无摘要"

# ====================== 辅助函数：按video_id加载数据（删除OCR+修复路径） ======================
def load_video_data_for_title_summary(video_id):
    """
    按video_id加载corrected_asr（彻底删除global_ocr加载逻辑）
    :param video_id: 视频ID（int）
    :return: corrected_asr 仅返回ASR文本，删除OCR相关返回
    """
    try:
        # 1. 安全构建路径：仅用视频ID，清理特殊字符+限制长度
        video_dir = os.path.join(VIDEO_DATA_DIR, sanitize_path(str(video_id)))
        corrected_asr_path = os.path.join(video_dir, "corrected_asr.txt")
        
        # 路径合法性校验（新增，避免文件名过长）
        if len(corrected_asr_path) > 200:
            raise Exception(f"路径过长：{corrected_asr_path}")
        
        if not Path(corrected_asr_path).exists():
            raise FileNotFoundError(f"corrected_asr.txt不存在：{corrected_asr_path}")
        
        with open(corrected_asr_path, "r", encoding="utf-8") as f:
            corrected_asr = f.read().strip()

        logger.info(f"✅ 加载视频{video_id}的corrected_asr成功")
        return corrected_asr

    except Exception as e:
        err_msg = f"加载数据失败：{str(e)[:100]}"
        logger.error(f"❌ {err_msg}")
        return ""

# ====================== 对外统一调用入口（删除OCR+保持接口兼容） ======================
def run_generate_web_title_summary(video_id=None, corrected_asr=None, subject_hint=None):
    """
    对外调用入口：支持两种方式（和原代码兼容，删除OCR参数）
    方式1：传video_id → 自动加载corrected_asr
    方式2：直接传corrected_asr → 直接生成
    :return: (title, summary) 与原返回值完全一致
    """
    try:
        # 方式1：按video_id加载数据
        if video_id and not corrected_asr:
            corrected_asr = load_video_data_for_title_summary(video_id)
            if not corrected_asr:
                return "未命名视频", "暂无摘要"
        
        # 方式2：直接使用传入的corrected_asr
        if not isinstance(corrected_asr, str) or not corrected_asr.strip():
            raise ValueError("corrected_asr不能为空且必须是字符串")
        
        # 调用核心函数（删除OCR参数）
        title, summary = generate_video_title_summary(corrected_asr, subject_hint=subject_hint)
        return title, summary

    except Exception as e:
        err_msg = f"生成标题+摘要异常：{str(e)[:100]}"
        logger.error(f"❌ {err_msg}")
        return "未命名视频", "暂无摘要"
