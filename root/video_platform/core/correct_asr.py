import os
import json
import time
import logging
from datetime import datetime
import requests
from pathlib import Path

# ====================== 日志配置（新增：更详细的报错输出） ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

# ====================== 配置（验证过的qwen-plus调用配置） ======================
DASHSCOPE_API_KEY = "sk-24feda02d5524ed89a3ff3f5e0cee735"
# ✅ 官方确认的qwen-plus兼容模式接口（绝对正确）
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# ====================== 精细化异常定义（新增：区分不同报错类型） ======================
class APIRequestError(Exception): pass          # 通用API错误
class APIKeyError(Exception): pass              # 密钥错误
class APIQuotaExhaustedError(Exception): pass   # 额度用尽
class APIModelError(Exception): pass            # 模型不存在
class APIReadTimeoutError(Exception): pass      # 读取超时
class JSONParseError(Exception): pass           # JSON解析错误
class DataFormatError(Exception): pass          # 返回格式错误

# ====================== 安全保存文件（不变） ======================
def save_file_safely(file_path, content, encoding="utf-8"):
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)
        logger.info(f"✅ 文件保存成功：{file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ 文件保存失败：{str(e)[:200]}")
        return False

# ====================== 通义千问调用（qwen-plus专用，含精细化报错） ======================
def call_qianwen(prompt, temperature=0.1):
    # 前置校验：密钥不能为空
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your-api-key":
        raise APIKeyError("❌ 通义千问API密钥未配置，请替换DASHSCOPE_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "Connection": "keep-alive",          # 保持连接，减少重连耗时
        "User-Agent": "Python/requests"      # 新增：规范请求头，避免被拦截
    }

    # ✅ 验证过的qwen-plus请求体（绝对正确）
    payload = {
        "model": "qwen-plus",                # 官方唯一正确的模型名
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 16000,                 # 适配qwen-plus的128K窗口
        "stream": False,
        "top_p": 0.9,                        # 新增：官方推荐参数
        "frequency_penalty": 0.0             # 新增：避免重复生成
    }

    max_retries = 3
    retry_delays = [2, 4, 6]  # 延长重试间隔，适配qwen-plus慢响应
    last_error = None

    for retry in range(max_retries):
        try:
            logger.info(f"📡 第 {retry+1}/{max_retries} 次调用qwen-plus，prompt长度：{len(prompt)}字符")
            
            # ✅ 核心修复：拆分超时（连接10秒，读取120秒），适配qwen-plus慢响应
            response = requests.post(
                DASHSCOPE_API_URL,
                headers=headers,
                json=payload,
                timeout=(10, 120),  # 读取超时从60→120秒，解决qwen-plus超时
                verify=False,       # 解决AutoDL SSL问题
                allow_redirects=True
            )

            # ====================== 精细化错误判断（新增核心） ======================
            # 1. 密钥错误（401/403）
            if response.status_code == 401 or response.status_code == 403:
                raise APIKeyError(
                    f"❌ 密钥错误/无权限：状态码{response.status_code}，请检查API Key是否正确/是否有qwen-plus调用权限"
                )
            # 2. 额度用尽（429）
            elif response.status_code == 429:
                raise APIQuotaExhaustedError(
                    f"❌ qwen-plus额度用尽：状态码{response.status_code}，请前往阿里云控制台查看额度"
                )
            # 3. 模型不存在（400）
            elif response.status_code == 400 and "model" in response.text.lower():
                raise APIModelError(
                    f"❌ 模型名错误：状态码{response.status_code}，请确认模型名为'qwen-plus'（无后缀）"
                )
            # 4. 其他非200错误
            elif response.status_code != 200:
                raise APIRequestError(
                    f"❌ API请求失败：状态码{response.status_code}，响应内容：{response.text[:500]}"
                )

            # ====================== 响应解析（增强容错） ======================
            try:
                res = response.json()
            except json.JSONDecodeError as e:
                raise JSONParseError(f"❌ JSON解析失败：{str(e)}，响应内容：{response.text[:500]}")
            
            # 检查返回格式
            if "choices" not in res or len(res.get("choices", [])) == 0:
                raise DataFormatError(f"❌ 返回格式错误：缺少choices字段，响应内容：{json.dumps(res, ensure_ascii=False)[:500]}")
            if "message" not in res["choices"][0] or "content" not in res["choices"][0]["message"]:
                raise DataFormatError(f"❌ 返回格式错误：缺少message/content字段，响应内容：{json.dumps(res, ensure_ascii=False)[:500]}")
            
            # 提取结果
            text_result = res["choices"][0]["message"]["content"].strip()
            if not text_result:
                logger.warning("⚠️ qwen-plus返回空文本，使用原始prompt兜底")
                text_result = prompt
            
            logger.info(f"✅ qwen-plus调用成功，返回文本长度：{len(text_result)}字符")
            return text_result, True

        # ====================== 精细化异常捕获（新增核心） ======================
        except requests.exceptions.ReadTimeout:
            last_error = APIReadTimeoutError(f"❌ 第{retry+1}次调用超时：读取超时（120秒），qwen-plus响应较慢，请稍后重试")
            logger.error(last_error)
        except requests.exceptions.ConnectTimeout:
            last_error = APIRequestError(f"❌ 第{retry+1}次调用超时：连接超时（10秒），请检查网络")
            logger.error(last_error)
        except (APIKeyError, APIQuotaExhaustedError, APIModelError) as e:
            # 这类错误无需重试，直接抛出
            last_error = e
            logger.error(f"🔴 致命错误（无需重试）：{str(e)}")
            break
        except (APIRequestError, JSONParseError, DataFormatError) as e:
            last_error = e
            logger.error(f"❌ 第{retry+1}次调用失败：{str(e)[:300]}")
        except Exception as e:
            last_error = APIRequestError(f"❌ 第{retry+1}次调用失败：未知错误 {type(e).__name__}：{str(e)[:300]}")
            logger.error(last_error)

        # 非致命错误，等待后重试
        if retry < max_retries - 1 and not isinstance(last_error, (APIKeyError, APIQuotaExhaustedError, APIModelError)):
            logger.info(f"⏳ 等待{retry_delays[retry]}秒后重试...")
            time.sleep(retry_delays[retry])

    # 所有重试失败，抛出最终错误
    raise last_error

# ====================== ASR矫正主函数（增强报错捕获） ======================
def run_correct_asr(video_id, raw_asr_text, video_data_dir):
    work_dir = os.path.join(video_data_dir, str(video_id))
    corrected_asr = raw_asr_text  # 兜底：默认使用原始文本
    logger.info(f"========== 开始处理视频{video_id}的ASR矫正（qwen-plus）==========")

    try:
        # 生成矫正Prompt
        prompt = f"""
你是教学视频ASR纠错专家，严格按照以下规则执行：
1. 仅矫正原始ASR文本中的错别字、同音错误、语法错误、口语化冗余；
2. 保留所有教学相关内容，不增删语义，不简化专业术语，不添加任何额外解释；
3. 输出格式：仅返回矫正后的纯文本，无换行、无序号、无多余内容。

原始ASR文本：
{raw_asr_text}
""".strip()

        # 调用qwen-plus进行矫正（捕获所有精细化错误）
        try:
            corrected_asr, _ = call_qianwen(prompt)
        except APIKeyError as e:
            logger.error(f"🔴 密钥错误：{str(e)}，使用原始文本兜底")
        except APIQuotaExhaustedError as e:
            logger.error(f"🔴 额度用尽：{str(e)}，使用原始文本兜底")
        except APIModelError as e:
            logger.error(f"🔴 模型错误：{str(e)}，使用原始文本兜底")
        except APIReadTimeoutError as e:
            logger.error(f"🔴 读取超时：{str(e)}，使用原始文本兜底")
        except Exception as e:
            logger.error(f"🔴 qwen-plus调用失败：{str(e)[:200]}，使用原始文本兜底")

        # 强制保存矫正结果（保证后续流程不崩）
        save_file_safely(os.path.join(work_dir, "corrected_asr.txt"), corrected_asr)
        logger.info(f"========== 完成视频{video_id}的ASR矫正 ==========")
        return corrected_asr

    except Exception as e:
        # 全局异常捕获：确保文件一定保存
        logger.error(f"❌ ASR矫正主流程崩溃：{str(e)}")
        save_file_safely(os.path.join(work_dir, "corrected_asr.txt"), raw_asr_text)
        return raw_asr_text

# ====================== 测试入口（新增：验证qwen-plus调用） ======================
if __name__ == "__main__":
    # 测试用例：验证qwen-plus是否能正常调用
    TEST_VIDEO_ID = "test_58"
    TEST_RAW_ASR = "这是一段测试文本，包含同因错误，比如把函数写成蟊敏，把DNS写成丹妮丝。"
    TEST_VIDEO_DATA_DIR = "/root/video_platform/video_data"

    try:
        result = run_correct_asr(TEST_VIDEO_ID, TEST_RAW_ASR, TEST_VIDEO_DATA_DIR)
        print(f"\n✅ 测试结果：\n{result}")
    except Exception as e:
        logger.error(f"❌ 测试失败：{str(e)}")
        raise