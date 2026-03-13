import os
import json
import time
import logging
import dashscope
from dashscope import Generation
from pathlib import Path

# ====================== 日志配置 ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ====================== 环境配置 ======================
dashscope.api_key = "sk-24feda02d5524ed89a3ff3f5e0cee735"
PROJECT_ROOT = "/root/video_platform"
VIDEO_DATA_DIR = Path(PROJECT_ROOT) / "video_data"
PRESET_KG_DIR = Path(PROJECT_ROOT) / "video_kg"
RETRY_TIMES = 3
RETRY_DELAY = 5
API_TIMEOUT = 600

# ====================== 路径函数（确认路径正确） ======================
def get_kg_file_paths(video_id):
    video_data_path = VIDEO_DATA_DIR / str(video_id)
    return {
        "SUMMARY_FILE": video_data_path / "subvideo_summaries_all.json",  # 路径正确
        "PRESET_KG_FILE": PRESET_KG_DIR / "preset_kg.json",
        "CUSTOM_KG_OUTPUT": video_data_path / f"custom_kg_{video_id}.json"  # 路径正确
    }

# ====================== 工具函数（无修改） ======================
def load_json_file(file_path, expected_type="dict"):
    file_path = Path(file_path)
    try:
        if not file_path.exists():
            logger.warning(f"⚠️ 文件不存在：{file_path}，返回{expected_type}类型兜底数据")
            return {} if expected_type == "dict" else []
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if expected_type == "dict":
            if isinstance(data, list):
                logger.warning(f"⚠️ 文件{file_path}加载出列表，强制转为字典（key=list_data）")
                return {"list_data": data}
            elif not isinstance(data, dict):
                logger.warning(f"⚠️ 文件{file_path}加载出{type(data)}，强制转为空字典")
                return {}
            return data
        
        elif expected_type == "list":
            if isinstance(data, dict):
                logger.warning(f"⚠️ 文件{file_path}加载出字典，强制转为列表（仅保留values）")
                return list(data.values())
            elif not isinstance(data, list):
                logger.warning(f"⚠️ 文件{file_path}加载出{type(data)}，强制转为空列表")
                return []
            return data
        
        else:
            logger.warning(f"⚠️ 无效的预期类型{expected_type}，返回原始数据")
            return data
            
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON解析失败 {file_path}：{str(e)}，返回{expected_type}类型兜底数据")
        return {} if expected_type == "dict" else []
    except Exception as e:
        logger.error(f"❌ 加载文件失败 {file_path}：{str(e)}，返回{expected_type}类型兜底数据")
        return {} if expected_type == "dict" else []

def save_json_file(data, file_path):
    file_path = Path(file_path)
    try:
        file_path.parent.mkdir(exist_ok=True, parents=True)
        
        def json_default(obj):
            if isinstance(obj, (int, float, bool, str)):
                return obj
            elif isinstance(obj, (list, dict)):
                return obj
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return str(obj)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                data, 
                f, 
                ensure_ascii=False, 
                indent=2, 
                default=json_default,
                separators=(',', ': ')
            )
        logger.info(f"✅ 保存JSON文件成功：{file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ 保存文件失败 {file_path}：{str(e)}")
        return False

# ====================== 核心修复：API调用函数（适配qwen-plus + dashscope 1.25.13） ======================
def call_dashscope_with_retry(prompt):
    # ========== 修改1：适配qwen-plus的日志提示 ==========
    logger.info("🔍 检测到使用qwen-plus模型（dashscope 1.25.13兼容，32k纯文本大窗口）")
    
    # ========== 保留32k窗口参数（适配qwen-plus的32k Token窗口） ==========
    MAX_PROMPT_LENGTH = 32000
    if len(prompt) > MAX_PROMPT_LENGTH:
        logger.warning(f"⚠️ Prompt过长，截断至{MAX_PROMPT_LENGTH}字符（qwen-plus最大支持32k Token）")
        prompt = prompt[:MAX_PROMPT_LENGTH] + "\n### 内容已截断 ###"
    
    for retry in range(RETRY_TIMES):
        try:
            logger.info(f"📡 API 第 {retry+1} 次尝试（超时{API_TIMEOUT}s）...")
            # ========== 修改2：核心模型配置（替换为qwen-plus） ==========
            response = Generation.call(
                model="qwen-plus",  # 关键：换回qwen-plus（纯文本模型，无参数错误）
                # 移除endpoint参数（1.25.13手动指定会触发URL错误）
                messages=[{"role": "user", "content": prompt}],
                result_format="text",
                temperature=0.1,
                top_p=0.9,
                max_tokens=16000,  # 适配32k窗口的生成长度
                timeout=API_TIMEOUT,
                incremental_output=False,
                repetition_penalty=1.1
            )

            # ========== 修改3：适配qwen-plus的报错处理 ==========
            if response.status_code == 401:
                logger.error("❌ [Qwen专属错误] API Key无效/过期！解决建议：1. 检查API Key是否正确；2. 登录阿里云控制台重新生成Key；3. 确认Key有qwen-plus调用权限")
                return None
            elif response.status_code == 403:
                logger.error("❌ [Qwen专属错误] 免费额度耗尽/权限不足！解决建议：1. 登录https://dashscope.console.aliyun.com/查看qwen-plus额度；2. 关闭「仅使用免费额度」开关（无需付费）；3. 切换到qwen-turbo模型")
                return None
            elif response.status_code == 400:
                logger.error("❌ [Qwen专属错误] 参数错误！解决建议：1. 确认模型名是qwen-plus（无后缀）；2. 确保移除了endpoint参数")
                return None
            elif response.status_code == 429:
                logger.error("❌ [Qwen专属错误] 调用频率超限！解决建议：1. 增加重试延迟；2. 降低调用频率")
                time.sleep(RETRY_DELAY * (2 ** retry))
                continue
            elif response.status_code != 200:
                raise ValueError(f"[Qwen通用错误] API错误码：{response.status_code} - {response.message}，参考文档：https://help.aliyun.com/zh/model-studio/error-code")
            
            # 输出内容校验
            if not hasattr(response, "output") or not response.output:
                raise ValueError("[Qwen输出错误] API返回空输出！可能是Prompt违规")
            if not isinstance(response.output.text, str):
                raise ValueError(f"[Qwen输出错误] API返回非字符串类型：{type(response.output.text)}，预期为文本")
            
            return response
        
        # ========== 修改4：qwen-plus专属异常提示 ==========
        except Exception as e:
            err_detail = str(e)[:300]  # 保留更长的错误信息
            logger.warning(f"⚠️ API调用异常（第{retry+1}次）：{err_detail}")
            
            # 针对性提示qwen-plus常见错误
            if "url error" in err_detail:
                logger.warning("💡 解决建议：1. 确保移除了endpoint参数；2. 模型名仅为qwen-plus（无v1/3.5后缀）")
            elif "API Key" in err_detail:
                logger.warning("💡 解决建议：检查API Key是否正确，是否有调用qwen-plus的权限")
            elif "free tier" in err_detail:
                logger.warning("💡 解决建议：切换到qwen-turbo模型，或关闭「仅使用免费额度」开关")
            
            if retry < RETRY_TIMES - 1:
                delay = RETRY_DELAY * (2 ** retry)
                logger.info(f"⏳ {delay}s 后重试...")
                time.sleep(delay)
            else:
                logger.error(f"❌ API重试{RETRY_TIMES}次失败，终止调用！最后错误：{err_detail}")
                return None

def safe_parse_json(content, expected_type="dict"):
    if not content or not isinstance(content, str):
        logger.warning(f"⚠️ 无有效解析内容，返回{expected_type}类型兜底数据")
        return {} if expected_type == "dict" else []
    
    content = content.strip()
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
    
    content = content.replace("\n", "").replace("\r", "").replace("\t", "")
    content = content.replace("\\'", "'").replace('\\"', '"')
    
    start_idx = content.find('{') if expected_type == "dict" else content.find('[')
    end_idx = content.rfind('}') + 1 if expected_type == "dict" else content.rfind(']') + 1
    
    if start_idx == -1 or end_idx == 0:
        logger.warning(f"⚠️ 未找到有效{expected_type}结构：{content[:100]}")
        return {} if expected_type == "dict" else []
    
    try:
        parsed = json.loads(content[start_idx:end_idx])
        if expected_type == "dict" and not isinstance(parsed, dict):
            logger.warning(f"⚠️ 解析结果为{type(parsed)}，强制转为字典")
            return {"parsed_data": parsed} if parsed else {}
        elif expected_type == "list" and not isinstance(parsed, list):
            logger.warning(f"⚠️ 解析结果为{type(parsed)}，强制转为列表")
            return [parsed] if parsed else []
        return parsed
    except json.JSONDecodeError as e:
        try:
            fixed_content = content[start_idx:end_idx].replace("}{", "},{")
            parsed = json.loads(fixed_content)
            if expected_type == "dict" and not isinstance(parsed, dict):
                return {"parsed_data": parsed} if parsed else {}
            elif expected_type == "list" and not isinstance(parsed, list):
                return [parsed] if parsed else []
            return parsed
        except:
            logger.error(f"❌ JSON解析失败：{content[:100]}，错误：{e}")
            return {} if expected_type == "dict" else []

# ====================== 核心函数（无修改） ======================
def generate_video_kg(video_id, **kwargs):
    try:
        custom_kg_path = kwargs.get(
            "custom_kg_path", 
            get_kg_file_paths(video_id)["CUSTOM_KG_OUTPUT"]
        )
        custom_kg_path = Path(custom_kg_path)
        
        custom_kg = generate_custom_knowledge_graph_full(video_id)
        if not isinstance(custom_kg, dict):
            logger.warning(f"⚠️ 知识图谱非字典类型，自动转换")
            custom_kg = {"raw_data": custom_kg} if custom_kg is not None else {}
        
        save_json_file(custom_kg, custom_kg_path)
        logger.info(f"✅ 视频{video_id}知识图谱生成完成：{custom_kg_path}")
        return custom_kg
    except Exception as e:
        error_msg = f"❌ 视频{video_id}生成失败：{str(e)[:200]}"
        logger.error(error_msg)
        fallback_kg = {
            "concept_hierarchy": [
                {
                    "parent": "数学核心概念",
                    "children": [],
                    "description": "兜底概念（生成失败）",
                    "related_video_names": [str(video_id)],
                    "related_video_ids": [video_id]
                }
            ],
            "preset_relations_adjusted": [],
            "video_specific_relations": [
                {
                    "video_name": str(video_id),
                    "relation": "兜底关系",
                    "description": f"生成失败：{error_msg[:50]}"
                }
            ],
            "explanation": f"视频{video_id}生成失败，使用兜底数据",
            "error_info": error_msg
        }
        save_json_file(fallback_kg, get_kg_file_paths(video_id)["CUSTOM_KG_OUTPUT"])
        return fallback_kg

def generate_custom_knowledge_graph_full(video_id):
    file_paths = get_kg_file_paths(video_id)
    SUMMARY_FILE = file_paths["SUMMARY_FILE"]
    PRESET_KG_FILE = file_paths["PRESET_KG_FILE"]
    CUSTOM_KG_OUTPUT = file_paths["CUSTOM_KG_OUTPUT"]

    logger.info(f"🔧 加载视频{video_id}全量摘要和预设知识图谱...")
    summary_data = load_json_file(SUMMARY_FILE, expected_type="dict")
    preset_kg = load_json_file(PRESET_KG_FILE, expected_type="dict")

    summary_dict = {}
    if "subvideo_summaries" in summary_data and isinstance(summary_data["subvideo_summaries"], list):
        for i, seg in enumerate(summary_data["subvideo_summaries"]):
            seg = seg if isinstance(seg, dict) else {"title": str(seg)}
            kg_name = seg.get("knowledge_point", seg.get("title", f"知识点{i+1}")).replace(" ", "_")[:20]
            summary_dict[f"{i+1}_{kg_name}"] = seg
    elif "list_data" in summary_data and isinstance(summary_data["list_data"], list):
        for i, seg in enumerate(summary_data["list_data"]):
            seg = seg if isinstance(seg, dict) else {"title": str(seg)}
            kg_name = seg.get("knowledge_point", seg.get("title", f"知识点{i+1}")).replace(" ", "_")[:20]
            summary_dict[f"{i+1}_{kg_name}"] = seg
    elif summary_data:
        summary_dict = summary_data
    else:
        logger.warning(f"⚠️ 视频{video_id}摘要为空，使用兜底数据")
        summary_dict = {
            "1_默认知识点": {
                "knowledge_point": "默认知识点", 
                "key_concepts": ["数学"], 
                "logic_flow": "兜底逻辑"
            }
        }

    video_core_info = []
    def sort_key(x):
        parts = x.split("_", 1)
        return int(parts[0]) if parts[0].isdigit() else 999
    sorted_video_names = sorted(summary_dict.keys(), key=sort_key)
    
    for video_name in sorted_video_names:
        summary = summary_dict.get(video_name, {})
        summary = summary if isinstance(summary, dict) else {"title": str(summary)}
        
        key_concepts = summary.get("key_concepts", summary.get("core_concepts", ["默认概念"]))
        key_concepts = key_concepts if isinstance(key_concepts, list) else [str(key_concepts)]
        
        logic_flow = summary.get("logic_flow", summary.get("content", "兜底逻辑描述"))
        logic_flow = str(logic_flow) if logic_flow else "兜底逻辑描述"
        
        video_core_info.append({
            "video_name": str(video_name),
            "video_id": int(video_id) if str(video_id).isdigit() else video_id,
            "title": str(summary.get("title", summary.get("knowledge_point", f"知识点{video_name.split('_')[0]}"))),
            "key_concepts": key_concepts,
            "logic_flow": logic_flow
        })
    logger.info(f"📌 视频{video_id}共加载 {len(video_core_info)} 个子视频")

    preset_kg_str = json.dumps(preset_kg, ensure_ascii=False, indent=1)[:8000]
    video_info_str = json.dumps(video_core_info, ensure_ascii=False, indent=1)[:8000]
    
    prompt = f"""
你是专业的数学知识图谱专家，需完成以下任务：
1. 基于「预设通用知识图谱」和「所有子视频的完整摘要」，生成贴合视频实际内容的定制化知识图谱；
2. 必须覆盖所有 {len(video_core_info)} 个子视频，每个视频的核心概念都要体现在图谱中；
3. 定制化图谱要求：
   - 裁剪：仅移除预设图谱中所有视频都未涉及的概念；
   - 补充：为每个视频补充至少1个特有概念/关系；
   - 调整：基于视频的讲解顺序调整概念层级；
   - 关联：每个概念必须关联对应的视频ID/视频名。

### 输入信息
1. 预设通用知识图谱（已截断）：
{preset_kg_str}

2. 所有子视频摘要（按编号顺序，已截断）：
{video_info_str}

### 输出要求
严格输出JSON格式，仅输出JSON，包含以下字段（缺一不可）：
1. "concept_hierarchy"：定制化概念层级（列表），每个元素包含：
   - parent：父概念（字符串）
   - children：子概念列表（数组）
   - description：概念说明（字符串）
   - related_video_names：关联视频名列表（数组）
   - related_video_ids：关联视频ID列表（数组）
2. "preset_relations_adjusted"：调整后的预设关系（列表）
3. "video_specific_relations"：每个视频的特有关系（列表，至少{len(video_core_info)}条）
4. "explanation"：详细说明每个视频在图谱中的对应关系（字符串）
"""

    resp = call_dashscope_with_retry(prompt)
    if not resp:
        error_msg = "API调用多次失败"
        logger.error(f"❌ 视频{video_id}：{error_msg}")
        fallback_kg = {
            "concept_hierarchy": [
                {
                    "parent": "数学核心概念",
                    "children": [v["title"] for v in video_core_info],
                    "description": f"API调用失败，兜底概念层级",
                    "related_video_names": [v["video_name"] for v in video_core_info],
                    "related_video_ids": [v["video_id"] for v in video_core_info]
                }
            ],
            "preset_relations_adjusted": preset_kg.get("relations", []),
            "video_specific_relations": [
                {
                    "video_name": v["video_name"],
                    "relation": f"{v['title']}-兜底关系",
                    "description": f"{v['title']}的核心概念：{','.join(v['key_concepts'])}"
                } for v in video_core_info
            ],
            "explanation": f"视频{video_id}生成失败（{error_msg}），使用兜底数据",
            "error_type": "API_FAILED"
        }
        save_json_file(fallback_kg, CUSTOM_KG_OUTPUT)
        return fallback_kg

    content = resp.output.text.strip()
    custom_kg = safe_parse_json(content, expected_type="dict")
    if not custom_kg:
        error_msg = "JSON解析失败/结果为空"
        logger.error(f"❌ 视频{video_id}：{error_msg}")
        fallback_kg = {
            "concept_hierarchy": [
                {
                    "parent": "数学核心概念",
                    "children": [v["title"] for v in video_core_info],
                    "description": "JSON解析失败，兜底概念层级",
                    "related_video_names": [v["video_name"] for v in video_core_info],
                    "related_video_ids": [v["video_id"] for v in video_core_info]
                }
            ],
            "preset_relations_adjusted": preset_kg.get("relations", []),
            "video_specific_relations": [
                {
                    "video_name": v["video_name"],
                    "relation": f"{v['title']}-兜底关系",
                    "description": f"{v['title']}的核心概念：{','.join(v['key_concepts'])}"
                } for v in video_core_info
            ],
            "explanation": f"视频{video_id}生成失败（{error_msg}），使用兜底数据",
            "error_type": "JSON_PARSE_FAILED"
        }
        save_json_file(fallback_kg, CUSTOM_KG_OUTPUT)
        return fallback_kg

    all_video_names = [str(v["video_name"]) for v in video_core_info]
    generated_video_names = []

    concept_hierarchy = custom_kg.get("concept_hierarchy", [])
    if isinstance(concept_hierarchy, list):
        for item in concept_hierarchy:
            if isinstance(item, dict):
                rel_videos = item.get("related_video_names", [])
                rel_videos = rel_videos if isinstance(rel_videos, list) else [str(rel_videos)]
                generated_video_names.extend([str(v).strip() for v in rel_videos if v])
    generated_video_names = list(set(generated_video_names))

    missing_videos = []
    for v in all_video_names:
        is_covered = any(v in gv or gv in v for gv in generated_video_names if gv)
        if not is_covered:
            missing_videos.append(v)
    
    if missing_videos:
        logger.warning(f"⚠️ 视频{video_id}未覆盖：{missing_videos}")
        if not isinstance(custom_kg.get("video_specific_relations"), list):
            custom_kg["video_specific_relations"] = []
        custom_kg["video_specific_relations"].extend([
            {
                "video_name": str(v),
                "relation": "补充缺失关系",
                "description": f"未被初始图谱覆盖，补充兜底关系（视频ID：{video_id}）"
            } for v in missing_videos
        ])
        save_json_file(custom_kg, CUSTOM_KG_OUTPUT)
    else:
        logger.info(f"✅ 视频{video_id}所有子视频都覆盖！")
    
    assert isinstance(custom_kg, dict), "知识图谱必须返回字典"
    return custom_kg

# ====================== 本地测试 ======================
if __name__ == "__main__":
    TEST_VIDEO_ID = "63"
    print(f"\n========== 开始测试视频{TEST_VIDEO_ID}的知识图谱生成 ==========")
    try:
        custom_kg_result = generate_video_kg(TEST_VIDEO_ID)
        if isinstance(custom_kg_result, dict):
            print(f"\n✅ 测试成功！视频{TEST_VIDEO_ID}知识图谱生成完成：")
            print(f"- 概念层级数量：{len(custom_kg_result.get('concept_hierarchy', []))}")
            print(f"- 视频特有关系数量：{len(custom_kg_result.get('video_specific_relations', []))}")
            print(f"- 输出文件路径：{get_kg_file_paths(TEST_VIDEO_ID)['CUSTOM_KG_OUTPUT']}")
            if "explanation" in custom_kg_result:
                print(f"\n📝 生成说明：{custom_kg_result['explanation'][:200]}...")
        else:
            print(f"⚠️ 测试警告：返回结果非字典类型")
    except Exception as e:
        print(f"\n❌ 测试失败：{str(e)[:300]}")
        import traceback
        traceback.print_exc()  # 打印完整错误栈，便于排查qwen-plus相关问题
    print(f"\n========== 视频{TEST_VIDEO_ID}测试结束 ==========")