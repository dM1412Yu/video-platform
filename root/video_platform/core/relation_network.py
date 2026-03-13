import os
import json
import time
import logging
import re
from datetime import datetime

# ====================== 全局配置（适配你的环境） ======================
# 可根据需要调整的参数
BATCH_SIZE = 5  # 每批处理的子视频数量（参考代码的核心参数）
RETRY_TIMES = 3  # API重试次数
RETRY_DELAY = 2  # 重试延迟（秒）
API_TIMEOUT = 120  # API超时时间（秒）
DASHSCOPE_API_KEY = "sk-24feda02d5524ed89a3ff3f5e0cee735"

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ====================== 基础工具函数（适配你的环境） ======================
def safe_load_json(file_path):
    """安全读取JSON文件（你的环境专属）"""
    if not os.path.exists(file_path):
        logger.warning(f"JSON文件不存在：{file_path}")
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败 {file_path}：{str(e)}")
        return {}
    except Exception as e:
        logger.error(f"读取文件 {file_path} 失败：{str(e)}", exc_info=True)
        return {}

def save_json_file(data, file_path):
    """保存JSON文件（你的环境专属）"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 文件已保存：{file_path}")
    except Exception as e:
        logger.error(f"保存文件 {file_path} 失败：{str(e)}", exc_info=True)
        raise

def extract_segment_id_from_filename(filename):
    """从你的子视频文件名提取ID（适配segment_1_xxx.mp4格式）"""
    match = re.match(r'^segment_(\d+)_', filename)
    if match:
        return int(match.group(1))
    match = re.match(r'^(\d+)_', filename)
    return int(match.group(1)) if match else 999

def safe_extract_video_name(node):
    """安全提取节点的video_name（兼容多字段容错）"""
    try:
        return node.get("video_name") or node.get("video_id") or ""
    except:
        return ""

def call_dashscope_with_retry(prompt, max_retries=RETRY_TIMES, timeout=API_TIMEOUT):
    """封装qwen-plus调用（带重试，适配你的环境）"""
    import dashscope
    from dashscope import Generation
    dashscope.api_key = DASHSCOPE_API_KEY
    
    for retry in range(max_retries):
        try:
            response = Generation.call(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                result_format="text",
                temperature=0.1,
                top_p=0.9,
                timeout=timeout
            )
            if response.status_code == 200:
                return response
            else:
                raise Exception(f"API返回非200状态码：{response.status_code}")
        except Exception as e:
            logger.error(f"❌ 第{retry+1}次调用失败：{str(e)[:80]}")
            if retry == max_retries - 1:
                raise
            time.sleep(RETRY_DELAY * (retry + 1))
    raise Exception("API调用多次失败，已达最大重试次数")

# ====================== 核心函数：仿写参考代码的generate_relation_network ======================
def generate_relation_network(video_id, work_dir, custom_kg_path):
    """
    两步法生成关系网（完全仿写参考代码逻辑，适配你的环境）
    :param video_id: 视频ID（如59）
    :param work_dir: 视频根目录（/root/video_platform/video_data/59/）
    :param custom_kg_path: 自定义知识图谱路径
    :return: 最终合并的关系网数据
    """
    # 1. 定义你的环境路径（替换参考代码的硬编码常量）
    SUMMARY_FILE = os.path.join(work_dir, "subvideo_summaries_all.json")  # 你的摘要文件
    RELATION_NETWORK_OUTPUT = os.path.join(work_dir, "relation_network.json")  # 关系网输出
    FINAL_LLM_INPUT = os.path.join(work_dir, "final_llm_input.json")  # 最终LLM输入文件

    # 2. 加载基础数据（仅修改这部分，适配列表/字典格式，其余不变）
    logger.info("🔧 加载全量摘要和定制化图谱...")
    summary_raw = safe_load_json(SUMMARY_FILE)
    # ======== 仅新增/修改这几行，修复list无get方法的问题 ========
    if isinstance(summary_raw, list):
        # 如果是纯列表，转成字典（以subvideo_id为key，兼容后续.keys()逻辑）
        summary_data = {}
        for item in summary_raw:
            sub_id = item.get("subvideo_id") or item.get("video_name") or str(item.get("id", 999))
            if sub_id:
                summary_data[sub_id] = item
    else:
        # 原逻辑不变，兼容字典格式
        summary_data = summary_raw.get("subvideo_summaries", {})
    # ==========================================================
    custom_kg = safe_load_json(custom_kg_path)

    # 3. 整理并排序所有子视频（适配你的文件名格式）
    sorted_video_names = sorted(
        summary_data.keys(),
        key=lambda x: extract_segment_id_from_filename(x)  # 替换参考代码的排序逻辑
    )
    total_videos = len(sorted_video_names)
    logger.info(f"📌 共加载 {total_videos} 个子视频（视频ID：{video_id}）")

    # 4. 构建全量视频节点（用于跨批次分析）
    all_video_nodes = []
    for video_name in sorted_video_names:
        summary = summary_data[video_name]
        all_video_nodes.append({
            "video_name": video_name,
            "video_id": summary.get("video_id", ""),
            "title": summary.get("title", ""),
            "key_concepts": summary.get("key_concepts", []),
            "logic_flow": summary.get("logic_flow", "")
        })

    # 5. 第一步：分批生成批次内关系（完全保留参考代码逻辑）
    logger.info("\n========== 第一步：生成批次内关系 ==========")
    batch_relations = []
    for batch_idx in range(0, total_videos, BATCH_SIZE):
        batch_video_names = sorted_video_names[batch_idx:batch_idx+BATCH_SIZE]
        batch_num = (batch_idx // BATCH_SIZE) + 1
        logger.info(f"\n处理第{batch_num}批（共{len(batch_video_names)}个视频）...")
        
        # 构建批次节点
        batch_nodes = [n for n in all_video_nodes if n["video_name"] in batch_video_names]
        
        # 批次内关系Prompt（完全复用参考代码的Prompt，仅适配变量名）
        prompt = f"""
你是专业的数学教学视频分析专家，请分析当前批次内子视频的相互关系。
核心要求：仅分析本批次内视频的关系，不考虑批次外视频，每个节点必须包含video_name字段。

### 输入信息
1. 定制化知识图谱：
{json.dumps(custom_kg, ensure_ascii=False, indent=2)}

2. 当前批次子视频摘要：
{json.dumps(batch_nodes, ensure_ascii=False, indent=2)}

### 关系类型（严格使用）
prerequisite/progressive/parallel/supplementary/summary

### 输出要求
严格输出JSON格式，字段说明：
- nodes：本批次节点列表，每个节点必须包含video_name/video_id/title/key_concepts；
- edges：本批次内关系边列表，每条边包含source/target/relation_type/reason/kg_concept_relation；
- coverage_check：本批次覆盖的video_name列表。
禁止输出多余文字，仅输出纯JSON内容。
"""
        
        # 调用API（适配你的call_dashscope_with_retry）
        response = call_dashscope_with_retry(prompt)
        content = response.output.text.strip()
        
        # 增强JSON解析容错（完全复用参考代码的容错逻辑）
        if isinstance(content, list) and "text" in content[0]:
            content = content[0]["text"]
        content = content.strip()
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()
        if content and content[0] != '{':
            start_idx = content.find('{')
            if start_idx != -1:
                content = content[start_idx:]
        
        if not content:
            raise ValueError(f"第{batch_num}批返回内容为空，无法解析")
            
        batch_relation = json.loads(content)
        batch_relations.append(batch_relation)
        
        logger.info(f"✅ 第{batch_num}批批次内关系生成完成")

    # 6. 第二步：生成跨批次关系（完全复用参考代码逻辑）
    logger.info("\n========== 第二步：生成跨批次关系 ==========")
    cross_batch_prompt = f"""
你是专业的数学教学视频分析专家，请分析所有子视频中**跨批次的关系**（不同批次视频之间的关联）。
核心要求：
1. 只分析不同批次视频之间的关系，不要重复生成批次内的关系；
2. 必须覆盖所有跨批次的相关视频，禁止遗漏；
3. 所有字段必须关联video_name（而非video_id/title）。

### 输入信息
1. 定制化知识图谱：
{json.dumps(custom_kg, ensure_ascii=False, indent=2)}

2. 所有子视频摘要（按编号顺序）：
{json.dumps(all_video_nodes, ensure_ascii=False, indent=2)}

3. 批次划分信息：
每批{BATCH_SIZE}个视频，共{len(batch_relations)}批，批次范围：
{[f"第{i+1}批：{sorted_video_names[i*BATCH_SIZE:(i+1)*BATCH_SIZE]}" for i in range(len(batch_relations))]}

### 关系类型（严格使用）
prerequisite/progressive/parallel/supplementary/summary

### 输出要求
严格输出JSON格式，仅包含以下字段：
1. "cross_edges"：跨批次关系边列表（每条边必须包含source/target/relation_type/reason/kg_concept_relation，source/target为video_name）；
2. "cross_coverage"：跨批次关系覆盖的video_name列表。
禁止输出多余文字，仅输出纯JSON内容。
"""
    
    # 调用API生成跨批次关系
    response = call_dashscope_with_retry(cross_batch_prompt)
    content = response.output.text.strip()
    
    # 同样增强解析容错（复用参考代码逻辑）
    if isinstance(content, list) and "text" in content[0]:
        content = content[0]["text"]
    content = content.strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    elif content.startswith("```") and content.endswith("```"):
        content = content[3:-3].strip()
    if content and content[0] != '{':
        start_idx = content.find('{')
        if start_idx != -1:
            content = content[start_idx:]
    
    if not content:
        raise ValueError("跨批次关系返回内容为空，无法解析")
        
    cross_batch_relation = json.loads(content)
    logger.info("✅ 跨批次关系生成完成")

    # 7. 第三步：合并所有关系（完全复用参考代码的合并逻辑）
    logger.info("\n========== 第三步：合并所有关系 ==========")
    # 合并批次内节点和边（增加去重+容错）
    merged_nodes = []
    merged_batch_edges = []
    for batch_rel in batch_relations:
        # 节点去重（兼容多字段）
        for node in batch_rel.get("nodes", []):
            node_name = safe_extract_video_name(node)
            if node_name and not any(safe_extract_video_name(exist_node) == node_name for exist_node in merged_nodes):
                merged_nodes.append(node)
        # 合并批次内边（过滤空值）
        batch_edges = batch_rel.get("edges", [])
        merged_batch_edges.extend([edge for edge in batch_edges if edge.get("source") and edge.get("target")])
    
    # 整合最终关系网（完全复用参考代码的结构）
    final_relation = {
        "nodes": merged_nodes,  # 全量节点
        "batch_edges": merged_batch_edges,  # 批次内关系边
        "cross_edges": cross_batch_relation.get("cross_edges", []),  # 跨批次关系边
        "all_edges": merged_batch_edges + cross_batch_relation.get("cross_edges", []),  # 所有边
        "overall_structure": f"""
        1. 批次内关系：共{len(merged_batch_edges)}条有效边，覆盖所有批次内视频关联；
        2. 跨批次关系：共{len(cross_batch_relation.get("cross_edges", []))}条有效边，覆盖跨批次视频关联；
        3. 总关系边数：{len(merged_batch_edges) + len(cross_batch_relation.get("cross_edges", []))}条；
        4. 总节点数：{len(merged_nodes)}个。
        """,
        "coverage_check": {
            "total_videos": total_videos,
            "batch_coverage": [len(batch_rel.get("nodes", [])) for batch_rel in batch_relations],
            "cross_coverage": cross_batch_relation.get("cross_coverage", []),
            "merged_node_names": [safe_extract_video_name(node) for node in merged_nodes if safe_extract_video_name(node)]
        },
        "batch_info": {
            "batch_size": BATCH_SIZE,
            "total_batches": len(batch_relations),
            "api_timeout": API_TIMEOUT,
            "retry_config": {"retry_times": RETRY_TIMES, "retry_delay": RETRY_DELAY}
        }
    }

    # 8. 保存结果（适配你的路径）
    save_json_file(final_relation, RELATION_NETWORK_OUTPUT)
    
    # 生成大模型输入文件（复用参考代码的结构，适配你的字段）
    final_input = {
        "metadata": {
            "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_id": video_id,
            "video_count": total_videos,
            "description": "批次内+跨批次完整子视频关系网（兼容多字段，带容错）"
        },
        "subvideo_summaries": summary_data,
        "custom_knowledge_graph": custom_kg,
        "subvideo_relation_network": final_relation
    }
    save_json_file(final_input, FINAL_LLM_INPUT)
    
    # 9. 验证覆盖情况（完全复用参考代码的验证逻辑）
    all_video_set = set(sorted_video_names)
    merged_node_names = set([safe_extract_video_name(node) for node in merged_nodes])
    merged_node_names = set([name for name in merged_node_names if name])
    missing_videos = all_video_set - merged_node_names
    
    if missing_videos:
        logger.warning(f"⚠️ 警告：缺失视频 {sorted(list(missing_videos))}")
    else:
        logger.info(f"✅ 验证通过：所有 {total_videos} 个子视频均覆盖，包含跨批次关系！")

    logger.info("\n🎉 完整关系网生成完成！")
    return final_relation

# ====================== 测试入口（修复custom_kg路径，适配你的实际路径） ======================
if __name__ == "__main__":
    # 测试参数：替换为你的实际路径
    TEST_VIDEO_ID = "63"
    TEST_WORK_DIR = f"/root/video_platform/video_data/{TEST_VIDEO_ID}"
    # ✅ 修复：custom_kg路径适配每个视频的独立目录（和之前生成的路径一致）
    TEST_CUSTOM_KG_PATH = os.path.join(TEST_WORK_DIR, f"custom_kg_{TEST_VIDEO_ID}.json")

    # 额外校验：确保custom_kg文件存在，避免加载空文件
    if not os.path.exists(TEST_CUSTOM_KG_PATH):
        logger.error(f"❌ 自定义知识图谱文件不存在：{TEST_CUSTOM_KG_PATH}")
        logger.info(f"💡 请先运行生成custom_kg的代码，确保文件存在后再执行本脚本")
    else:
        try:
            result = generate_relation_network(TEST_VIDEO_ID, TEST_WORK_DIR, TEST_CUSTOM_KG_PATH)
            logger.info(f"\n📁 最终关系网文件：{os.path.join(TEST_WORK_DIR, 'relation_network.json')}")
            logger.info(f"📁 final_llm_input.json：{os.path.join(TEST_WORK_DIR, 'final_llm_input.json')}")
        except Exception as e:
            logger.error("❌ 测试失败", exc_info=True)