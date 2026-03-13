# -*- coding: utf-8 -*-
import requests
import json
from config import VLM_API_KEY, VLM_MODEL, VLM_TEMPERATURE, VLM_MAX_TOKENS, PRE_QUESTION_COUNT, POST_QUESTION_COUNT

class VLMClient:
    def __init__(self):
        self.api_key = VLM_API_KEY
        self.model = VLM_MODEL
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    
    def call_vlm(self, prompt):
        """调用通义千问API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "temperature": VLM_TEMPERATURE,
                "max_tokens": VLM_MAX_TOKENS
            }
        }
        response = requests.post(self.base_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"VLM API调用失败：{response.text}")
        
        result = response.json()
        return result['output']['choices'][0]['message']['content']
    
    def extract_knowledge_with_questions(self, asr_segments, ocr_result, segments):
        """提取知识点+生成前后提问"""
        # 拼接视频文本内容
        video_text = ""
        for seg in asr_segments:
            video_text += f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}\n"
        
        # 生成提示词
        prompt = f"""
        以下是视频的ASR转写内容：
        {video_text}
        
        视频已被分割为以下分段：
        {json.dumps(segments, ensure_ascii=False, indent=2)}
        
        请完成以下任务：
        1. 为每个分段提取1个核心知识点，包含：知识点ID、名称、描述、对应分段ID、开始/结束时间；
        2. 为每个知识点生成{PRE_QUESTION_COUNT}个预习问题（知识点前弹窗）、{POST_QUESTION_COUNT}个复习问题（知识点后弹窗）；
        3. 为每个问题提供标准答案。
        
        输出格式（JSON）：
        [
            {{
                "knowledge_id": "kp_1",
                "name": "知识点名称",
                "description": "知识点详细描述",
                "segment_id": "seg_1",
                "start": 0.0,
                "end": 30.0,
                "pre_questions": [
                    {{
                        "question_id": "pre_1",
                        "question": "预习问题1",
                        "answer": "标准答案1"
                    }},
                    ...
                ],
                "post_questions": [
                    {{
                        "question_id": "post_1",
                        "question": "复习问题1",
                        "answer": "标准答案1"
                    }},
                    ...
                ]
            }},
            ...
        ]
        """
        
        # 调用VLM生成结果
        result = self.call_vlm(prompt)
        # 清理结果（去除markdown格式）
        result = result.replace("```json", "").replace("```", "").strip()
        return json.loads(result)
    
    def answer_question(self, question, knowledge_points):
        """回答用户侧边栏提问"""
        prompt = f"""
        视频知识点：
        {json.dumps(knowledge_points, ensure_ascii=False, indent=2)}
        
        用户问题：{question}
        
        请基于视频知识点回答用户问题，要求：
        1. 回答准确、简洁，贴合视频内容；
        2. 列出相关的知识点ID和名称；
        3. 不要编造未提及的内容。
        
        输出格式：
        回答内容|||相关知识点列表（如：kp_1:知识点1,kp_2:知识点2）
        """
        
        result = self.call_vlm(prompt)
        answer, related = result.split("|||")
        # 解析相关知识点
        related_knowledge = []
        for item in related.split(','):
            if ':' in item:
                kp_id, kp_name = item.split(':', 1)
                related_knowledge.append({"id": kp_id.strip(), "name": kp_name.strip()})
        
        return answer.strip(), related_knowledge
    
    def analyze_user_answer(self, question, user_answer, video_id, knowledge_id):
        """解析用户答题（弹窗提交）"""
        prompt = f"""
        问题：{question}
        用户答案：{user_answer}
        
        请完成以下分析：
        1. 判断答案正确性（正确/部分正确/错误）；
        2. 给出详细解析；
        3. 提供标准答案；
        4. 给出学习建议。
        
        输出格式（JSON）：
        {{
            "correctness": "正确/部分正确/错误",
            "analysis": "详细解析",
            "true_answer": "标准答案",
            "suggestion": "学习建议"
        }}
        """
        
        result = self.call_vlm(prompt)
        result = result.replace("```json", "").replace("```", "").strip()
        return json.loads(result)