from typing import Iterable


SUBJECT_KEYWORDS = {
    "计算机网络": [
        "计算机网络",
        "应用层",
        "运输层",
        "tcp",
        "udp",
        "dns",
        "ftp",
        "http",
        "https",
        "telnet",
        "dhcp",
        "ip地址",
        "域名",
        "端口号",
        "osi",
        "万维网",
        "电子邮件",
    ],
    "程序设计": [
        "程序设计",
        "编程",
        "算法",
        "数据结构",
        "python",
        "java",
        "c语言",
        "c++",
        "函数",
        "变量",
        "循环",
        "数组",
        "链表",
        "递归",
        "指针",
        "面向对象",
    ],
    "高等数学": [
        "高等数学",
        "微积分",
        "导数",
        "积分",
        "极限",
        "函数",
        "定积分",
        "不定积分",
        "偏导",
        "微分",
        "级数",
        "矩阵",
        "线性代数",
        "概率论",
    ],
    "物理": [
        "物理",
        "力学",
        "电学",
        "电磁",
        "热学",
        "光学",
        "波动",
        "动量",
        "能量",
        "加速度",
        "速度",
        "牛顿",
        "电场",
        "磁场",
    ],
    "化学": [
        "化学",
        "离子",
        "分子",
        "原子",
        "化学反应",
        "氧化还原",
        "酸碱",
        "有机",
        "无机",
        "化合物",
        "元素",
        "摩尔",
    ],
    "生物": [
        "生物",
        "细胞",
        "遗传",
        "dna",
        "rna",
        "蛋白质",
        "酶",
        "生态",
        "光合作用",
        "呼吸作用",
        "基因",
    ],
    "英语": [
        "英语",
        "语法",
        "单词",
        "词汇",
        "听力",
        "阅读",
        "写作",
        "翻译",
        "时态",
        "从句",
        "口语",
        "英文",
    ],
    "语文": [
        "语文",
        "文言文",
        "现代文",
        "诗歌",
        "作文",
        "修辞",
        "阅读理解",
        "古诗",
        "散文",
        "小说",
    ],
    "历史": [
        "历史",
        "朝代",
        "革命",
        "战争",
        "制度",
        "近代史",
        "古代史",
        "世界史",
        "史料",
    ],
    "地理": [
        "地理",
        "气候",
        "地形",
        "板块",
        "人口",
        "城市化",
        "农业区位",
        "工业区位",
        "经纬度",
    ],
    "政治": [
        "政治",
        "哲学",
        "经济生活",
        "政治生活",
        "文化生活",
        "法律",
        "宪法",
        "马克思主义",
        "辩证法",
    ],
}


def _iter_texts(texts: Iterable[object]) -> Iterable[str]:
    for item in texts:
        value = str(item or "").strip()
        if value:
            yield value.lower()


def infer_subject_hint(*texts: object) -> str:
    corpus = "\n".join(_iter_texts(texts))
    if not corpus:
        return ""

    best_subject = ""
    best_score = 0
    for subject, keywords in SUBJECT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in corpus:
                score += max(1, len(keyword) // 2)
        if score > best_score:
            best_subject = subject
            best_score = score

    return best_subject if best_score >= 2 else ""


def build_subject_constraint_text(subject_hint: str) -> str:
    subject_hint = str(subject_hint or "").strip()
    if subject_hint:
        return (
            f"内容必须限制在「{subject_hint}」学科范围内。"
            "不要跨到其他学科补充背景，不要把相似术语误判成别的课程知识。"
        )
    return "请先根据材料判断主学科，并始终限制在该学科范围内，避免跨学科扩写或乱联想。"
