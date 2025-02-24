import re
from log_config import get_logger
import json
import os

# 建议将配置加载和 logger 初始化放在主程序中，然后传入此模块
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
log_path = config.get("log_path", "logs/streamlit.log")
logger = get_logger(__name__, log_file=log_path)

# 定义固定关键词
BOUNDARY_KEYWORDS = [
    "完成下面小题", "回答下面小题", "阅读材料，回答问题", "阅读资料，回答下列问题",
    "读资料，回答下列问题", "阅读资料，回答下列问题", "回答下列问题", "完成下列要求",
    "读图，回答下列问题"
]

# 定义正则模式列表，后续需要添加其他模式时，只需在此列表中加入即可
BOUNDARY_REGEXES = [
    r"完成\d+(、\d+)*题"
    # 如果以后需要匹配其他格式，可以继续添加，例如：r"其他模式"
]

def is_boundary_line(line, keywords=BOUNDARY_KEYWORDS, regexes=BOUNDARY_REGEXES):
    # 先判断固定关键词
    for kw in keywords:
        if kw in line:
            return True
    # 遍历所有正则模式进行匹配
    for pattern in regexes:
        if re.search(pattern, line):
            return True
    return False

def chunk_exam_file(raw_text, keywords=BOUNDARY_KEYWORDS):
    """
    根据设定的关键词对试卷文本进行分块。
    仅当行内容完全为“参考答案”或“答案”，或以“参考答案”开头时，才认为进入答案部分，
    否则不会终止分块，从而保留后续题目。
    :param raw_text: 待处理的原始文本
    :param keywords: 分块边界关键词列表
    :return: 分块后的文本列表
    """
    lines = raw_text.splitlines()
    chunks = []
    in_answer_section = False
    last_boundary_index = 0

    for i, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        # 修改判断条件：只有整行为“参考答案”或“答案”，或者以“参考答案”开头时，才认为进入答案区
        if text in ["参考答案", "答案"] or text.startswith("参考答案"):
            in_answer_section = True
            if last_boundary_index < i:
                chunk_lines = lines[last_boundary_index:i]
                cleaned = "\n".join(l.strip() for l in chunk_lines if l.strip())
                if cleaned:
                    chunks.append(cleaned)
            break
        if is_boundary_line(text):
            if i > last_boundary_index:
                chunk_lines = lines[last_boundary_index:i]
                cleaned = "\n".join(l.strip() for l in chunk_lines if l.strip())
                if cleaned:
                    chunks.append(cleaned)
            last_boundary_index = i
    if not in_answer_section and last_boundary_index < len(lines):
        chunk_lines = lines[last_boundary_index:]
        cleaned = "\n".join(l.strip() for l in chunk_lines if l.strip())
        if cleaned:
            chunks.append(cleaned)
    logger.info("文本分块完成，共分 %d 块", len(chunks))
    return chunks

def extract_chunk_relationships(chunks):
    """
    提取每个分块中的图片占位符信息，返回列表，每项包含分块编号、文本和图片标识列表。
    :param chunks: 分块后的文本列表
    :return: 关系列表，每项为字典 {chunk_index, chunk_text, images}
    """
    relationships = []
    pattern = re.compile(r'\[(image\d+\.png)\]')
    for idx, chunk in enumerate(chunks, start=1):
        images = pattern.findall(chunk)
        relationships.append({
            "chunk_index": idx,
            "chunk_text": chunk,
            "images": images
        })
    logger.info("提取 chunk 与图片关系完成")
    return relationships
