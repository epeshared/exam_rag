# text_chunk.py
import re
from log_config import get_logger

logger = get_logger(__name__, "logs/logs.log")

def chunk_exam_geography(raw_text):
    BOUNDARY_KEYWORDS = [
        "完成下面小题", "回答下面小题", "阅读材料，回答问题", "阅读资料，回答下列问题",
        "读资料，回答下列问题", "阅读资料，回答下列问题"
    ]
    lines = raw_text.splitlines()
    chunks = []
    in_answer_section = False
    last_boundary_index = 0

    def is_boundary_line(line):
        for kw in BOUNDARY_KEYWORDS:
            if kw in line:
                return True
        return False

    for i, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        if "参考答案" in text or "答案" in text:
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
