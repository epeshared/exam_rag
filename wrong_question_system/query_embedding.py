# query_embedding.py
import requests
import numpy as np
from log_config import get_logger
from embedding import load_embedding_model, compute_chunk_embedding

import json
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

log_path = config.get("log_path", "logs/logs.log")
logger = get_logger(__name__, log_file=log_path)

def get_query_embedding_deepseek(query: str) -> list:
    """
    调用外部 API（例如 Ollama 部署的 Deepseek r1 671b 模型）生成查询向量。
    请根据实际情况修改 URL 和请求格式。
    """
    url = "http://localhost:8000/api/embedding"  # 修改为实际 API 地址
    payload = {"text": query}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        embedding = result.get("embedding")
        if embedding is None:
            logger.error("响应中未包含 embedding 字段: %s", result)
        else:
            logger.info("Deepseek 查询向量生成成功")
        return embedding
    except Exception as e:
        logger.error("调用 Deepseek API 获取查询向量失败: %s", e, exc_info=True)
        return None

def get_query_embedding_bge(query: str) -> list:
    """
    直接使用 BGE-large-zh-v1.5 模型生成查询向量，
    与生成题库 embedding 时使用相同的模型。
    """
    # 模型路径可配置为参数，这里直接写死
    embed_tokenizer, embed_model, embed_device = load_embedding_model("/nvme0/models/BAAI/bge-large-zh-v1.5")
    embedding = compute_chunk_embedding(query, embed_tokenizer, embed_model, embed_device)
    if embedding is not None:
        logger.info("BGE 查询向量生成成功")
    else:
        logger.error("BGE 查询向量生成失败")
    return embedding
