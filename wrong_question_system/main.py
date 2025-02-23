import os
import json
import numpy as np
from log_config import get_logger
from vdb_index import VectorDBIndex
from query_embedding import get_query_embedding_bge  # 或 get_query_embedding_deepseek

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

log_path = config.get("log_path")
logger = get_logger(__name__, log_file=log_path)

def process_query(query: str, config, class_name):
    """
    根据输入查询文本，加载指定类别下的 embedding，
    构建向量数据库索引，生成查询向量后进行相似性搜索，打印查询结果。
    """
    embeddings_dir = os.path.join(config["embedding_output_dir"], class_name)
    
    # 创建 VectorDBIndex 实例
    vdb = VectorDBIndex(engine="faiss")
    # 加载所有 embedding 并保存元数据
    vdb.load_all_embeddings(embeddings_dir)
    # 构建索引
    vdb.build_index()
    
    # 生成查询向量（使用 BGE 模型，保证查询向量与库向量一致）
    query_embedding = get_query_embedding_bge(query)
    if query_embedding is None:
        logger.error("查询向量生成失败，无法搜索")
        exit(1)
    query_vector = np.array(query_embedding, dtype="float32").reshape(1, -1)
    
    # 使用 vdb 的 search 方法查找相似向量
    results = vdb.search(query_vector, k=5)
    if results:
        print("相似题目查询结果:")
        for res in results:
            meta = res["metadata"]
            print(f"文档: {meta['doc']} 题目编号: {meta['chunk_id']}, 距离: {res['distance']:.4f}")
    else:
        print("查询失败，请检查日志")

if __name__ == "__main__":
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.critical("加载配置文件出错: %s", e, exc_info=True)
        raise

    process_query("地球的自转是什么意思？", config, "地理")
