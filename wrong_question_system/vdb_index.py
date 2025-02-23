import os
import json
import numpy as np
import faiss
from log_config import get_logger

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

log_path = config.get("log_path")
logger = get_logger(__name__, log_file=log_path)

class VectorDBIndex:
    """
    VectorDBIndex 封装向量数据库索引的构建和搜索功能。
    当前实现基于 FAISS，但通过 engine 参数可扩展支持其他向量数据库。
    """
    def __init__(self, engine="faiss"):
        self.engine = engine.lower()
        self.embeddings = None  # 所有向量列表
        self.metadata = None    # 与向量一一对应的元数据列表
        self.index = None       # 向量数据库索引

    def load_all_embeddings(self, embeddings_dir):
        """
        从指定目录中加载所有 .emb 文件，返回两个列表：
          - all_embeddings: 每个元素为一个 embedding 向量（列表或数组）
          - metadata: 每个元素为字典，包含所属文档名和题目（chunk）编号
        """
        all_embeddings = []
        metadata = []
        for filename in os.listdir(embeddings_dir):
            if filename.endswith(".emb"):
                filepath = os.path.join(embeddings_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)  # 格式示例：{"1": [向量], "2": [向量], ...}
                    doc_name = os.path.splitext(filename)[0]
                    for chunk_id, embedding in data.items():
                        all_embeddings.append(embedding)
                        metadata.append({"doc": doc_name, "chunk_id": chunk_id})
                    logger.info("加载 %s 中的 %d 个 embedding", filename, len(data))
                except Exception as e:
                    logger.error("加载文件 %s 出错: %s", filename, e, exc_info=True)
        if not all_embeddings:
            logger.error("没有加载到任何 embedding")
        self.embeddings = all_embeddings
        self.metadata = metadata
        return all_embeddings, metadata

    def build_index(self):
        """
        根据加载的 embedding 构造向量数据库索引。
        当前采用 FAISS 的 IndexFlatIP 内积索引实现。
        """
        if self.embeddings is None:
            logger.error("没有加载到 embedding 数据，无法构建索引")
            return None
        vectors = np.array(self.embeddings).astype("float32")
        dimension = vectors.shape[1]
        if self.engine == "faiss":
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(vectors)
            logger.info("使用 FAISS 构建索引完成，向量数: %d, 维度: %d", vectors.shape[0], dimension)
        else:
            logger.error("不支持的向量数据库引擎: %s", self.engine)
            raise NotImplementedError("当前只支持 FAISS")
        return self.index

    def search(self, query_vector, k=5):
        """
        根据输入的查询向量进行相似性搜索，返回前 k 个匹配结果。
        :param query_vector: numpy 数组，形状 (1, dimension)
        :param k: 返回的最近邻数量
        :return: 包含元数据和距离的结果列表
        """
        if self.index is None:
            logger.error("索引尚未构建，无法进行搜索")
            return None
        distances, indices = self.index.search(query_vector, k)
        results = []
        for i, idx in enumerate(indices[0]):
            result = {
                "metadata": self.metadata[idx],
                "distance": float(distances[0][i])
            }
            results.append(result)
        return results
