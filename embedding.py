# embedding.py
import os
import json
import torch
from transformers import AutoModel, AutoTokenizer

from log_config import get_logger

logger = get_logger(__name__, "logs/logs.log")

def load_embedding_model(model_path):
    try:
        logger.info("加载 embedding 模型: %s", model_path)
        embed_tokenizer = AutoTokenizer.from_pretrained(model_path)
        embed_model = AutoModel.from_pretrained(model_path)
        embed_model.eval()
        embed_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embed_model.to(embed_device)
        logger.info("Embedding 模型加载成功，使用设备: %s", embed_device)
        return embed_tokenizer, embed_model, embed_device
    except Exception as e:
        logger.error("加载 embedding 模型时出错: %s", e, exc_info=True)
        raise

def compute_chunk_embedding(chunk_text, embed_tokenizer, embed_model, embed_device="cpu", max_length=512, stride=256):
    try:
        inputs = embed_tokenizer(chunk_text, return_tensors="pt", truncation=False)
        inputs = inputs.to(embed_device)
        input_ids = inputs["input_ids"][0]
        embeddings = []
        for start in range(0, len(input_ids), stride):
            end = start + max_length
            window_ids = input_ids[start:end]
            window_inputs = {"input_ids": window_ids.unsqueeze(0)}
            with torch.no_grad():
                outputs = embed_model(**window_inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                window_emb = outputs.pooler_output
            else:
                window_emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(window_emb.squeeze(0))
        final_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        logger.info("Chunk embedding 计算完成")
        return final_embedding.squeeze().cpu().tolist()
    except Exception as e:
        logger.error("计算 chunk embedding 时出错: %s", e, exc_info=True)
        return []

def save_embeddings(docx_file, embeddings, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("创建 embedding 输出目录: %s", output_dir)
    base = os.path.splitext(os.path.basename(docx_file))[0]
    emb_filepath = os.path.join(output_dir, f"{base}.emb")
    try:
        with open(emb_filepath, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=4)
        logger.info("Embedding结果已保存到 %s", emb_filepath)
    except Exception as e:
        logger.error("保存 embedding 结果时出错: %s", e, exc_info=True)
