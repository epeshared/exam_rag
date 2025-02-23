# main.py
import os
import json
from wrong_question_system.log_config import get_logger
from wrong_question_system.docx_parser import extract_text_and_images_from_docx, replace_image_placeholders_with_captions, remove_line_numbering
from wrong_question_system.image_caption import load_chinese_image_captioning_model
from wrong_question_system.text_chunk import chunk_exam_geography, extract_chunk_relationships
from wrong_question_system.embedding import load_embedding_model, compute_chunk_embedding, save_embeddings

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

log_path = config.get("log_path", "logs/logs.log")
logger = get_logger(__name__, log_file=log_path)

def save_relationships(docx_file, relationships, output_path):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"docx_file": docx_file, "chunks": relationships}, f, ensure_ascii=False, indent=4)
        logger.info("关系信息已保存到 %s", output_path)
    except Exception as e:
        logger.error("保存关系信息时出错: %s", e, exc_info=True)

def save_chunks_as_files(docx_file, chunks, base_output_dir):
    base = os.path.splitext(os.path.basename(docx_file))[0]
    output_dir = os.path.join(base_output_dir, base)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("创建 chunks 输出目录: %s", output_dir)
    for idx, chunk in enumerate(chunks, start=1):
        filename = f"chunk_{idx}.txt"
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(chunk)
            logger.info("保存 chunk %d 到 %s", idx, filepath)
        except Exception as e:
            logger.error("保存 chunk %d 时出错: %s", idx, e, exc_info=True)

def process_geo_test_paper(docx_file, config, class_name):
    try:
        base_image_dir = os.path.join(config["base_image_dir"], class_name)
        base_chunks_output_dir = os.path.join(config["base_chunks_output_dir"], class_name)
        relationships_output_file = config["relationships_output_file"]
        embedding_output_dir = os.path.join(config["embedding_output_dir"], class_name)
        qwen_model_name = config["vl_model_path"]
        embedding_model_path = config["embedding_model_path"]
        logger.info("开始处理文件: %s", docx_file)

        # 图像描述模型与 DOCX 解析
        processor, model, device = load_chinese_image_captioning_model(qwen_model_name)
        text_with_placeholders, file_image_dir = extract_text_and_images_from_docx(docx_file, base_image_dir)
        logger.debug("提取到的文本(含占位符):\n%s", text_with_placeholders)

        # 替换图片占位符，并去除行首编号
        text_with_captions = replace_image_placeholders_with_captions(text_with_placeholders, file_image_dir, processor, model, device)
        text_with_captions = remove_line_numbering(text_with_captions)
        logger.debug("替换后(带中文图片描述)的文本:\n%s", text_with_captions)

        # 文本分块与关系提取
        question_chunks = chunk_exam_geography(text_with_captions)
        for i, chunk in enumerate(question_chunks, start=1):
            logger.debug("第%d块内容:\n%s", i, chunk)
        relationships = extract_chunk_relationships(question_chunks)
        save_relationships(docx_file, relationships, relationships_output_file)
        save_chunks_as_files(docx_file, question_chunks, base_chunks_output_dir)

        # Embedding 计算
        embed_tokenizer, embed_model, embed_device = load_embedding_model(embedding_model_path)
        embeddings = {}
        for idx, chunk in enumerate(question_chunks, start=1):
            emb = compute_chunk_embedding(chunk, embed_tokenizer, embed_model, embed_device)
            embeddings[idx] = emb
        save_embeddings(docx_file, embeddings, output_dir=embedding_output_dir)
        logger.info("处理文件 %s 完成", docx_file)
    except Exception as e:
        logger.error("处理文件 %s 时出现未捕获错误: %s", docx_file, e, exc_info=True)

def process_geo_files(config):
    try:
        upload_file_path = config["upload_file_path"]
        geo_dir = os.path.join(upload_file_path, "地理")
        for filename in os.listdir(geo_dir):
            if filename.lower().endswith(".docx"):
                docx_file_path = os.path.join(geo_dir, filename)
                process_geo_test_paper(docx_file_path, config, "地理")
    except Exception as e:
        logger.error("处理 geo 文件夹时出现错误: %s", e, exc_info=True)

if __name__ == "__main__":
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.critical("加载配置文件出错: %s", e, exc_info=True)
        raise

    try:
        process_geo_files(config)
    except Exception as e:
        logger.critical("主程序运行时发生致命错误: %s", e, exc_info=True)
