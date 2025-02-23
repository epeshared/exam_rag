import os
import re
import json
import logging
import logging.handlers
import docx
import torch
from PIL import Image, UnidentifiedImageError
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
from qwen_vl_utils import process_vision_info

# --- 先加载配置 ---
from log_config import get_logger
logger = get_logger(__name__, "logs/preprocess_geo.log")


##############################################
# 1) 加载图像描述模型 (Qwen2.5-VL-7B-Instruct)
##############################################

def load_chinese_image_captioning_model(model_name):
    try:
        logger.info("加载图像描述模型: %s", model_name)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("模型加载成功，使用设备: %s", device)
        return processor, model, device
    except Exception as e:
        logger.error("加载图像描述模型时发生错误: %s", e, exc_info=True)
        raise

def chinese_image_caption_inference(image_path, processor, model, device, prompt="请用中文描述这张图片"):
    try:
        image = Image.open(image_path).convert("RGB")
    except (OSError, UnidentifiedImageError) as e:
        logger.error("无法打开图片 %s: %s", image_path, e, exc_info=True)
        return "[无法识别的图片格式]"

    w, h = image.size
    if w < 28 or h < 28:
        logger.warning("图片 %s 太小 (%dx%d)，无法处理", image_path, w, h)
        return "[图片太小，无法处理]"

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        generated_ids_trimmed = []
        for in_ids, out_ids in zip(inputs.input_ids, outputs):
            generated_ids_trimmed.append(out_ids[len(in_ids):])
        caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        result = caption[0].strip() if caption else ""
        logger.info("生成图片描述成功: %s", result)
        return result
    except Exception as e:
        logger.error("生成图片描述失败: %s", e, exc_info=True)
        return "[描述生成失败]"

##############################################
# 2) 从 docx 提取文本 & 图片
##############################################

def extract_text_and_images_from_docx(docx_file, base_image_dir):
    try:
        base = os.path.splitext(os.path.basename(docx_file))[0]
        image_dir = os.path.join(base_image_dir, base)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            logger.info("创建图片保存目录: %s", image_dir)

        doc_object = docx.Document(docx_file)
        image_counter = 0
        final_lines = []

        for paragraph in doc_object.paragraphs:
            paragraph_chunks = []
            for run in paragraph.runs:
                drawing_el = run._element.xpath('.//*[local-name()="drawing"]')
                if drawing_el:
                    blip_elems = run._element.xpath('.//*[local-name()="blip"]')
                    if blip_elems:
                        blip = blip_elems[0]
                        rid = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                        if rid:
                            image_counter += 1
                            part = doc_object.part.related_parts[rid]
                            image_filename = f"image{image_counter}.png"
                            out_path = os.path.join(image_dir, image_filename)
                            with open(out_path, "wb") as f:
                                f.write(part._blob)
                            logger.info("提取图片: %s", out_path)
                            paragraph_chunks.append(f"[{image_filename}]")
                        else:
                            paragraph_chunks.append("[image_no_rid]")
                    else:
                        paragraph_chunks.append("[image_no_blip]")
                else:
                    paragraph_chunks.append(run.text)
            line_text = "".join(paragraph_chunks).strip()
            if line_text:
                final_lines.append(line_text)
        return "\n".join(final_lines), image_dir
    except Exception as e:
        logger.error("提取 docx 文本和图片时出错: %s", e, exc_info=True)
        raise

##############################################
# 3) 替换图片占位符, 调用千问-VL生成中文描述
##############################################

def replace_image_placeholders_with_captions(text, image_dir, processor, model, device):
    pattern = re.compile(r'\[image(\d+)\.png\]', re.IGNORECASE)

    def _replace_func(match):
        image_number = match.group(1)
        image_filename = f"image{image_number}.png"
        image_path = os.path.join(image_dir, image_filename)
        logger.info("正在处理图片: %s", image_path)
        if os.path.exists(image_path):
            caption = chinese_image_caption_inference(image_path, processor, model, device)
            return f"[{image_path}] {caption} [/]"

        else:
            logger.warning("图片未找到: %s", image_path)
            return f"[图片 {image_filename} 未找到] [/]"
    try:
        new_text = pattern.sub(_replace_func, text)
        return new_text
    except Exception as e:
        logger.error("替换图片占位符时出错: %s", e, exc_info=True)
        return text

##############################################
# 4) 按关键词分块
##############################################

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

##############################################
# 5) 去除行首编号（可选）
##############################################

def remove_line_numbering(text: str) -> str:
    return re.sub(r'\d+\.\s+', '', text)

##############################################
# 6) 提取 chunk 与图片的关系
##############################################

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

def save_relationships(docx_file, relationships, output_path):
    data = {"docx_file": docx_file, "chunks": relationships}
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
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

##############################################
# 7) 加载文本 embedding 模型，并对每个 chunk 做 embedding
##############################################

def load_embedding_model(model_path):
    try:
        from transformers import AutoModel, AutoTokenizer
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

def compute_chunk_embedding(chunk_text, embed_tokenizer, embed_model, embed_device, max_length=512, stride=256):
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

##############################################
# 主程序示例
##############################################

def process_geo_test_paper(docx_file, config, class_name):
    try:
        base_image_dir = os.path.join(config["base_image_dir"], class_name)
        base_chunks_output_dir = os.path.join(config["base_chunks_output_dir"], class_name)
        relationships_output_file = config["relationships_output_file"]
        embedding_output_dir = os.path.join(config["embedding_output_dir"], class_name)
        qwen_model_name = config["vl_model_path"]
        embedding_model_path = config["embedding_model_path"]
        logger.info("开始处理文件: %s", docx_file)

        processor, model, device = load_chinese_image_captioning_model(qwen_model_name)
        text_with_placeholders, file_image_dir = extract_text_and_images_from_docx(docx_file, base_image_dir)
        logger.debug("提取到的文本(含占位符):\n%s", text_with_placeholders)

        text_with_captions = replace_image_placeholders_with_captions(text_with_placeholders, file_image_dir, processor, model, device)
        text_with_captions = remove_line_numbering(text_with_captions)
        logger.debug("替换后(带中文图片描述)的文本:\n%s", text_with_captions)

        question_chunks = chunk_exam_geography(text_with_captions)
        for i, chunk in enumerate(question_chunks, start=1):
            logger.debug("第%d块内容:\n%s", i, chunk)

        relationships = extract_chunk_relationships(question_chunks)
        save_relationships(docx_file, relationships, relationships_output_file)
        save_chunks_as_files(docx_file, question_chunks, base_chunks_output_dir)

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
