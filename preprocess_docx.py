import docx2txt
import os
import re
import docx
import json

# --- 为图像描述模型做准备 ---
import torch
from PIL import Image, UnidentifiedImageError

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
from qwen_vl_utils import process_vision_info


##############################################
# 1) 加载图像描述模型 (Qwen2.5-VL-7B-Instruct)
##############################################

def load_chinese_image_captioning_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    加载一个中文多模态模型 Qwen2.5-VL-7B-Instruct, 用于看图说话。
    返回 (processor, model, device)。
    """
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model, device


def chinese_image_caption_inference(
    image_path, processor, model, device,
    prompt="请用中文描述这张图片"
):
    """
    给定单张图片的本地路径, 使用 Qwen2.5-VL-7B-Instruct 生成中文图像描述。
    内部使用官方的 apply_chat_template / process_vision_info。
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except (OSError, UnidentifiedImageError):
        return "[无法识别的图片格式]"
    
    w, h = image.size
    if w < 28 or h < 28:
        return "[图片太小，无法处理]"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )

    generated_ids_trimmed = []
    for in_ids, out_ids in zip(inputs.input_ids, outputs):
        generated_ids_trimmed.append(out_ids[len(in_ids):])
    
    caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    return caption[0].strip() if caption else ""


##############################################
# 2) 从 docx 提取文本 & 图片
##############################################

def extract_text_and_images_from_docx(docx_file, base_image_dir):
    """
    使用 python-docx 从 docx 中读取所有段落,
    当段落里遇到行内图片时, 导出图片并在文本中插入 [imageX.png] 占位符。
    
    修改点：在 base_image_dir 下为每个文件创建子目录保存图片。
    """
    # 根据 docx 文件名创建子目录
    base = os.path.splitext(os.path.basename(docx_file))[0]
    image_dir = os.path.join(base_image_dir, base)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

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
    return "\n".join(final_lines), image_dir  # 返回文本和图片存放的子目录路径


##############################################
# 3) 替换图片占位符, 调用千问-VL生成中文描述
##############################################

def replace_image_placeholders_with_captions(text, image_dir, processor, model, device):
    """
    在给定 text 中查找形如 [imageX.png] 的占位符,
    用 Qwen2.5-VL 生成中文描述, 并将占位符替换掉。
    被替换的文字末尾会追加 "[/]"。
    """
    pattern = re.compile(r'\[image(\d+)\.png\]', re.IGNORECASE)

    def _replace_func(match):
        image_number = match.group(1)
        image_filename = f"image{image_number}.png"
        image_path = os.path.join(image_dir, image_filename)
        print(f"正在处理图片 {image_path} ...")
        if os.path.exists(image_path):
            caption = chinese_image_caption_inference(image_path, processor, model, device)
            print("生成的中文描述: ", caption)
            return f"[{image_path}] {caption} [/]"
        else:
            return f"[图片 {image_filename} 未找到] [/]"
    return pattern.sub(_replace_func, text)



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
    return chunks


##############################################
# 5) 去除行首编号（可选）
##############################################

def remove_line_numbering(text: str) -> str:
    text_no_numbers = re.sub(r'\d+\.\s+', '', text)
    return text_no_numbers


##############################################
# 6) 提取 chunk 与图片的关系
##############################################

def extract_chunk_relationships(chunks):
    """
    对于每个 chunk, 提取其中所有图片占位符 (例如 "[image1.png]")，
    返回一个列表，每个元素为字典，记录 chunk 序号、chunk 文本和图片列表。
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
    return relationships


def save_relationships(docx_file, relationships, output_path):
    """
    保存文件名及其 chunk 与图片关系为 JSON 文件。
    """
    data = {
        "docx_file": docx_file,
        "chunks": relationships
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_chunks_as_files(docx_file, chunks, base_output_dir):
    """
    将每个 chunk 单独保存为文本文件，
    在 base_output_dir 下以 docx 文件的基本名称创建子目录，
    文件名格式为: 原始文件名_chunk_{i}.txt
    """
    base = os.path.splitext(os.path.basename(docx_file))[0]
    output_dir = os.path.join(base_output_dir, base)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, chunk in enumerate(chunks, start=1):
        filename = f"chunk_{idx}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chunk)


##############################################
# 主程序示例
##############################################

if __name__ == "__main__":
    # 需要处理的 docx 文件
    docx_file = "docx/地理/2021北京东城高三一模地理（教师版）.docx"
    
    # 基础输出目录
    base_image_dir = "extracted_images"
    base_chunks_output_dir = "chunks_texts"
    relationships_output_file = "relationships.json"

    # 1) 加载 "千问2.5-VL" (中文多模态)
    processor, model, device = load_chinese_image_captioning_model(
        model_name="/nvme0/models/Qwen/Qwen2.5-VL-7B-Instruct/"
    )

    # 2) 提取文本 + 图片 (插入占位符)
    # 提取函数返回文本以及该文件对应的图片目录（子目录）
    text_with_placeholders, file_image_dir = extract_text_and_images_from_docx(docx_file, base_image_dir)
    print("===== 提取到的文本 (含占位符) =====")
    print(text_with_placeholders)

    # 3) 替换占位符，调用千问多模态生成中文描述
    text_with_captions = replace_image_placeholders_with_captions(
        text_with_placeholders,
        file_image_dir,
        processor, model, device
    )
    
    # 可选：去除行首编号
    text_with_captions = remove_line_numbering(text_with_captions)
    print("\n===== 替换后（带中文图片描述）的文本 =====")
    print(text_with_captions)

    # 4) 按关键词分块
    question_chunks = chunk_exam_geography(text_with_captions)
    print("\n===== 分块结果 =====")
    for i, chunk in enumerate(question_chunks, start=1):
        print(f"\n--- 第{i}块 ---")
        print(chunk)

    # 5) 提取 chunk 与图片的关系
    relationships = extract_chunk_relationships(question_chunks)

    # 6) 保存关系信息为 JSON 文件
    save_relationships(docx_file, relationships, relationships_output_file)
    print(f"\n关系信息已保存到 {relationships_output_file}")

    # 7) 将每个 chunk 单独保存为文本文件
    save_chunks_as_files(docx_file, question_chunks, base_chunks_output_dir)
    print(f"每个 chunk 的文本已保存到目录 {base_chunks_output_dir}")
