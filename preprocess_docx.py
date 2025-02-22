import docx2txt
import os
import re
import docx

# --- 为图像描述模型做准备 ---
import torch
from PIL import Image, UnidentifiedImageError

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
# 这是官方提供的多模态辅助工具, 需自行放在本地 qwen_vl_utils.py
# 或安装 Qwen-VL 源码
from qwen_vl_utils import process_vision_info


##############################################
# 1) 加载图像描述模型 (Qwen2.5-VL-7B-Instruct)
##############################################

def load_chinese_image_captioning_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    加载一个中文多模态模型 Qwen2.5-VL-7B-Instruct, 用于看图说话。
    返回 (processor, model, device)。
    
    注意:
      - pip install --upgrade transformers accelerate safetensors pillow
      - 并接受 Qwen 模型协议, 否则无法下载/加载权重。
      - 必须使用 Qwen2_5_VLForConditionalGeneration, 它才有 .generate()。
    """
    # 处理器
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # 模型: Qwen2_5_VLForConditionalGeneration => 带有生成头, 可用 model.generate()
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
    内部使用了官方的 apply_chat_template / process_vision_info。

    prompt: 用户希望对图片说的话, 默认 "请用中文描述这张图片"。
    """
    # 尝试用 Pillow 打开图片
    try:
        image = Image.open(image_path).convert("RGB")
    except (OSError, UnidentifiedImageError):
        return "[无法识别的图片格式]"
    
    # 如果宽或高 < 28, 直接跳过或做放大
    w, h = image.size
    if w < 28 or h < 28:
        return "[图片太小，无法处理]"

    # 构造 messages, 与官方示例兼容
    # "type": "image", "image": <本地路径或者URL>
    # "type": "text",  "text": <用户问题>
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,  # 这里是本地图片路径
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # 1) 先把多轮对话 messages 转成可解析文本
    text = processor.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )

    # 2) 分析 messages, 提取 image_inputs / video_inputs
    image_inputs, video_inputs = process_vision_info(messages)

    # 3) 用 processor 把 "文本 + 图像" 打包成可传给模型的输入
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    # 4) 调用 model.generate() 推理
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )

    # 5) 解码输出, 去掉输入部分
    #    processor里往往在 input_ids 前面有指令, 需截掉
    #    这里简化写法: 截取, 并 decode
    generated_ids_trimmed = []
    for in_ids, out_ids in zip(inputs.input_ids, outputs):
        generated_ids_trimmed.append(out_ids[len(in_ids) :])

    caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    if caption:
        return caption[0].strip()
    else:
        return ""


##############################################
# 2) 从 docx 提取文本 & 图片
##############################################

def extract_text_and_images_from_docx(docx_file, image_dir):
    """
    使用 python-docx 从 docx 中读取所有段落,
    当段落里遇到行内图片时, 导出图片并在文本中插入 [imageX.png] 占位符。
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    doc_object = docx.Document(docx_file)

    image_counter = 0
    final_lines = []

    # 遍历文档中的所有段落
    for paragraph in doc_object.paragraphs:
        paragraph_chunks = []

        for run in paragraph.runs:
            # 在 run._element 里查找 <w:drawing>
            drawing_el = run._element.xpath('.//*[local-name()="drawing"]')
            if drawing_el:
                # 再找所有 <...:blip>
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
                        # 在文本中插入占位符
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

    return "\n".join(final_lines)


##############################################
# 3) 遇到图片占位符就调用千问-VL生成中文描述
##############################################

def replace_image_placeholders_with_captions(text, image_dir,
                                             processor, model, device):
    """
    在给定 text 中查找形如 [image1.png] 的占位符,
    用 Qwen2.5-VL 生成中文描述, 并将占位符替换掉。
    """
    pattern = re.compile(r'\[image(\d+)\.png\]', re.IGNORECASE)

    def _replace_func(match):
        image_number = match.group(1)
        image_filename = f"image{image_number}.png"
        image_path = os.path.join(image_dir, image_filename)

        print(f"正在处理图片 {image_path} ...")
        if os.path.exists(image_path):
            # 调用多模态推理
            caption = chinese_image_caption_inference(image_path, processor, model, device)
            print("生成的中文描述: ", caption)
            return f"[图片描述] {caption}"
        else:
            return f"[图片 {image_filename} 未找到]"

    return pattern.sub(_replace_func, text)


##############################################
# 4) 按关键词分块
##############################################

def chunk_exam_geography(raw_text):
    BOUNDARY_KEYWORDS = [
        "读图", "如图", "下图", "材料", "完成下面小题", "回答下列问题",
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
# 主程序示例
##############################################

if __name__ == "__main__":
    docx_file = "docx/2021dongcheng.docx"
    image_dir = "extracted_images"

    # 1) 加载 "千问2.5-VL" (中文多模态)
    processor, model, device = load_chinese_image_captioning_model(
        model_name="/nvme0/models/Qwen/Qwen2.5-VL-7B-Instruct/"  # 或本地路径
    )

    # 2) 提取文本 + 图片(插入占位符)
    text_with_placeholders = extract_text_and_images_from_docx(docx_file, image_dir)
    print("===== 从 docx 中提取到的文本 (含占位符) =====")
    print(text_with_placeholders)

    # 3) 调用千问多模态生成中文描述, 替换图片占位符
    text_with_captions = replace_image_placeholders_with_captions(
        text_with_placeholders,
        image_dir,
        processor, model, device
    )

    print("\n\n===== 替换后（带中文图片描述）的文本 =====")
    print(text_with_captions)

    # 4) 按关键词分块
    question_chunks = chunk_exam_geography(text_with_captions)

    print("\n\n===== 分块结果 =====")
    for i, chunk in enumerate(question_chunks, start=1):
        print(f"\n--- 第{i}块 ---")
        print(chunk)
