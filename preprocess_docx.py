import docx2txt
import os
import re
import docx

# --- 为图像描述模型做准备 ---
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image, UnidentifiedImageError

##############################################
# 1) 加载图像描述模型
##############################################

def load_image_captioning_model(model_name="nlpconnect/vit-gpt2-image-captioning"):
    """
    加载一个开源的图像描述模型 (Image Captioning)。
    默认为 nlpconnect/vit-gpt2-image-captioning。
    
    返回 (model, feature_extractor, tokenizer, device)。
    """
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, feature_extractor, tokenizer, device


def image_caption_inference(image_path, model, feature_extractor, tokenizer, device):
    """
    给定单张图片的路径，调用上面加载的模型生成图像描述（英文短句）。
    """
    # 打开图片并转成 RGB
    print(image_path)
    try:
        image = Image.open(image_path).convert("RGB")
    except (OSError, UnidentifiedImageError) as e:
        # 无法识别该图片格式，返回占位描述
        return ""
    
    # 特征提取
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # 模型推理
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


##############################################
# 2) 从 docx 提取文本 & 图片
##############################################

# def extract_text_and_images_from_docx(docx_file, image_dir):
#     """
#     使用 docx2txt 将 docx 中的所有文字提取为字符串，并把图片保存到 image_dir。
#     返回的文本中会出现形如 [image1.png] [image2.png] 的占位符。
#     """
#     if not os.path.exists(image_dir):
#         os.makedirs(image_dir)

#     text = docx2txt.process(docx_file, image_dir)
#     return text

def extract_text_and_images_from_docx(docx_file, image_dir):
    """
    使用 python-docx 从 docx 中读取所有段落，
    当段落里遇到行内图片时，导出图片并在文本中插入 [imageX.png] 占位符。

    不使用 xpath 的 namespaces= 参数，而是通过 local-name() 匹配标签。
    """

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    doc = docx.Document(docx_file)

    image_counter = 0
    final_lines = []

    # 遍历文档中的所有段落
    for paragraph in doc.paragraphs:
        paragraph_chunks = []

        for run in paragraph.runs:
            # 在这个 run 的底层 XML (CT_R) 中，用 local-name()="drawing" 判断是否存在 <w:drawing>。
            drawing_el = run._element.xpath('.//*[local-name()="drawing"]')
            if drawing_el:
                # 再找所有 <...:blip> 节点。忽略命名空间前缀，直接用 local-name()="blip"。
                blip_elems = run._element.xpath('.//*[local-name()="blip"]')
                if blip_elems:
                    # 拿第一个 blip
                    blip = blip_elems[0]
                    # 从 embed 属性中获取关系 ID (rIdX)
                    rid = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                    if rid:
                        # 导出图片
                        image_counter += 1
                        part = doc.part.related_parts[rid]
                        # 这里统一用 .png 后缀；若需区分 jpg/png，可用 part.content_type 判断
                        image_filename = f"image{image_counter}.png"
                        out_path = os.path.join(image_dir, image_filename)
                        with open(out_path, "wb") as f:
                            f.write(part._blob)

                        # 在文本里插入占位符
                        paragraph_chunks.append(f"[{image_filename}]")
                    else:
                        paragraph_chunks.append("[image_no_rid]")
                else:
                    paragraph_chunks.append("[image_no_blip]")
            else:
                # 无 <w:drawing>，说明只是普通文本
                paragraph_chunks.append(run.text)

        line_text = "".join(paragraph_chunks).strip()
        if line_text:
            final_lines.append(line_text)

    return "\n".join(final_lines)


##############################################
# 3) 遇到图片占位符就调用图像描述模型替换成文字说明
##############################################

def replace_image_placeholders_with_captions(text, image_dir,
                                             model, feature_extractor, tokenizer, device):
    """
    在给定的 text 中查找形如 [image1.png] 的占位符，
    并使用 image_caption_inference(...) 生成描述文本，
    将占位符替换为对应描述。
    """

    # 匹配 [image1.png]、[image2.png] 这样的占位符
    pattern = re.compile(r'\[image(\d+)\.png\]', re.IGNORECASE)

    def _replace_func(match):
        image_number = match.group(1)  # 得到 "1" "2" 等数字
        image_filename = f"image{image_number}.png"
        image_path = os.path.join(image_dir, image_filename)

        if os.path.exists(image_path):
            # 如果有对应图片文件，则生成图像描述
            caption = image_caption_inference(image_path, model, feature_extractor, tokenizer, device)
            # 这里你也可以根据需要在描述前后加一些提示，如 “【图1】”
            return f"[图片描述] {caption}"
        else:
            # 没找到对应图片文件，就保留占位符，或返回别的信息
            return f"[图片 {image_filename} 未找到]"

    new_text = pattern.sub(_replace_func, text)
    return new_text


##############################################
# 4) 将替换后的文本按题号分块
##############################################

def extract_text_from_docx(docx_file, image_dir="extracted_images"):
    """
    用 docx2txt 从 docx 中提取纯文本，并把图片导出到 image_dir。
    返回带 [imageX.png] 占位符的文本。
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    text = docx2txt.process(docx_file, image_dir)
    return text

def chunk_exam_geography(raw_text):
    """
    将地理试卷文本分块，每块包含“描述、图片、几道选择题”。
    思路：
      - 以常见关键词(如 '读图','下图','如图','材料','完成下面小题','回答下列问题') 
        作为“块起始”标记。
      - 在下一个“块起始”或 '参考答案' 出现之前，所有行都属于同一个区块。
    
    返回一个列表，每个元素为一个 chunk（字符串）。
    """

    # 这里可以根据卷子的实际情况调整或增加关键字
    BOUNDARY_KEYWORDS = [
        "读图", "如图", "下图", "材料", "完成下面小题", "回答下列问题", 
        # 如果卷子里有别的提示，如 "请完成以下问题"、"完成下列各题" 等，也可以加进来
    ]
    
    # 拆成行，准备按行处理
    lines = raw_text.splitlines()

    chunks = []
    current_chunk_lines = []
    in_answer_section = False  # 如果检测到“参考答案”，就停止分块

    # 一个函数，用来判断某行是否包含任意一个关键字
    def is_boundary_line(line):
        for kw in BOUNDARY_KEYWORDS:
            if kw in line:  # 简单判断是否含有关键词
                return True
        return False

    # 我们需要记录“上一个边界”的位置
    # 初始时，先假装有个边界在开头
    last_boundary_index = 0  

    for i, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue  # 跳过空行

        # 如果出现“参考答案”就停止分块
        if "参考答案" in text or "答案" in text:
            in_answer_section = True
            # 把之前累积的作为一个chunk存起来，然后结束
            if last_boundary_index < i:
                chunk_lines = lines[last_boundary_index:i]
                cleaned = "\n".join(l.strip() for l in chunk_lines if l.strip())
                if cleaned:
                    chunks.append(cleaned)
            break

        # 如果本行是一个边界行
        if is_boundary_line(text):
            # 如果这不是第一次边界，那么说明上一个chunk到这里结束
            if i > last_boundary_index:
                chunk_lines = lines[last_boundary_index:i]
                cleaned = "\n".join(l.strip() for l in chunk_lines if l.strip())
                if cleaned:
                    chunks.append(cleaned)
            # 将 last_boundary_index 设为当前行，让新chunk从这里开始
            last_boundary_index = i
    
    # 循环结束后，如果还没到“参考答案”，把最后剩余的也做成一个chunk
    if not in_answer_section and last_boundary_index < len(lines):
        chunk_lines = lines[last_boundary_index:]
        cleaned = "\n".join(l.strip() for l in chunk_lines if l.strip())
        if cleaned:
            chunks.append(cleaned)

    return chunks

def chunk_exam_text(raw_text):
    lines = raw_text.splitlines()
    question_pattern = re.compile(r'^(\d{1,2})[\.．、\s]+')
    
    chunks = []
    current_lines = []
    current_qnum = None
    in_answer_section = False

    for line in lines:
        text = line.strip()
        if not text:
            # 跳过空行
            continue

        # 如果出现 “参考答案” 或 “答案”，就视为答案区，停止分题
        if "参考答案" in text or "答案" in text:
            in_answer_section = True

        if in_answer_section:
            break

        match = question_pattern.match(text)
        if match:
            # 如果已经在记录上一个题目，就先存起来
            if current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []

            current_qnum = match.group(1)
            current_lines.append(line)
        else:
            if current_qnum is not None:
                current_lines.append(line)
            # 否则可能是卷首标题之类，无需处理

    # 处理最后一个题目
    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


##############################################
# 主程序汇总
##############################################

if __name__ == "__main__":
    # 0) 指定文档 & 输出图片目录
    docx_file = "docx/2021dongcheng.docx"
    image_dir = "extracted_images"

    # 1) 加载图像描述模型
    model, feature_extractor, tokenizer, device = load_image_captioning_model(
        model_name="/home/xtang/models/nlpconnect/vit-gpt2-image-captioning"
    )

    # 2) 提取文本和图片
    text_with_placeholders = extract_text_and_images_from_docx(docx_file, image_dir)
    print("===== 从 docx 中提取到的文本 =====")
    print(text_with_placeholders)

    # 3) 遇到图片占位符，调用模型生成描述并替换
    text_with_captions = replace_image_placeholders_with_captions(
        text_with_placeholders,
        image_dir,
        model, feature_extractor, tokenizer, device
    )

    print("\n\n===== 替换后（带图片描述）的文本 =====")
    print(text_with_captions)

    # 4) 将处理后的文本按题号分块
    question_chunks = chunk_exam_geography(text_with_captions)

    print("\n\n===== 分题结果 =====")
    for i, chunk in enumerate(question_chunks, start=1):
        print(f"\n--- 第{i}题 chunk ---")
        print(chunk)
