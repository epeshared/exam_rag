# image_caption.py
import torch
from PIL import Image, UnidentifiedImageError
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from log_config import get_logger

import json
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

log_path = config.get("log_path")
logger = get_logger(__name__, log_file=log_path)

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
