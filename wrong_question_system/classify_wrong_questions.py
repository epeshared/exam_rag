import os
import json
import requests
import base64
import streamlit as st
import pandas as pd
from log_config import get_logger
from image_caption import chinese_image_caption_inference, load_chinese_image_captioning_model

# 加载配置文件
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# 初始化 logger
log_path = config.get("log_path")
logger = get_logger(__name__, log_file=log_path)

# 预先加载图像描述模型（用于图片文件的处理）
vl_model_name = config["vl_model_path"]
processor, model, device = load_chinese_image_captioning_model(vl_model_name)

# 设定 Deepseek R1 671b API 的 URL（请根据实际情况修改）
DEEPSEEK_API_URL = config.get("deepseek_api_url", "http://localhost:11434/api/generate")

def classify_question(question_text, is_image=False):
    """
    调用 Deepseek R1 671b 模型，对错题进行归类，询问“这道题考察的知识点是什么？”。
    如果 is_image 为 True，则认为 question_text 为图片的二进制数据，
    将其转换为 Base64 并作为 "images" 参数传递；否则作为文本传递 prompt。
    
    :param question_text: 错题文本或图片二进制数据
    :param is_image: 布尔值，表示是否为图片
    :return: 分类结果（知识点），字符串类型
    """
    if is_image:
        # 将二进制数据转换为 Base64 字符串
        image_b64 = base64.b64encode(question_text).decode("utf-8")
        payload = {
            "model": "deepseek-r1:671b",
            "prompt": "这道题考察的知识点是什么？",
            "stream": False,
            "images": [image_b64]
        }
    else:
        payload = {
            "model": "deepseek-r1:671b",
            "prompt": question_text + "\n\n请回答：这道题考察的知识点是什么？",
            "format": "json",
            "stream": False
        }
    try:
        logger.info("调用 Deepseek 进行分类...")
        response = requests.post(DEEPSEEK_API_URL, json=payload)        
        response.raise_for_status()
        result = response.json()
        logger.info("Deepseek 响应：%s", result)
        classification = result.get("response")
        if classification is None:
            logger.error("Deepseek响应中未包含分类字段: %s", result)
            return "未知"
        logger.info("分类结果：%s", classification)
        return classification
    except Exception as e:
        logger.error("调用Deepseek进行分类失败: %s", e, exc_info=True)
        return "错误"

def process_course_wrong_questions(course):
    """
    处理某个课程下所有错题文件进行归类。
    读取 config["wrong_question_path"]/course 目录下所有 txt 和图片文件，
    对每个文件调用 classify_question 获取知识点归类信息，并建立 meta 信息。
    
    :param course: 课程名称
    :return: 错题归类结果列表，每个元素为字典 {文件名, 错题内容, 考察知识点}
    """
    wrong_base_path = config.get("wrong_question_path", "wrong_questions")
    course_dir = os.path.join(wrong_base_path, course)
    logger.info("处理科目 %s 的错题目录: %s", course, course_dir)
    if not os.path.exists(course_dir):
        st.error(f"科目 {course} 的错题目录不存在：{course_dir}")
        return []
    results = []
    for fname in os.listdir(course_dir):
        file_path = os.path.join(course_dir, fname)
        logger.info("处理文件: %s", file_path)
        try:
            if fname.lower().endswith("txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                classification = classify_question(content, is_image=False)
            elif fname.lower().endswith((".png", "jpg", "jpeg")):
                try:
                    with open(file_path, "rb") as f:
                        binary_content = f.read()
                    logger.info("读取图片二进制内容成功: %s", fname)
                    classification = classify_question(binary_content, is_image=True)
                    # 保存原始二进制内容作为错题内容
                    content = binary_content
                except Exception as e:
                    logger.error("读取图片 %s 二进制内容失败: %s", fname, e, exc_info=True)
                    content = ""
                    classification = "错误"
            else:
                content = ""
                classification = "未知"
            results.append({
                "文件名": fname,
                # "错题内容": content,
                "考察知识点": classification
            })
        except Exception as e:
            logger.error("处理文件 %s 时出错: %s", fname, e, exc_info=True)
    return results
