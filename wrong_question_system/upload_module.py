import os
import shutil
from log_config import get_logger

import json
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

log_path = config.get("log_path")
logger = get_logger(__name__, log_file=log_path)

def save_uploaded_mistake(file_obj, course, filename, base_dir="wrong_questions"):
    """
    保存上传的错题文件到对应课程的目录中
    :param file_obj: 上传的文件对象
    :param course: 课程名称（如“地理”）
    :param filename: 文件名称
    :param base_dir: 基础存储目录
    """
    course_dir = os.path.join(base_dir, course)
    if not os.path.exists(course_dir):
        os.makedirs(course_dir)
    file_path = os.path.join(course_dir, filename)
    try:
        # 如果 file_obj 是一个临时文件路径，可以直接移动或复制
        with open(file_path, "wb") as f:
            f.write(file_obj.read())
        logger.info("文件 %s 保存到 %s 成功", filename, course_dir)
        return file_path
    except Exception as e:
        logger.error("保存文件 %s 时出错: %s", filename, e, exc_info=True)
        return None
