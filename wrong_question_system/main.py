import os
import json
import numpy as np
import tempfile
import io
import pandas as pd
import streamlit as st
from PIL import Image
from log_config import get_logger
from vdb_index import VectorDBIndex
from query_embedding import get_query_embedding_bge
from image_caption import load_chinese_image_captioning_model, chinese_image_caption_inference
from docx_generator import generate_search_docx
from preprocess_files import process_single_file
from classify_wrong_questions import process_course_wrong_questions

# 加载配置文件
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# 通过配置文件设置日志路径
log_path = config.get("log_path")
logger = get_logger(__name__, log_file=log_path)

# 预先加载图像描述模型（在上传错题时也可能需要调用）
vl_model_name = config["vl_model_path"]
processor, model, device = load_chinese_image_captioning_model(vl_model_name)

def upload_exam_paper():
    """
    上传试卷功能：
      - 用户选择试卷所属科目；
      - 上传试卷文件后，将文件保存到配置项 "upload_file_path" 对应的科目目录下，
        如果目录不存在则创建（文件存在则覆盖）；
      - 保存后调用 preprocess_files.py 中的 process_single_file 进行预处理。
    """
    st.title("上传试卷")
    st.write("请选择科目并上传试卷文件。")
    course_upload = st.selectbox("选择试卷所属科目", ["语文", "数学", "英语", "历史", "地理", "政治"])
    exam_file = st.file_uploader("上传试卷文件", type=["docx", "pdf", "txt", "png", "jpg", "jpeg"])
    
    if st.button("上传试卷"):
        if exam_file is None:
            st.error("请先上传试卷文件。")
        else:
            # 从配置中获取上传路径
            upload_base_path = config.get("upload_file_path", "upload_files")
            subject_dir = os.path.join(upload_base_path, course_upload)
            if not os.path.exists(subject_dir):
                os.makedirs(subject_dir)
                logger.info("创建目录: %s", subject_dir)
            filename = exam_file.name if exam_file.name else "exam_file"
            file_path = os.path.join(subject_dir, filename)
            try:
                with open(file_path, "wb") as f:
                    f.write(exam_file.getvalue())
                st.success(f"试卷文件已保存到 {file_path}")
                logger.info("试卷 %s 保存到 %s 成功", filename, subject_dir)
                # 调用预处理函数
                process_single_file(config, course_upload, file_path)
                st.success(f"试卷已上传并预处理。科目：{course_upload}")
                logger.info("试卷 %s 上传并预处理成功，科目：%s", filename, course_upload)
            except Exception as e:
                st.error("试卷上传或预处理失败，请检查日志。")
                logger.error("处理试卷 %s 时出错: %s", filename, e, exc_info=True)

def query_wrong_question():
    """
    查询错题功能：
      - 用户选择错题所属科目，并输入查询文本或上传错题文件（文本或图片）；
      - 上传的错题文件保存到配置项 "wrong_question_path" 对应科目目录下，
        如果目录不存在则创建，存在则覆盖；
      - 如果上传的是文本文件，则读取内容作为查询文本；若上传的是图片，
        此处你可以选择调用图像描述模型生成描述文本（本示例仅示范文本读取）。
      - 合并文本输入和上传文件的内容后利用 BGE 模型生成查询向量，
        并使用 VectorDBIndex 检索相似错题，最终生成 DOCX 文档并提供下载。
    """
    st.title("查询错题")
    st.write("请选择科目，输入查询文本或上传错题文件，系统将保存上传的文件，并生成相关错题 DOCX 文档。")
    subject = st.selectbox("选择错题所属科目", ["语文", "数学", "英语", "历史", "地理", "政治"])
    query_text_input = st.text_area("查询文本（可选）", height=150)
    wrong_file = st.file_uploader("上传错题文件（可选，文本或图片）", type=["txt", "docx", "png", "jpg", "jpeg"])
    
    if st.button("上传并查询错题"):
        saved_query_text = ""
        if wrong_file is not None:
            wrong_base_path = config.get("wrong_question_path", "wrong_questions")
            subject_dir = os.path.join(wrong_base_path, subject)
            if not os.path.exists(subject_dir):
                os.makedirs(subject_dir)
                logger.info("创建目录: %s", subject_dir)
            wrong_filename = wrong_file.name if wrong_file.name else "wrong_question"
            file_path = os.path.join(subject_dir, wrong_filename)
            try:
                with open(file_path, "wb") as f:
                    f.write(wrong_file.getvalue())
                st.success(f"错题文件已保存到 {file_path}")
                logger.info("错题文件 %s 保存到 %s 成功", wrong_filename, subject_dir)
                # 仅处理 txt 文件内容作为查询文本；若为 docx 可扩展解析逻辑
                if wrong_filename.lower().endswith("txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        saved_query_text = f.read().strip()
                # 如果上传的是图片，可以调用图像描述模型（示例中调用）：
                elif wrong_filename.lower().endswith((".png", "jpg", "jpeg")):
                    # 使用图像描述生成查询文本
                    saved_query_text = chinese_image_caption_inference(file_path, processor, model, device)
                    logger.info("图片描述文本：%s", saved_query_text)
            except Exception as e:
                st.error("错题文件保存失败，请检查日志。")
                logger.error("保存错题文件 %s 时出错: %s", wrong_filename, e, exc_info=True)
        # 合并文本输入和上传文件内容
        final_query_text = ""
        if query_text_input and query_text_input.strip():
            final_query_text = query_text_input.strip()
        if saved_query_text:
            final_query_text = (final_query_text + " " + saved_query_text).strip() if final_query_text else saved_query_text
        if not final_query_text:
            st.error("请提供查询文本或上传错题文件！")
        else:
            logger.info("最终查询文本：%s", final_query_text)
            # 初始化向量数据库索引（加载对应科目下的 embedding 文件）
            EMBEDDINGS_DIR = os.path.join(config["embedding_output_dir"], subject)
            vdb = VectorDBIndex(engine="faiss")
            vdb.load_all_embeddings(EMBEDDINGS_DIR)
            vdb.build_index()
            # 使用 BGE 模型生成查询向量
            query_embedding = get_query_embedding_bge(final_query_text)
            if query_embedding is None:
                st.error("生成查询向量失败，请检查日志。")
            else:
                query_vector = np.array(query_embedding, dtype="float32").reshape(1, -1)
                results = vdb.search(query_vector, k=5)
                if not results:
                    st.info("未找到相似错题。")
                else:
                    # 生成 DOCX 文档
                    docx_stream = generate_search_docx(results, config, subject)
                    st.download_button(
                        label="下载相关错题文档",
                        data=docx_stream,
                        file_name="related_mistakes.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                    st.success("相关错题文档已生成，请点击上方按钮下载。")

def classify_wrong_questions():
    """
    整理错题功能：
      - 用户选择科目后，系统将读取配置中 wrong_question_path 下对应科目的所有错题（以 txt 文件为例），
      - 对每道错题调用 Deepseek（或其他大模型）询问“这道题考察的知识点是什么？”
      - 将返回的知识点保存为每道题的 meta 信息，并展示在页面上，同时提供下载 CSV 的功能。
    """
    st.title("整理错题")
    st.write("请选择科目，系统将对该课程下的所有错题进行归类。")
    subject = st.selectbox("选择科目", ["语文", "数学", "英语", "历史", "地理", "政治"])
    
    if st.button("开始归类"):
        st.info("正在归类，请稍候...")
        # 读取并归类当前科目下所有 txt 格式错题
        classification_results = process_course_wrong_questions(subject)
        logger.info("归类结果：%s", classification_results)
        if classification_results:
            df = pd.DataFrame(classification_results)
            st.write("归类结果：")
            st.dataframe(df)
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下载归类结果 (CSV)",
                data=csv_data,
                file_name=f"{subject}_wrong_questions_classification.csv",
                mime="text/csv"
            )
        else:
            st.error("没有找到可归类的错题文件。")

# 主程序侧边栏导航：选择功能
st.sidebar.title("导航")
app_mode = st.sidebar.selectbox("选择功能", ["上传试卷", "查询错题", "整理错题"])

if app_mode == "上传试卷":
    upload_exam_paper()
elif app_mode == "查询错题":
    query_wrong_question()
elif app_mode == "整理错题":
    classify_wrong_questions()
