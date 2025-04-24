#!/usr/bin/env python3
# coding: utf-8

import os, sys, time, tempfile
import pandas as pd
import torch, numpy as np
from sentence_transformers import SentenceTransformer
from fpdf import FPDF
from ollama import chat
from generate_compensate import generate_compensate as raw_generate_compensate
from generate_truth import generate_fact_statement as raw_generate_fact_statement
from utils import Tools
import gradio as gr
import warnings

# 抑制 fpdf 字型警告
warnings.filterwarnings("ignore", message="cmap value too big/small")

# 專案初始化
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "KG_RAG_B"))
sys.path.append(os.path.join(os.path.dirname(__file__), "chunk_RAG"))

from KG_RAG_B.KG_Faiss_Query_3068 import query_simulation
from chunk_RAG.main import retrieval

# 載入資料
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model.to(device)

df = pd.read_csv('dataset.csv')
df2 = pd.read_csv('dataset(no_law).csv')
inputs = df["模擬輸入內容"].tolist()[:-2]
template_output = df2["gpt-4o-mini-2024-07-18\n3000筆"].tolist()

history = []
debug_logs = []

def get_similar_examples(input_text, top_k=3):
    input_embedding = embedding_model.encode(input_text, convert_to_tensor=True)
    all_embeddings = embedding_model.encode(inputs, convert_to_tensor=True)
    similarities = torch.nn.functional.cosine_similarity(input_embedding, all_embeddings)
    top_k_idx = similarities.argsort(descending=True)[:top_k]
    return [(inputs[i], template_output[i], float(similarities[i])) for i in top_k_idx]

def generate_fact_statement(*args, **kwargs):
    result = raw_generate_fact_statement(*args, **kwargs)
    if isinstance(result, tuple):
        return result
    return result, ""

def generate_compensate(*args, **kwargs):
    result = raw_generate_compensate(*args, **kwargs)
    if isinstance(result, tuple):
        return result
    return result, ""

def generate_lawsheet(input_data, rag_option="1"):
    debug = []
    start_time = time.time()
    debug.append(f"[時間] {time.strftime('%Y-%m-%d %H:%M:%S')} 啟動起訴狀生成")
    if rag_option == "1":
        references = query_simulation(input_data)
        debug.append("[RAG] 使用 KG_RAG 查詢成功")
    elif rag_option == "2":
        references = retrieval(input_data)
        debug.append("[RAG] 使用 chunk_RAG 查詢成功")
    else:
        return "請輸入正確的選項(1或2)", "", ""

    facts, laws, compensations = [], [], []
    data = Tools.split_user_input(input_data)

    for i, ref in enumerate(references):
        parsed = Tools.split_user_output(ref)
        if not parsed:
            debug.append(f"[清洗] 第{i+1}筆資料格式錯誤，跳過")
            continue
        facts.append(parsed["fact"])
        laws.append(parsed["law"])
        compensations.append(parsed["compensation"])
        debug.append(f"[清洗] 第{i+1}筆資料成功解析")

    part1, log1 = generate_fact_statement(data["case_facts"] + '\n' + data["injury_details"], facts)
    part2 = laws[0] if laws else ""
    part3, log2 = generate_compensate(input_data, compensations)

    result = part1 + '\n\n' + part2 + '\n\n' + part3
    history.append(result)

    examples = get_similar_examples(input_data)
    sim_str = "\n\n".join([f"範例{i+1}: 相似度{sim:.4f}\n輸入: {q[:40]}...\n輸出: {a[:40]}..." for i, (q, a, sim) in enumerate(examples)])
    debug.append("[查詢] 相似案例查詢完成")
    debug.append(f"[完成] 花費時間：{time.time() - start_time:.2f} 秒")

    debug_logs.append("\n".join(debug + ["\n========= 推理紀錄 ============", log1, log2]))

    return result, sim_str, "\n".join(debug + ["\n========= 推理紀錄 ============", log1, log2])

def export_pdf(content: str):
    try:
        font_path = "/usr/share/fonts/truetype/noto/NotoSansCJKtc-Regular.ttf"
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("NotoSans", fname=font_path, uni=True)
        pdf.set_font("NotoSans", size=12)
        for line in content.split("\n"):
            pdf.multi_cell(w=0, h=10, txt=line, align="L")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_file.name)
        return temp_file.name
    except Exception as e:
        print(f"⚠️ PDF 生成失敗：{e}")
        return None

def update_history_dropdown():
    return gr.update(choices=[f"記錄 {i+1}" for i in range(len(history))])

def view_history(selected):
    if not selected:
        return "請選擇歷史紀錄"
    idx = int(selected.split(" ")[1]) - 1
    return history[idx] if idx < len(history) else "紀錄不存在"

def view_debug_logs():
    return "\n\n".join(debug_logs[-3:])

with gr.Blocks() as demo:
    gr.Markdown("## 起訴狀自動生成器（含推理過程）")
    with gr.Row():
        user_input = gr.Textbox(label="請輸入案件描述")
        rag_selector = gr.Dropdown(choices=["1", "2"], label="選擇 RAG 資料庫 (1=KG_RAG, 2=chunk_RAG)", value="1")
    generate_btn = gr.Button("生成起訴狀")
    result_output = gr.Textbox(label="生成內容")
    similar_output = gr.Textbox(label="相似案例分析")
    debug_output = gr.Textbox(label="推理紀錄 / 系統紀錄")
    pdf_btn = gr.Button("下載 PDF")
    pdf_file = gr.File(label="PDF 檔案")
    history_dropdown = gr.Dropdown(choices=[], label="查看歷史記錄")
    view_btn = gr.Button("載入歷史紀錄")
    history_text = gr.Textbox(label="歷史紀錄內容")
    refresh_debug_btn = gr.Button("檢視最近推理紀錄")

    generate_btn.click(generate_lawsheet, inputs=[user_input, rag_selector], outputs=[result_output, similar_output, debug_output])
    generate_btn.click(update_history_dropdown, outputs=history_dropdown)
    pdf_btn.click(export_pdf, inputs=result_output, outputs=pdf_file)
    view_btn.click(view_history, inputs=history_dropdown, outputs=history_text)
    refresh_debug_btn.click(view_debug_logs, outputs=debug_output)

demo.launch()
