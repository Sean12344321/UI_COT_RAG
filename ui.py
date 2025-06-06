#!/usr/bin/env python3
# coding: utf-8

import os, sys, time, tempfile, re
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

warnings.filterwarnings("ignore", message="cmap value too big/small")

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "KG_RAG_B"))
sys.path.append(os.path.join(os.path.dirname(__file__), "chunk_RAG"))

from KG_RAG_B.KG_Faiss_Query_3068 import query_simulation
from chunk_RAG.ts_main import retrieval

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model.to(device)

# 預設資料集
original_df = pd.read_csv('dataset.csv')
df2 = pd.read_csv('dataset(no_law).csv')
inputs = original_df["模擬輸入內容"].tolist()[:-2]
template_output = df2["gpt-4o-mini-2024-07-18\n3000筆"].tolist()

history = []
debug_logs = []

excel_df = None  # 儲存上傳 Excel 內容的變數（全域）

def get_similar_examples(input_text, top_k=3):
    input_embedding = embedding_model.encode(input_text, convert_to_tensor=True)
    all_embeddings = embedding_model.encode(inputs, convert_to_tensor=True)
    similarities = torch.nn.functional.cosine_similarity(input_embedding, all_embeddings)
    top_k_idx = similarities.argsort(descending=True)[:top_k]
    return [(inputs[i], template_output[i], float(similarities[i])) for i in top_k_idx]

def generate_fact_statement(*args, **kwargs):
    result = yield from raw_generate_fact_statement(*args, **kwargs)
    return result if isinstance(result, tuple) else (result, "")

def generate_compensate(*args, **kwargs):
    result = yield from raw_generate_compensate(*args, **kwargs)
    return result if isinstance(result, tuple) else (result, "")

def generate_lawsheet(input_data, rag_option="1", top_k=3, model_choice="kenneth85/llama-3-taiwan:8b-instruct-dpo"):
    tools = Tools(model_choice)
    debug, facts, laws_id, compensations = [], [], [], []
    start_time = time.time()
    debug.append(f"[時間] {time.strftime('%Y-%m-%d %H:%M:%S')} 啟動起訴狀生成")

    references = query_simulation(input_data, top_k) if rag_option == "1" else retrieval(input_data,top_k)
    debug.append("[RAG] 使用 {} 查詢成功".format("KG_RAG" if rag_option == "1" else "chunk_RAG"))

    data = tools.split_user_input(input_data)

    for i, ref in enumerate(references):
        parsed = tools.split_user_output(ref["case_text"])
        if not parsed:
            debug.append(f"[清洗] 第{i+1}筆資料格式錯誤，跳過")
            continue
        facts.append(parsed["fact"])
        laws_id.append(ref["case_id"])
        compensations.append(parsed["compensation"])
        debug.append(f"[清洗] 第{i+1}筆資料成功解析")

    main_output = ""
    reference_case_text=[f"第{i+1}筆參考範例:\n{reference["case_text"]}"for i, reference in enumerate(references)] 
    reference_ouptut = references="\n\n".join(reference_case_text)
    debug_output = ""
    log1 = ""
    log2 = ""
    yield main_output, reference_ouptut, debug_output

    for part1, ref1, audit1 in generate_fact_statement(data["case_facts"] + '\n' + data["injury_details"], facts, tools):
        main_output += part1 + ('\n' if part1 else '')
        reference_ouptut += ref1 + ('\n' if ref1 else '')
        debug_output += audit1 + ('\n' if audit1 else '')
        log1 += audit1
        yield main_output, reference_ouptut, debug_output

    main_output += '\n' + tools.generate_laws(laws_id, 2) + "\n\n"
    yield main_output, reference_ouptut, debug_output

    for part3, ref3, audit3 in generate_compensate(input_data, compensations, tools):
        main_output += part3 + ('\n\n' if part3 else '')
        reference_ouptut += ref3 + ('\n' if ref3 else '')
        debug_output += audit3 + ('\n' if audit3 else '')
        if "超過 7 次" in audit3:
            main_output = re.sub(r'（一）.*', '', main_output, flags=re.DOTALL)
        log2 += audit3
        yield main_output, reference_ouptut, debug_output

    history.append(main_output)

    examples = get_similar_examples(input_data, top_k=top_k)
    sim_str = "\n\n".join([
        f"--- 範例 {i+1} ---\n相似度: {sim:.4f}\n輸入：{q.strip().replace('\\n', chr(10))}\n\n輸出：{a.strip().replace('\\n', chr(10))}"
        for i, (q, a, sim) in enumerate(examples)
    ])

    debug.append("[查詢] 相似案例查詢完成")
    debug.append(f"[完成] 花費時間：{time.time() - start_time:.2f} 秒")
    debug_logs.append("\n".join(debug + ["\n========= 推理紀錄 ============", log1, log2]))

    return main_output, sim_str, debug_logs[-1]

def export_pdf(content: str):
    try:
        font_path = "NotoSansCJKtc-Regular.ttf"
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
    idx = int(selected.split(" ")[1]) - 1 if selected else -1
    return history[idx] if 0 <= idx < len(history) else "紀錄不存在"

def view_debug_logs():
    return "\n\n".join(debug_logs[-3:])

def handle_excel_upload(file):
    global excel_df
    excel_df = pd.read_excel(file.name)
    return gr.update(choices=list(excel_df.columns))

def populate_input(col_name, row_idx):
    if excel_df is not None and col_name in excel_df.columns:
        try:
            return str(excel_df.at[int(row_idx), col_name])
        except:
            return "資料行列無效"
    return "欄位尚未上傳或不存在"

with gr.Blocks(css="""
.highlight-output textarea {
    font-size: 18px;
    font-weight: bold;
    color: #222;
    background-color: #fffbe6;
    border: 2px solid #f90;
    border-radius: 8px;
    padding: 12px;
}
""") as demo:
    gr.Markdown("## 起訴狀自動生成器（含推理過程 + Excel 上傳）")
    with gr.Row():
        user_input = gr.Textbox(label="請輸入案件描述")
        rag_selector = gr.Dropdown(choices=["段落切割", "語意切割"], label="選擇 RAG chunk", value="段落切割")
        model_selector = gr.Dropdown(choices=["kenneth85/llama-3-taiwan:8b-instruct-dpo", "gemma3:27b"], label="選擇 LLM 模型", value="kenneth85/llama-3-taiwan:8b-instruct-dpo")
        top_k_slider = gr.Slider(label="相似案例數量", minimum=1, maximum=10, step=1, value=3)

    with gr.Row():
        excel_upload = gr.File(label="上傳 Excel 檔案 (.xlsx)")
        col_dropdown = gr.Dropdown(label="選擇欄位", choices=[])
        row_slider = gr.Slider(label="選擇列 index", minimum=0, maximum=5000, step=1, value=0)
        inject_btn = gr.Button("匯入欄位內容")

    inject_btn.click(populate_input, inputs=[col_dropdown, row_slider], outputs=user_input)
    excel_upload.change(handle_excel_upload, inputs=excel_upload, outputs=col_dropdown)

    generate_btn = gr.Button("生成起訴狀")
    result_output = gr.Textbox(label="🔶 生成內容 (主輸出)", lines=20, max_lines=40, interactive=True, show_copy_button=True, elem_classes=["highlight-output"])
    similar_output = gr.Textbox(label="相似案例")
    debug_output = gr.Textbox(label="COT推理紀錄")
    pdf_btn = gr.Button("下載 PDF")
    pdf_file = gr.File(label="PDF 檔案")
    history_dropdown = gr.Dropdown(choices=[], label="查看歷史記錄")
    view_btn = gr.Button("載入歷史紀錄")
    history_text = gr.Textbox(label="歷史紀錄內容")

    generate_btn.click(generate_lawsheet, inputs=[user_input, rag_selector, top_k_slider, model_selector], outputs=[result_output, similar_output, debug_output])
    generate_btn.click(update_history_dropdown, outputs=history_dropdown)
    pdf_btn.click(export_pdf, inputs=result_output, outputs=pdf_file)
    view_btn.click(view_history, inputs=history_dropdown, outputs=history_text)

Tools("kenneth85/llama-3-taiwan:8b-instruct-dpo")
demo.queue().launch(share=True)   