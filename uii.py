#!/usr/bin/env python3
# coding: utf-8

import os, sys, time, tempfile, re
import pandas as pd
import torch, numpy as np
from sentence_transformers import SentenceTransformer
from fpdf import FPDF
from ollama import chat
from generate_compensate import generate_compensate as raw_generate_compensate, generate_simple_compensate
from generate_truth import generate_fact_statement as raw_generate_fact_statement, generate_simple_fact_statement
from generate_laws import check_and_generate_laws
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

# original_df = pd.read_csv('dataset.csv')
# df2 = pd.read_csv('dataset(no_law).csv')
# inputs = original_df["模擬輸入內容"].tolist()[:-2]
# template_output = df2["gpt-4o-mini-2024-07-18\n3000筆"].tolist()

df3 = pd.read_excel("data_2995.xlsx")
reference_inputs = df3["事實陳述"]
reference_outputs = df3["起訴書"]

history = []
debug_logs = []

excel_df = None

# def get_similar_examples(input_text, rag_option, top_k=3):
#     # input_embedding = embedding_model.encode(input_text, convert_to_tensor=True)
#     # all_embeddings = embedding_model.encode(inputs, convert_to_tensor=True)
#     # similarities = torch.nn.functional.cosine_similarity(input_embedding, all_embeddings)
#     # top_k_idx = similarities.argsort(descending=True)[:top_k]
#     # return [(inputs[i], template_output[i], float(similarities[i])) for i in top_k_idx]
#     if rag_option == "段落切割":
#         rag_option = "1"
#         top_k = top_k + 1
#     else: 
#         rag_option = "2"
#     input_embedding = embedding_model.encode(input_text, convert_to_tensor=True)
#     references = query_simulation(input_text, top_k) if rag_option == "1" else retrieval(input_text,top_k)
#     top_k_idx = [reference["case_id"] for reference in references]
#     if rag_option == "1":
#         top_k_idx = top_k_idx[1:]
#     results = []
#     for case_id in top_k_idx:
#         reference_input_text = reference_inputs[case_id]
#         reference_output_text = reference_outputs[case_id]
#         case_embedding = embedding_model.encode(reference_input_text, convert_to_tensor=True)
#         similarity = float(torch.nn.functional.cosine_similarity(input_embedding, case_embedding, dim=0))
#         results.append((reference_input_text, reference_output_text, similarity))
#     results = sorted(results, key=lambda x: x[2], reverse=True)
#     return results
def generate_fact_statement(*args, **kwargs):
    result = yield from raw_generate_fact_statement(*args, **kwargs)
    return result if isinstance(result, tuple) else (result, "")

def generate_compensate(*args, **kwargs):
    result = yield from raw_generate_compensate(*args, **kwargs)
    return result if isinstance(result, tuple) else (result, "")


def generate_lawsheet(input_data, rag_option="1", top_k=3, model_choice="kenneth85/llama-3-taiwan:8b-instruct-dpo"):
    reference_output = ""
    fast_output = ""
    first_part_output = ""
    first_part_cot = ""
    second_part_output = ""
    second_part_cot = ""
    third_part_output = ""
    third_part_cot = ""
    final_response = ""
    def current_state():
        return [
            reference_output,
            fast_output,
            first_part_output,
            first_part_cot,
            second_part_output,
            second_part_cot,
            third_part_output,
            third_part_cot,
            final_response
        ]
    tools = Tools(model_choice)
    if rag_option == "段落切割":
        rag_option = "1"
    else: 
        rag_option = "2"
    debug, facts, laws, compensations = [], [], [], []
    start_time = time.time()
    debug.append(f"[時間] {time.strftime('%Y-%m-%d %H:%M:%S')} 啟動起訴狀生成")
    references = query_simulation(input_data, top_k) if rag_option == "1" else retrieval(input_data,top_k)
    debug.append("[RAG] 使用 {} 查詢成功".format("KG_RAG" if rag_option == "1" else "chunk_RAG"))
    reference_case_text=[f"<div style='color:blue'>第{i+1}筆參考範例:</div>{reference['case_text'].replace('\n','<br>')}<br>" for i, reference in enumerate(references)]
    reference_output = "".join(reference_case_text)
    yield current_state()
    data = tools.split_user_input(input_data)
    for i, ref in enumerate(references):
        parsed = tools.split_user_output(ref["case_text"])
        if not parsed:
            debug.append(f"[清洗] 第{i+1}筆資料格式錯誤，跳過")
            continue
        facts.append(parsed["fact"])
        laws.append(parsed["law"])
        compensations.append(parsed["compensation"])
        debug.append(f"[清洗] 第{i+1}筆資料成功解析")
    log1 = ""
    log2 = ""
    extracted_laws = tools.extract_cases_to_laws(laws)
    ##generate simple lawsheet
    for part_1 in generate_simple_fact_statement(data["case_facts"] + '\n' + data["injury_details"], facts, tools):
        fast_output += part_1
        yield current_state()
    fast_output += '<br><br>' + tools.generate_laws(extracted_laws, 2).replace('\n', '<br>') + "<br><br>"
    yield current_state()
    for part_3 in generate_simple_compensate(input_data, data["injury_details"], data["compensation_request"], compensations, tools):
        fast_output += part_3 + "<br><br>" if part_3 else ""
        yield current_state()
    ##generate full lawsheet
    # split_fast_output = tools.split_user_output(fast_output)
    cnt = 0
    for part1, ref1, summary1, audit1, final_judge1 in generate_fact_statement(data["case_facts"] + '\n' + data["injury_details"], facts, tools):
        if summary1: 
            cnt += 1
            first_part_cot += f"<div style='color:#e327d3'>第{cnt}次事實陳述推論:</div>"
            first_part_cot += summary1
        
        first_part_output += part1
        first_part_cot += audit1
        first_part_cot += final_judge1 + ('<br><br>' if final_judge1 else '')
        log1 += summary1 + audit1 + final_judge1
        yield current_state()
    # second_part_output += '<br><br>' + tools.generate_laws(laws_id, 2).replace('\n', '<br>') + "<br><br>"
    for part2, ref2, summary2, audit2, final_judge2 in check_and_generate_laws(input_data, tools, top_k, extracted_laws):
        second_part_cot += audit2 + "<br>" if audit2 else ""
        second_part_output += part2
        yield current_state()

    # second_part_output += tools.generate_laws(laws_id, 2).replace('\n', '<br>')
    # yield current_state()
    cnt = 0
    item = 1
    for part3, ref3, summary3, audit3, final_judge3 in generate_compensate(input_data, compensations, tools):
        if summary3:   
            cnt += 1 
            third_part_cot += f"<div style='color:#e327d3'>第{item}項賠償金額推論(第{cnt}次):</div>"
            third_part_cot += summary3
        third_part_output += part3 + "<br><br>" if part3 else ""
        reference_output += ref3
        third_part_cot += audit3
        third_part_cot += final_judge3 + ('<br><br>' if final_judge3 else '')
        if "Accept" in final_judge3 or "無法通過檢查，直接繼續生成" in final_judge3:
            item += 1
            cnt = 0
        log2 += summary3 + audit3 + final_judge3
        yield current_state()
    final_response = first_part_output +"<br><br>" +second_part_output +"<br><br>" + third_part_output
    history.append(final_response.replace("<br>", "\n"))
    yield current_state()
    # examples = get_similar_examples(input_data, rag_option, top_k=top_k)
    # sim_str = "<br><br>".join([
    #     f"<div style='background:#eef;padding:10px'><b>範例 {i+1}</b><br>相似度: {sim:.4f}<br><b>輸入：</b>{q.strip().replace('\\n', '<br>')}<br><b>輸出：</b>{a.strip().replace('\\n', '<br>')}</div>"
    #     for i, (q, a, sim) in enumerate(examples)
    # ])

    debug.append("[查詢] 相似案例查詢完成")
    debug.append(f"[完成] 花費時間：{time.time() - start_time:.2f} 秒")
    debug_logs.append("<br>".join(debug + ["<br>========= 推理紀錄 ============<br>", log1, log2]))

    # return main_output, examples, debug_logs[-1]

def export_pdf(content: str):
    try:
        font_path = "./fonts/NotoSansCJKtc-Regular.ttf"
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("NotoSans", fname=font_path, uni=True)
        pdf.set_font("NotoSans", size=12)
        content = content.replace("<br", "\n")
        for line in content.split("\n"):
            clean_line = line.lstrip("> ").strip()
            pdf.multi_cell(w=0, h=10, txt=clean_line, align="L")
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
    return "<br><br>".join(debug_logs[-3:])

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
def handle_txt_upload(file):
    if file is None:
        return "請先上傳 TXT 檔案"
    try:
        with open(file.name, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"無法讀取 TXT：{e}"
    
# 顯示目前的相似案例（1條）
def display_similar_example(example_list, idx):
    if not example_list:
        return "無相似案例"
    idx = max(0, min(idx, len(example_list) - 1))
    q, a, sim = example_list[idx]
    return f"""
    <div style='background:#eef;padding:10px;color:#000'>
        <b style='color:#2980b9'>範例 {idx+1}</b><br>
        相似度: {sim:.4f}<br>
        <b>輸入：</b>{q.strip().replace('\\n', '<br>')}<br>
        <b>輸出：</b>{a.strip().replace('\\n', '<br>')}
    </div>
    """
# 上一頁 / 下一頁按鈕 callback
# def update_example(example_list, idx, direction):
#     new_idx = max(0, min(idx + direction, len(example_list)-1))
#     return display_similar_example(example_list, new_idx), new_idx
# def store_similar_examples(input_text, rag_option, top_k):
#         examples = get_similar_examples(input_text, rag_option, top_k)
#         return examples, 0, display_similar_example(examples, 0)
# with gr.Blocks(title="法律文書生成系統") as demo:
css = """
/* 主輸出區塊 */
.highlight-output {
    font-size: 18px;
    font-weight: bold;
    color: #222;
    background-color: #fffbe6;
    border: 2px solid #f90;
    border-radius: 8px;
    padding: 16px;
    overflow: auto;
    max-height: 500px;
}

/* Ensure paired blocks align vertically and have consistent heights */
.output-pair {
    display: flex;
    align-items: stretch; /* Stretch to match the tallest block in the pair */
    margin-bottom: 16px; /* Space between pairs */
}

/* Ensure similar_cases_block has a consistent appearance and stretches */
.similar_cases_block {
    font-size: 15px;
    min-height: 150px;
    max-height: 400px;
    background-color: #272729;
    border: 3px solid #302f36;
    border-radius: 8px;
    overflow: auto;
    padding: 12px;
    scrollbar-width: thin;
    color: #fff;
    flex: 1; /* Allow the block to stretch to match its pair */
    display: flex;
    flex-direction: column;
}

/* 輸入框強調 */
textarea, .input-component {
    border: 2px solid #e67e22 !important;
    box-shadow: 0 0 8px rgba(230,126,34,0.5);
    font-size: 16px;
}

/* 分級標題樣式 */
.title {
    background: #e67e22;
    color: black !important;
    padding: 6px 12px;
    font-size: 25px;
    font-weight: bold;
    border-radius: 6px;
    margin-top: 8px;
    margin-bottom: 8px;
}
.section-title {
    background: #e67e22;
    color: black !important;
    padding: 6px 12px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 6px;
    margin-top: 8px;
    margin-bottom: 8px;
}
.subsection-title {
    background: #9b59b6;
    color: white;
    padding: 4px 10px;
    font-size: 14px;
    border-radius: 4px;
    margin-top: 6px;
    margin-left: 12px;
    margin-bottom: 8px;
}

#generate-btn {
    background-color: #e74c3c !important;  
    color: black !important;
    font-weight: bold;
    font-size: 25px !important;  
    padding: 20px 32px !important;
    border: none !important;
    border-radius: 10px !important;
    transition: background 0.3s ease;
}

#generate-btn:hover {
    background-color: #c0392b !important;  /* 深紅 */
}

#inject-btn {
    background-color: #f39c12;
    color: white;
    font-weight: bold;
    font-size: 22px;
    padding: 14px 28px;
    border: none;
    border-radius: 10px;
    transition: background 0.3s ease;
}
#inject-btn:hover {
    background-color: #e67e22;
}

"""
   
with gr.Blocks(title="法律文書生程系統", css=css) as demo:
    gr.Markdown("## 起訴狀自動生成器")
    gr.HTML("<div class='title'>📥 請輸入案件描述</div>")
    with gr.Row():
        user_input = gr.Textbox(lines=5, placeholder="請輸入完整案件內容", label="", scale=3)
        with gr.Column(scale=1):
            rag_selector = gr.Dropdown(["段落切割", "語意切割"], label="RAG 模式", value="段落切割")
            model_selector = gr.Dropdown(["kenneth85/llama-3-taiwan:8b-instruct-dpo", "gemma3:27b"], label="模型選擇", value="kenneth85/llama-3-taiwan:8b-instruct-dpo")
            top_k_slider = gr.Slider(label="相似案例數量", minimum=1, maximum=10, value=3, step=1)
    with gr.Row():
        excel_upload = gr.File(label="上傳 Excel 檔案 (.xlsx)")
        txt_upload = gr.File(label="上傳 TXT 檔案 (.txt)", file_types=[".txt"])
        col_dropdown = gr.Dropdown(label="選擇欄位", choices=[])
        row_slider = gr.Slider(label="選擇列 index", minimum=0, maximum=5000, step=1, value=0)
        inject_btn = gr.Button("📤 匯入欄位內容", elem_id="inject-btn", variant="stop")

    excel_upload.change(handle_excel_upload, inputs=excel_upload, outputs=col_dropdown)
    inject_btn.click(populate_input, inputs=[col_dropdown, row_slider], outputs=user_input)
    generate_btn = gr.Button("🚀 生成起訴狀", elem_id="generate-btn", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.HTML("<p style='font-weight:bold;font-size: 18px;'>📚 相似案例</p>")
            similar_cases = gr.HTML(elem_classes=["similar_cases_block"])
        with gr.Column():
            gr.HTML("<p style='font-weight:bold;font-size: 18px;'>⚡ 快速生成內容 (不含COT)</p>")
            draft_output = gr.HTML(elem_classes=["similar_cases_block"])

    # Fact, Law, Compensate Pairs
    with gr.Row():
        # Fact Pair
        with gr.Row(elem_classes=["output-pair"]):
            with gr.Column(scale=1):
                gr.HTML("<div class='section-title'> ◉ 事實陳述生成</div>")
                fact_output = gr.HTML(elem_classes=["similar_cases_block"])
            with gr.Column(scale=1):
                gr.HTML("<div class='subsection-title'> ⬆ 事實 COT</div>")
                fact_cot = gr.HTML(elem_classes=["similar_cases_block"])
        
        # Law Pair
        with gr.Row(elem_classes=["output-pair"]):
            with gr.Column(scale=1):
                gr.HTML("<div class='section-title'> ◉ 法條引用生成</div>")
                law_output = gr.HTML(elem_classes=["similar_cases_block"])
            with gr.Column(scale=1):
                gr.HTML("<div class='subsection-title'> ⬆ 法條 COT</div>")
                law_cot = gr.HTML(elem_classes=["similar_cases_block"])
        
        # Compensate Pair
        with gr.Row(elem_classes=["output-pair"]):
            with gr.Column(scale=1):
                gr.HTML("<div class='section-title'> ◉ 賠償金額生成</div>")
                compensate_output = gr.HTML(elem_classes=["similar_cases_block"])
            with gr.Column(scale=1):
                gr.HTML("<div class='subsection-title'> ⬆ 賠償 COT</div>")
                compensate_cot = gr.HTML(elem_classes=["similar_cases_block"])

    gr.HTML("<h2 style='color:#d35400'>🔶 完整生成內容 (含COT)</h2>")
    final_output = gr.HTML(elem_classes=["highlight-output"])
    pdf_btn = gr.Button("📥 下載 PDF", elem_classes=["cta-button"])
    pdf_file = gr.File(label="PDF 檔案")
    history_dropdown = gr.Dropdown(choices=[], label="查看歷史記錄")
    view_btn = gr.Button("📜 載入歷史紀錄", elem_classes=["cta-button"])
    history_text = gr.Textbox(label="歷史紀錄內容")

    generate_btn.click(generate_lawsheet, 
        inputs=[user_input, rag_selector, top_k_slider, model_selector], 
        outputs=[similar_cases, draft_output, fact_output, fact_cot, law_output, law_cot, compensate_output, compensate_cot, final_output])
    generate_btn.click(update_history_dropdown, outputs=history_dropdown)
    pdf_btn.click(export_pdf, inputs=final_output, outputs=pdf_file)
    view_btn.click(view_history, inputs=history_dropdown, outputs=history_text)
    txt_upload.change(handle_txt_upload, inputs=txt_upload, outputs=user_input)

Tools("kenneth85/llama-3-taiwan:8b-instruct-dpo")
demo.queue().launch(share=True)