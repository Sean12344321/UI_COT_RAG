import os
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd
from ollama_rag import generate_lawsheet
from utils import Tools
from collections import deque

from utils import Tools
from collections import deque

load_dotenv()

# Google Sheets API 配置
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
# 試算表 ID 和範圍
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
# 初始化 Google Sheets 客戶端
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('sheets', 'v4', credentials=creds)
sheet =  service.spreadsheets().values()
# 載入數據
df = pd.read_csv('dataset.csv')
df2 = pd.read_csv('dataset(no_law).csv')
# 加入標題和內容
template_outputs = [["gpt-4o-mini"]]  # 標題行
template_outputs.extend([[item] for item in df2["gpt-4o-mini-2024-07-18\n3000筆"].tolist()[:50]])  # 將輸出每一項獨立一行

inputs = [["模擬輸入內容"]]  # 標題行

# write_sheets(model_outputs, RANGE_WRITE_MODEL_OUTPUT)
def write_sheets_single(output, row_index, range_base):
    """將單行輸出即時寫入 Google Sheets"""
    print(f"DEBUG: row_index={row_index}, range_base={range_base}")  # 偵錯
    
    # 確保 range_base 格式正確
    sheet_name = range_base.split('!')[0]  # 提取工作表名稱
    column_range = range_base.split('!')[1]  # 提取範圍
    range_single = f"{sheet_name}!{column_range}{row_index + 1}"  
    
    print(f"Writing to range: {range_single}")  # 偵錯
    
    body = {'values': [[output]]}

    response = sheet.update(
        spreadsheetId=SPREADSHEET_ID,
        range=range_single,
        valueInputOption="RAW",
        body=body
    ).execute()
    print(f"Row {row_index + 1} updated: {output}")
# 逐行生成並即時寫入
tool1 = Tools("kenneth85/llama-3-taiwan:8b-instruct-dpo")
tool2 = Tools("gemma3:27b")
tools = [tool1, tool2]
cols = ["J", "K"]

for tool, col in zip(tools, cols):
    for i, item in enumerate(df["模擬輸入內容"].tolist()):
        output = deque(generate_lawsheet(item, tool), maxlen=1)[0]  # 生成單個輸出
        part, reference, summary, log, final_judge = output
        write_sheets_single(part.replace('<br>', '\n'), i + 1, f"Sheet1!{col}:{col}")  # 寫入對應欄