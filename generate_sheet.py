import os
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd
from rag_finetune.ollama_rag import generate_output
load_dotenv()

# Google Sheets API 配置
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
# 試算表 ID 和範圍
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
RANGE_WRITE_INPUT = 'Sheet1!A:A'  # 寫入輸入範圍
RANGE_WRITE_TEMPLATE_OUTPUT = 'Sheet1!B:B' # 寫入gpt-4o-mini 輸出範圍
RANGE_WRITE_MODEL_OUTPUT = 'Sheet1!E10:E' # 寫入模型llama3.1輸出範圍
# 初始化 Google Sheets 客戶端
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()

# 載入數據
df = pd.read_csv('dataset.csv')
df2 = pd.read_csv('dataset(no_law).csv')
# 加入標題和內容
template_outputs = [["gpt-4o-mini"]]  # 標題行
template_outputs.extend([[item] for item in df2["gpt-4o-mini-2024-07-18\n3000筆"].tolist()[:50]])  # 將輸出每一項獨立一行

inputs = [["模擬輸入內容"]]  # 標題行
inputs.extend([[item] for item in df["模擬輸入內容"].tolist()[:-1]])  # 將輸出每一項獨立一行
model_outputs = ([[generate_output(item)] for item in df["模擬輸入內容"].tolist()[8:15]])  # 將輸出每一項獨立一行

def write_sheets(inputs, range):
    body = {
        'values': inputs  
    }
    response = sheet.values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=range,
        valueInputOption="RAW",
        body=body
    ).execute()
    print(f"{response.get('updatedCells')} cells updated.")

write_sheets(model_outputs, RANGE_WRITE_MODEL_OUTPUT)