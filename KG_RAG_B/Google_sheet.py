from KG_Faiss_Query_3068 import query_faiss
from define_case_type import get_case_type
from Neo4j_Query import get_siminput_case,get_simoutput_case
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import os
from dotenv import load_dotenv

load_dotenv()

# Google Sheets API 配置
SERVICE_ACCOUNT_FILE = os.getenv("PATH_TO_GOOGLE_JSON")
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# 试算表 ID 和范围
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID_B")
RANGE_READ = 'Sheet1!A:A'  # 读取 A 栏
RANGE_WRITE_START = 'Sheet1!B1'  # 从 B1 开始写入

# 初始化 Google Sheets 客户端
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()

# 读取试算表数据并逐条生成结果
def read_and_write_sheets():
    # 读取 A 栏数据
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_READ).execute()
    values = result.get("values", [])

    if not values:
        print("试算表中没有可用的数据")
        return

    for i, row in enumerate(values, start=1):  # start=1 表示试算表行数从 1 开始
        user_input = row[0] if row else ""
        if not user_input.strip():
            continue  # 跳过空行

        print(f"正在处理第 {i} 行数据...")
        try:
            # 获取最相近的案件
            case_type=get_case_type(user_input)
            closest_cases = query_faiss(user_input,case_type)
            case_ids = [case["id"] for case in closest_cases]
            c = [str(x) for x in case_ids]
            case_ids_str = ",".join(c)
            print(case_ids_str)
            # 获取对应的模拟输入
            sim_inputs = [get_siminput_case(int(case_id)) for case_id in case_ids]
            sim_outputs = [get_simoutput_case(int(case_id)) for case_id in case_ids]
            # 准备写入的数据
            write_values = [[case_ids_str, sim_inputs[0], sim_inputs[1], sim_inputs[2], sim_outputs[0], sim_outputs[1], sim_outputs[2]]]
        except Exception as e:
            print(f"第 {i} 行生成失败: {e}")
            write_values = [["生成失败", "", "", "" , "", "", ""]]

        # 写入到 B、C、D、E 栏
        write_range = f"Sheet1!B{i}:H{i}"
        sheet.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=write_range,
            valueInputOption="RAW",
            body={"values": write_values}
        ).execute()
        print(f"第 {i} 行结果已写入试算表。")

# 执行程序
if __name__ == "__main__":
    read_and_write_sheets()
