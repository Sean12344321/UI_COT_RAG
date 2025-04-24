import pandas as pd
import os, sys, time
from ollama import chat, ChatResponse
from generate_compensate import generate_compensate
from generate_truth import generate_fact_statement, generate_simple_fact_statement
from utils import Tools
from KG_RAG_B.KG_Faiss_Query_3068 import query_simulation
from chunk_RAG.main import retrieval
os.chdir(os.path.dirname(__file__))
# 將 KG_RAG 目錄添加到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "KG_RAG_B"))
sys.path.append(os.path.join(os.path.dirname(__file__), "chunk_RAG"))
df = pd.read_csv('dataset.csv')
df2 = pd.read_csv('dataset(no_law).csv')
inputs = df["模擬輸入內容"].tolist()[:-2]
template_output = df2["gpt-4o-mini-2024-07-18\n3000筆"].tolist()

def generate_lawsheet(input_data):
    """處理單個生成請求並輸出結果"""
    start_time = time.time()
    userinput = input("請選擇使用的RAG資料庫(1: KG_RAG, 2: chunk_RAG): ")
    if userinput == "1":
        # 使用 KG_RAG
        references = query_simulation(input_data)
    elif userinput == "2":
        # 使用 chunk_RAG
        references = retrieval(input_data)
    else:
        print("請輸入正確的選項(1或2)")
        return None
    facts = []
    laws = []
    compensations = []
    data = Tools.split_user_input(input_data)
    for i, reference in enumerate(references):
        splited_reference = Tools.split_user_output(reference)
        if splited_reference == False:
            print(f"第{i}格式錯誤，不參考這筆資料")
            continue
        facts.append(splited_reference["fact"])
        laws.append(splited_reference["law"])
        compensations.append(splited_reference["compensation"])
    first_part = generate_fact_statement(data["case_facts"] + '\n' + data["injury_details"], facts)
    second_part = laws[0]
    third_part = generate_compensate(input_data, compensations)
    # print(compensations)
    print(first_part + '\n\n' + second_part + '\n\n' + third_part)
    print("=" * 50)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n執行時間: {hours}h {minutes}m {seconds}s")
    return first_part + '\n\n' + second_part + '\n\n' + third_part 
    

tmp_prompt = """一、事故發生緣由:
被告賴俊諺於民國110年12月4日上午10時27分許，無照駕駛被告賴惠敏所有車牌號碼000-0000號自用小貨車，沿桃園市新屋區中山東路1段往新屋方向行駛，行經同市區○○○○0段00號前時，本應注意車前狀況，並隨時採取必要之安全措施，而依當時天候晴、日間自然光線、柏油路面乾燥、無缺陷、無障礙物且視距良好，並無不能注意之情事，竟疏未注意前方車輛停等紅燈，不慎自後追撞前方由原告羅靖崴所駕駛並搭載原告邱品妍之車牌號碼000-0000號自用小客車(下稱系爭車輛)。
另查被告賴惠敏將其所有之車牌號碼000-0000號自用小貨車供未領有駕駛執照之被告賴俊諺駕駛，被告賴俊諺並於上開時地駕駛車牌號碼000-0000號自用小貨車，未注意車前狀況，自後追撞系爭車輛，而使原告2人受有傷害，因此被告兩人應負共同侵權行為之連帶損害賠償責任。

二、原告受傷情形:
原告羅靖崴因本件事故而受有頭部外傷併輕微腦震盪之傷害。
原告邱品妍則因本件車禍受有頭暈及頸部扭傷等傷害。

三、請求賠償的事實根據:
按民法第184條第1項前段、第185條第1項、第191條之2本文、第193條第1項、第195條第1項前段，請求下列損害
查原告羅靖崴因系爭車禍受有前揭傷害而前往聯新醫院就診，有聯新醫院診斷證明書可作為證據，其因而支出醫療費用2,443元、交通費1,235元。
原告羅靖崴因本件事故受傷，需在家休養16日而無法工作，又原告羅靖崴每月工資應為37,778元，又依聯新醫院診斷證明書分別於110年12月4日及111年1月10日建議原告羅靖崴應休養3日及兩週，是原告羅靖崴應有17日不能工作，但原告羅靖崴僅請求16日工資損失，因此請求不能工作之損失20,148元
原告羅靖崴因本件車禍而受有頭部外傷併輕微腦震盪之傷害，影響日常生活甚鉅，於精神上可能承受之無形痛苦，故請求被告賠償10,000元精神慰撫金。

原告邱品妍因系爭車禍受有前揭傷害同樣前往聯新醫院醫治，有聯新醫院診斷證明書作為證據，其因而支出醫療費用57,550元、交通費22,195元。
另外原告邱品妍因本件事故受傷，需在家休養1月又28日而無法工作，又原告邱品妍每月工資為34,535元，又依聯新醫院診斷證明書分別於110年12月4日、111年12月6日、 110年12月17日、110年12月24日、111年1月10日持續建議休養1週至1個月，總計1個月又28日，其不能工作之損失應為66,768元。
另查系爭車輛，因被告之過失行為，受有交易上價值貶損33,000元及支出鑑定費3,000元，因此原告邱品妍向被告請求賠償之本件車輛交易價值減損及鑑定費共計36,000元。
原告邱品妍因本件車禍受有頭暈及頸部扭傷等傷害，影響其工作、生活之行動，於精神上造成無形痛苦，故請求被告連帶賠償60,000元精神慰撫金。
"""

print(generate_lawsheet(tmp_prompt))