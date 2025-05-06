import pandas as pd
import os, sys, time
from generate_compensate import generate_compensate
from generate_truth import generate_fact_statement
from utils import Tools
os.chdir(os.path.dirname(__file__))
# 將 KG_RAG 目錄添加到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "KG_RAG_B"))
sys.path.append(os.path.join(os.path.dirname(__file__), "chunk_RAG"))
from KG_RAG_B.KG_Faiss_Query_3068 import query_simulation
from chunk_RAG.ts_main import retrieval
df = pd.read_csv('dataset.csv')
df2 = pd.read_csv('dataset(no_law).csv')
inputs = df["模擬輸入內容"].tolist()[:-2]
template_output = df2["gpt-4o-mini-2024-07-18\n3000筆"].tolist()
def generate_lawsheet(input_data):
    """處理單個生成請求並輸出結果"""
    userinput = input("請選擇使用的RAG資料庫(1: KG_RAG, 2: chunk_RAG): ")
    if userinput == "1":
        # 使用 KG_RAG
        references = query_simulation(input_data, 3)
    elif userinput == "2":
        # 使用 chunk_RAG
        references = retrieval(input_data, 3)
    else:
        print("請輸入正確的選項(1或2)")
        return None
    print(references)
    facts = []
    case_ids = []
    compensations = []
    data = tools.split_user_input(input_data)
    for i, reference in enumerate(references):
        splited_reference = tools.split_user_output(reference["case_text"])
        case_ids.append(reference["case_id"])
        if splited_reference == False:
            print(f"第{i}格式錯誤，不參考這筆資料")
            continue
        facts.append(splited_reference["fact"])
        compensations.append(splited_reference["compensation"])
    first_part = yield from generate_fact_statement(data["case_facts"] + '\n' + data["injury_details"], facts, tools)
    print(first_part)
    print("第一段生成完成")
    print("=" * 50)
    yield tools.show_debug_to_UI(f"{first_part}\n第一段生成完成\n{'=' * 50}")
    second_part = tools.generate_laws(case_ids, 2)
    print(second_part)
    print("第二段生成完成")
    print("=" * 50)
    yield tools.show_debug_to_UI(f"{second_part}\n第二段生成完成\n{'=' * 50}")
    third_part = yield from generate_compensate(input_data, compensations, tools)
    result = first_part + '\n\n' + second_part + '\n\n' + third_part
    print(result)
    yield tools.show_result_to_UI(result)
    return result
    

tmp_prompt = """一、事故發生緣由:
 被告於民國000年0月0日下午18時14分許，駕駛車號000-0000自用小客車，沿新北市淡水區淡金路往臺北方向行駛，行至該路段欲右轉進入中正東路一段時，應注意轉彎車應禮讓直行車先行，竟未注意即貿然右轉，當時訴外人葉銘倫正騎乘車號000-0000普通重型機車搭載原告，同向行駛在被告車輛之右側，因閃避不及遭被告所駕之自用小客車碰撞而人車倒地。另外，被告之行為涉及過失傷害，經臺灣士林地方法院112年度交易字第44號刑事判處拘役50日，上訴後，經鈞院刑事庭112年度交上易字第199號判決上訴駁回確定，可證明被告確實有過失。
 
 二、原告受傷情形:
 原告因本件車禍事故受有左肩、左前臂、左手擦傷、左踝擦挫傷等傷害。
 
 三、請求賠償的事實根據:
按民法第184條第1項前段、第191條之2本文、第193條第1項、第195條第1項前段，請求下列損害
 原告主張因本件車禍受傷至淡水馬偕紀念醫院、祐民聯合診所就醫，支出醫療費用5萬4,741元，並且在111年9月27日購買藥品支出1,890元，以上有淡水馬偕紀念醫院醫療費用收據、祐民聯合診所醫療費用收據可以證明，而後至112年仍需要雷射除疤、中醫調養門診、來往醫院交通費用、自費醫材等預估醫療費用18萬元，並且有113年3月29日美仕媞時尚醫美診所1萬8,444元醫療收據可以證明原告後續預估的醫療費用有後續治療之必要及實際支出。
 
 原告主張因受到系爭傷害，馬偕紀念醫院乙種診斷證明書上記載「患者甲○○於本院民國111年8月7日急診診1次…因患者甲○○左側足踝慢性傷口遲未癒合，為進一步治療，於本院自民國111年9月7日入院，於民國111年9月8日接受清創及人工真皮植皮手術，術後使用負壓照護系統。需專人照護一個月。」，因此原告因本件事故自111年9月7日至111年9月13日共7日住院治療，出院後需專人照護一個月，於111年9月13日至111年10月13日期間，以一般醫院全日看護之收費行情為2,000元以上計算，一共支出看護費用7萬4,000元（即每日2,000元×37日＝7萬4,000元）。
 
 原告主張本件事故發生前原告受僱康舒科技股份有限公司，每個月月薪7萬2,800元，因本件車禍受傷導致3個月期間無法工作，因此請求受有3個月不能工作之薪資損害大約共42萬元，有診斷證明書、110年度綜合所得稅各類所得資料清單、健保櫃檯個人投保紀錄等可以作為證據。
 
 原告主張因本件事故，衣褲毀損支出購置費用1,580元、因鞋子毀損支出購置費用1,393元；其機車因毀損而支出修理費用合計2萬0,900元。
 
 原告因本件事故受有左肩、左前臂、左手擦傷、左踝擦挫傷等傷害，並且經過持續回診之情形，精神上受有相當痛苦；而原告為大學畢業，原任職於康舒科技股份有限公司，月薪7萬2,800元，請求鈞院衡量原告所受傷勢、本件事故發生原因及兩造財產資力等一切情狀，命被告支付精神慰撫金99萬元，以填補原告所受非財產上損害。
"""
if __name__ == "__main__":
    start_time = time.time()
    tools = Tools("kenneth85/llama-3-taiwan:8b-instruct-dpo")
    for part, ref, audit in generate_lawsheet(tmp_prompt):
        # print(f"生成的內容:\n{part}")
        # print(f"參考資料:\n{ref}")
        # print(f"推理紀錄:\n{audit}")
        pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"\n執行時間: {hours}h {minutes}m {seconds}s")