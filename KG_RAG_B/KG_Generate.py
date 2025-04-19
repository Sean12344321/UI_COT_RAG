from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from KG_RAG_B.KG_Faiss_Query import query_faiss, get_statutes_for_case, get_simulation_output
import re
# 函數：生成引用的法條
def generate_legal_references(case_facts, injury_details):
    input_text = f"{case_facts} {injury_details}"
    similar_inputs = query_faiss(input_text, top_k=5)
    statutes_set = set()
    for fact in similar_inputs:
        fact_id = fact["case_id"]
        statutes_info = get_statutes_for_case(fact_id)
        for info in statutes_info:
            statutes_set.update(info["statutes"])

    # legal_references = "\n".join(sorted(statutes_set))
    # return legal_references
def query_simulation(input_text):
    # 1. 查詢最相近的 "模擬輸入"
    print("在faiss中查詢5個模擬輸入")
    sim_inputs = query_faiss(input_text, top_k=5)
    # 2. 查詢對應的 "模擬輸出"
    print("在neo4j中找到對應的起訴狀")
    results = []
    for sim_input in sim_inputs:
        sim_outputs = get_simulation_output(sim_input["id"])
        results.append(sim_outputs[0]["text"])
    return results

# 定義提示模板
prompt_template = PromptTemplate(
    input_variables=["case_facts", "injury_details", "compensation_request", "legal_references"],
    template="""
你是一個台灣原告律師，你要撰寫一份車禍起訴狀，請根據下列格式進行輸出，並確保每個段落內容完整：
（一）事實概述：完整描述事故經過，事件結果及要求賠償盡量越詳細越好
（二）法律依據：先對每一條我給你的引用法條做判斷，如過確定在這起案件中要引用，列出所有相關法律條文，並對每一條文做出詳細解釋與應用。
  模板：
  - 民法第xxx條第x項：「...法律條文...」。
    - 案件中的應用：本條適用於 [事實情節]，因為 [具體行為] 屬於 [法條描述的範疇]，因此 [解釋為何負責賠償]。
（三）損害項目：列出所有損害項目的金額，並說明對應事實。
  模板：
    損害項目名稱： [損害項目描述]
    金額： [金額數字] 元
    事實根據： [描述此損害項目的原因和依據]
（四）總賠償金額：需要將每一項目的金額列出來並總結所有損害項目，計算總額，並簡述賠償請求的依據。
  模板:
    損害項目總覽：
    總賠償金額： [總金額] 元
    賠償依據：
    依據 [法律條文] 規定，本案中 [被告行為] 對原告造成 [描述損害]，被告應負賠償責任。總賠償金額為 [總金額] 元。
### 案件事實：
{case_facts}
### 受傷情形：
{injury_details}
### 引用法條：
{legal_references}
### 賠償請求：
{compensation_request}
"""
)
user_input="""一、事故發生緣由:
 被告於民國000年0月00日下午5時許，駕駛車牌號碼0000-00號自用小客車（下稱A車），自基隆市中正區碧砂漁港駛出欲右轉往中正路方向行駛之際，於該址路口（下稱系爭地點）本應注意行駛在閃紅燈之支線道車應暫停讓幹線道車先行，以避免危險或交通事故之發生，而依當時天候及路況，並無不能注意之情事，竟疏未注意上情而貿然前行，此時原告甲父騎乘車牌號碼000-000號普通重型機車（下稱B車）附載原告甲及甲母行駛至系爭地點，雙方因而發生碰撞（下稱系爭事故），原告甲、甲父、甲母因此分別受有身體傷害及財物損失。
 
 二、原告受傷情形:
 原告甲父因系爭事故受有脾臟重度撕裂傷合併腹內出血及休克、左側第6至第9肋骨骨折、左小腿及左腳踝大面積傷口、左側肺挫傷併少量血胸等傷害。
 原告甲母因系爭事故受有四肢及臉部多處挫傷及擦傷。
 原告甲因系爭事故受有左膝挫傷、擦傷及左手第五掌骨閉鎖性骨折之傷害。
 
 三、請求賠償的事實根據:
原告甲父部分：
 1.醫療費用：原告甲父因系爭事故受有脾臟重度撕裂傷合併腹內出血及休克、左側第6至第9肋骨骨折、左小腿及左腳踝大面積傷口、左側肺挫傷併少量血胸等傷害，為此赴衛生福利部基隆醫院（下稱部立基隆醫院）、長庚財團法人基隆長庚紀念醫院（下稱基隆長庚醫院）就醫，支出醫療費用合計5萬3,715元。
 2.看護費用：原告甲父因受有前揭傷害，自110年2月16日起至110年3月3日在基隆長庚醫院住院治療並接受腹腔鏡血塊引流手術，住院期間需專人全日照顧，術後則因脾臟重度撕裂傷，需休養1個月；肋骨骨折部分則需休養3個月，而有接受專人照顧3個月之需求。故原告甲父主張住院期間以每日看護新臺幣2,500元計算，休養3個月部分則以半日看護費用1,250元計算，合計得請求看護費用為15萬2,500元。
 3.交通費用：原告甲父因就醫需求，於110年2月16日曾自費搭乘救護車前往急診，其後並多次搭乘計程車往返住家及部立基隆醫院及基隆長庚醫院，合計支出7,215元，惟此處僅以965元為度，請求被告賠償其中965元。
 4.工作收入損失：原告甲父於系爭事故發生時在生達化學製藥股份有限公司任職，於事發前1年（即110年度）之全年薪資為134萬0,292元，折算日薪為3,723元，又原告甲父因受前揭傷害而有3個月不能工作，於休養期間之工作收入損失合計為33萬5,070元。
 5.精神慰撫金：原告因系爭事故受有脾臟重度撕裂傷，導致腹內出血、休克進入加護病房治療，住院期間並先後接受血管栓塞止血及腹內血塊引流手術，且出院後尚需休養3個月，迄今仍覺身體機能無法回復、不時疼痛，因此身心俱疲，所受打擊非屬一般，故請求被告賠償精神慰撫金80萬元。
 6.B車受損修理費用及甲父所有之手機、眼鏡、手錶、衣服、安全帽價值損失：原告甲父因系爭事故需支出B車維修費用7,977元，並因系爭事故造成身上攜帶之手機、眼鏡、手錶；身著之安全帽及衣服破損，受有價值相當於6,250元之損害，合計受有1萬4,227元之損失。
 
 原告甲母部分：
 精神慰撫金：原告甲母因系爭事故受有四肢及臉部多處挫傷及擦傷，因此破相，且傷口難免留疤而永久影響外觀。職故，原告甲母確因系爭事故受有精神上痛苦，爰請求被告賠償精神慰撫金5萬元。
 
 原告甲部分：
 精神慰撫金：原告甲於系爭事故發生時年紀尚輕，因系爭事故受有左膝挫傷、擦傷及左手第五掌骨閉鎖性骨折之傷害，經多次回診治療，左手掌需以外物固定而無法彎曲，生活學業受有很大影響，且有永久影響左手握力與日常生活功能之虞。職故，原告確因系爭事故受有精神上痛苦，爰請求被告賠償精神慰撫金15萬元。

"""

def split_input(user_input):
    sections = re.split(r"(一、|二、|三、)", user_input)
    input_dict = {
        "case_facts": sections[2].strip(),
        "injury_details": sections[4].strip(),
        "compensation_request": sections[6].strip()
    }
    return input_dict

def generate_lawsuit(user_input):
    input_data=split_input(user_input)
    legal_references = generate_legal_references(input_data["case_facts"], input_data["injury_details"])
    input_data["legal_references"] = legal_references
    llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct",
                    temperature=0.1,
                    keep_alive=0,
                    num_predict=50000
                    )
    # 創建 LLMChain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    # 傳入數據生成起訴書
    # lawsuit_draft = llm_chain.run({
    #     "case_facts": input_data["case_facts"],
    #     "injury_details": input_data["injury_details"],
    #     "legal_references": legal_references,
    #     "compensation_request": input_data["compensation_request"]
    # })
    return legal_references

# print(query_simulation(user_input))