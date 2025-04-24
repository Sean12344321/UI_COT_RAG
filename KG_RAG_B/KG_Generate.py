from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from define_case_type import get_case_type
from KG_Faiss_Query_3068 import query_faiss
from Neo4j_Query import get_statude_case
import re
import time
s="""
一、事故發生緣由:
被告甲○○係民國00年0月00日生，為12歲以上未滿18歲之人。甲○○於112年6月23日上午10時3分許，駕駛微型電動二輪車（下稱A車），沿臺南市永康區富強路1段由南往北方向行駛，行經富強路1段350號前時，本應注意慢車不得侵入快車道行駛，而依當時狀況，並無不能注意之情事，竟疏未注意而行駛內側快車道，適有原告駕駛車牌號碼000-000號普通輕型機車（下稱B車），於車道中停等，欲由東往西跨越車道，未看清來往車輛，致二車發生碰撞；又甲○○於本件車禍事故發生時之112年6月23日，已年滿14歲，為未滿18歲之限制行為能力人，且有識別能力，應由其法定代理人即被告乙○○與甲○○負連帶損害賠償責任。

二、原告受傷情形:
此次車禍造成原告因而受有雙下肢大片撕脫傷、骨盆骨折等傷害。

三、請求賠償的事實根據:
（一）醫療費用
原告主張其因本件車禍事故受傷，就醫急診、住院及門診治療期間，共支出醫療費用2萬3,877元，並有奇美醫院、柳營奇美醫院診斷證明書及電子收據為證。

（二）看護費用
按奇美醫院診斷證明書記載原告需專人照顧8周，因此原告由於本件車禍事故受傷，需專人照顧，於112年6月25日至112年7月8日之上半日，委請慈惠聘雇照顧服務員，支出1萬8,900元，於112年7月26日下半日至112年8月2日，委請群富旺看護中心看護，再支出1萬1,550元，並有慈惠聘雇照顧服務員費用單、群富旺看護中心看護費用收據聯為證。除此之外，原告在112年7月9日下半日至112年7月25日上半日、112年8月3日至112年9月7日，均由親屬提供看護，費用分別為2萬3,800元、7萬8,400元。

（三）增加生活上必要費用
原告因本件車禍事故受傷，購買各式醫療耗材，支出2,901元，有免用統一發票收據、估價單、統一發票、電子發票證明聯等為證。

（四）工作損失
原告於本件車禍事故發生前，從事家庭代工工作，每月收入以最低基本工資2萬6,400元計算，因本件車禍事故受傷，於112年6月23日至113年6月24日，共1年又2日不能工作，並以1年計，請求31萬6,800元。

（五）精神慰撫金
原告為國中畢業，於本件車禍事故發生前即已退休，每月領取勞保退休金1萬2,000元，無存款及負債，查原告因本件車禍事故，受有雙下肢大片撕脫傷、骨盆骨折等傷勢，經急診、住院接受雙下肢原位植皮重建手術，及多次門診治療，且原告於受傷治療及休養期間，應對於其行動、生活均造成諸多不便，原告精神上因此受有相當痛苦，爰請求精神慰撫金20萬元，以資慰藉。
"""
fact_template = PromptTemplate(
    input_variables=["case_facts"],
    template="""
你是一個台灣原告律師，你要撰寫一份車禍起訴狀，但你只需要根據下列格式進行輸出，並確保每個段落內容完整：
你只需要撰寫這份起訴狀中的事實概述部分
注意要點：完整描述事故經過，事件結果盡量越詳細越好，要使用"緣被告"做開頭，並且在這段中都要以"原告""被告"作人物代稱，如果我給你的案件事實中沒有出現原告或被告的姓名，則請直接使用"原告""被告"作為代稱，請絕對不要自己憑空杜撰被告的姓名
備註:請絕對不要寫出任何賠償相關的資訊
### 案件事實：
{case_facts}
"""
)
legal_template = PromptTemplate(
    input_variables=["case_facts", "legal_references"],
    template="""
你是一個台灣原告律師，你要撰寫一份車禍起訴狀，但你只需要根據下列格式進行輸出，並確保每個段落內容完整：
  你需要從我給你的案件事實還有我給你的相似案件中的引用法條部分，來寫出這份起訴書需要的引用法條
  模板:按「民法第A條條文」、「民法第B條條文」、...、「民法第N條條文」民法第A條、民法第B條、...、民法第N條分別定有明文。查被告因上開侵權行為，致原告受有下列損害，依前揭規定，被告應負損害賠償責任：
  雖然我給你的模板格式中有提到"被告應負損害賠償責任"，但請你在輸出的時候不要在這段後面加上任何的賠償資訊
  以下是使用此模板的範例，其中引用的法條僅供參考，輸出的時候要記得我給你的只是範例，如果其中有提到部分的案件內容(例如人名)，請一定要無視掉。:
  {legal_references}
  ### 案件事實，撰寫時請一定要以這裡的事實為主：
  {case_facts}
"""
)
comp_promt=PromptTemplate(
    input_variables=["injury_details", "compensation_request"],
    template="""
你是一個台灣原告律師，你要幫助原告整理賠償資訊，你只需要根據下列格式進行輸出，並確保每個段落內容完整：
要確保完全照著模板的格式輸出。
損害項目：列出所有損害項目的金額，並說明對應事實。
  模板：
    損害項目名稱： [損害項目描述]
    金額： [金額數字] 元
    事實根據： [描述此損害項目的原因和依據]
    備註:如果有多名原告，需要針對每一位原告列出損害項目
    範例:
    原告A部分:
    損害項目名稱1：...
    金額:..
    事實根據：...
    損害項目名稱2：...
    金額:..
    事實根據：...
    原告B分:
    損害項目名稱1：...
    金額:..
    事實根據：...
    損害項目名稱2：...
    金額:..
    事實根據：...
總賠償金額：需要將每一項目的金額列出來並總結所有損害項目，計算總額，並簡述賠償請求的依據。
  模板:
    損害項目總覽：
    總賠償金額： [總金額] 元
    賠償依據：
    依據 [法律條文] 規定，本案中 [被告行為] 對原告造成 [描述損害]，被告應負賠償責任。總賠償金額為 [總金額] 元。
 ### 受傷情形：
{injury_details}
### 賠償請求：
{compensation_request}
備註:請盡量不要使用#字號
"""
)
llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct-dpo-q8_0",temperature=0.1,keep_alive=0)
def generate_fact(input_data):
    # 創建 LLMChain
    llm_chain = LLMChain(llm=llm, prompt=fact_template)
    # 傳入數據生成起訴書
    lawsuit_draft = llm_chain.run({
        "case_facts": input_data,
    })
    return lawsuit_draft

def generate_legal(input_data, case_type):
     # 創建 LLMChain
    llm_chain = LLMChain(llm=llm, prompt=legal_template)
    # 查詢最相似的案件
    closest_cases = query_faiss(input_data, case_type,1)
    ids=[i['id'] for i in closest_cases]
    legal_references = []
    for i in ids:
        # 查詢法條資訊
        legal_reference = get_statude_case(i)
        if legal_reference:
            legal_references.append(legal_reference)
    # 提取最相似案件的法條資訊
    # 傳入數據生成起訴書
    lawsuit_draft = llm_chain.run({
        "case_facts": input_data,
        "legal_references": legal_references
    })
    return lawsuit_draft

def generate_comp(user_input):
    input_data=split_input(user_input)
    llm_chain = LLMChain(llm=llm, prompt=comp_promt)
    # 傳入數據生成起訴書
    lawsuit_draft = llm_chain.run({
        "injury_details": input_data["injury_details"],
        "compensation_request": input_data["compensation_request"]
    })
    return lawsuit_draft

def split_input(user_input):
    sections = re.split(r"(一、|二、|三、)", user_input)
    input_dict = {
        "case_facts": sections[2].strip(),
        "injury_details": sections[4].strip(),
        "compensation_request": sections[6].strip()
    }
    return input_dict

def generate_lawsuit(user_input):
    start_time = time.time()  # 記錄開始時間
    input_dict = split_input(user_input)
    case_type=get_case_type(user_input)
    case_facts = input_dict["case_facts"]
    fact=generate_fact(case_facts)
    #print(fact)
    #print()
    legal=generate_legal(input_dict["case_facts"], case_type)
    #print(legal)
    #print()
    comp=generate_comp(user_input)
    end_time = time.time()  # 記錄結束時間
    execution_time = end_time - start_time  # 計算執行時間
    print(f"執行時間: {execution_time} seconds")
    return fact+"\n\n"+legal+"\n\n"+comp

#start_time = time.time()  # 記錄開始時間
#l=generate_lawsuit(s)
#print(l)
#end_time = time.time()  # 記錄結束時間
#execution_time = end_time - start_time  # 計算執行時間
#print(f"執行時間: {execution_time} seconds")