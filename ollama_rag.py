import pandas as pd
import os, sys
from ollama import chat, ChatResponse
from generate_compensate import generate_compensate
from generate_truth import generate_fact_statement
os.chdir(os.path.dirname(__file__))
# 將 KG_RAG 目錄添加到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "KG_RAG"))
df = pd.read_csv('dataset.csv')
df2 = pd.read_csv('dataset(no_law).csv')
inputs = df["模擬輸入內容"].tolist()[:-2]
template_output = df2["gpt-4o-mini-2024-07-18\n3000筆"].tolist()

# 單位原告
# money_prompt_singlePerson = """請依照以下格式撰寫，輸出必須分為兩個部分:
# 1. 逐一列出原告的費用，將費用依功能進行歸類，例如:「醫療花費」、「精神慰撫金」、「工作損失」等，而非列出過於詳細的細項(如具體診所名稱)。
# 2. 最後統一撰寫「綜上所陳」段落，總結這位原告的損害賠償金額。
# 請注意:
# - 僅列出金額大於 0 的項目，若某項費用為 0 元，請忽略該項目，不需顯示。

# 範例如下:
# (一) {費用項目1}費用:台幣{費用金額1}元
#    原告因{事故原因1}導致{後果或需要醫療的情況1}，產生相關費用或損失，共計{費用金額1}元。
# (二) {費用項目2}費用:台幣{費用金額2}元
#    原告因{事故原因2}導致{後果或需要醫療的情況2}，產生相關費用或損失，共計{費用金額2}元。
# (三) 綜上所陳，應連帶賠償原告{受害者名稱1}之損害，包含{費用項目1}費用{費用金額1}元，{費用項目2}費用{費用金額2}元，共計{總金額}元；應連帶賠償原告{受害者名稱2}之損害，包含{費用項目1}費用{費用金額1}元，共計{總金額}元。
# """
#多位原告
def generate_response(input_data):
    """發送單次生成請求並回傳結果"""
    try:
        response: ChatResponse = chat(
            messages=[
                {
                    'role': 'user',
                    'content': input_data,
                },
            ],
            model='kenneth85/llama-3-taiwan:8b-instruct-dpo',
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"

def convert_law(laws):
    transfer = {
    "民法第184條": "「因故意或過失，不法侵害他人之權利者，負損害賠償責任。」民法第184條第1項前段",
    "民法第185條": "「汽車、機車或其他非依軌道行駛之動力車輛，在使用中加損害於他人者，駕駛人應賠償因此所生之損害。但於防止損害之發生，已盡相當之注意者，不在此限。」民法第185條第1項",
    "民法第187條": "「無行為能力人或限制行為能力人，不法侵害他人之權利者，以行為時有識別能力為限，與其法定代理人連帶負損害賠償責任。」民法第187條第1項",
    "民法第188條": "「受僱人因執行職務，不法侵害他人之權利者，由僱用人與行為人連帶負損害賠償責任。」第188條第1項本文",
    "民法第191-2條": "「汽車、機車或其他非依軌道行駛之動力車輛，在使用中加損害於他人者，駕駛人應賠償因此所生之損害。但於防止損害之發生，已盡相當之注意者，不在此限。」民法第191條之2",
    "民法第193條": "「不法侵害他人之身體或健康者，對於被害人因此喪失或減少勞動能力或增加生活上之需要時，應負損害賠償責任。」民法第193條第1項",
    "民法第195條": "「不法侵害他人之身體、健康、名譽、自由、信用、隱私、貞操，或不法侵害其他人格法益而情節重大者，被害人雖非財產上之損害，亦得請求賠償相當之金額。」民法第195條第1項",
    }
    for original, replacement in transfer.items():
        if original in laws:
            laws = laws.replace(original, replacement)
    # 切分 input，提取每條條文
    laws = laws.split("「")
    laws = [law.strip("」").strip() for law in laws if law]
    contents = ['「' + law.split("民法")[0] for law in laws]
    clauses = [law.split("民法")[1] for law in laws if len(law.split("民法")) > 1]
    # 法條部分的合併
    output = "二、按"
    for index, content in enumerate(contents):
        output += content
        if index != len(contents) - 1:
            output += "、"
    output += "民法"
    for clause in clauses:
        output += clause + '、'
    # 在最後補上相應的引用，並更正格式
    output = output[:-1] + "分別定有明文。查被告因上開侵權行為，致原告受有下列損害，依前揭規定，被告應負損害賠償責任:"
    return output

def generate_lawsheet(input_data):
    """處理單個生成請求並輸出結果"""
    from KG_RAG.KG_Generate import generate_legal_references, split_input
    data = split_input(input_data)
    laws = generate_legal_references(data["case_facts"], data["injury_details"])
    first_part = generate_fact_statement(data["case_facts"] + '\n' + data["injury_details"])
    second_part = convert_law(laws)
    third_part = generate_compensate(data["compensation_request"])
    return first_part + '\n\n' + second_part + '\n' + third_part 

tmp_prompt = """一、事故發生緣由:
被告於民國000年0月00日下午5時35分許，駕駛車牌號碼000-0000號營業自小客車，沿國道三號高速公路由南往北方向行駛，行經桃園市○○區○道○號66公里100公尺處時，本應注意車前狀況並隨時採取必要之安全措施，而依當時天候晴、日間自然光線、柏油路面乾燥、無缺陷、無障礙物阻擋、視距良好，並無不能注意之情事，竟疏未注意及此，即貿然前行，適有原告陳皆宏駕駛車牌號碼000-0000號自用小客車（下稱系爭車輛），車上搭載乘客訴外人王惠滿，另一原告王惠華，沿同向行駛於上揭被告所駕駛車輛前方，嗣被告所駕上揭車輛竟自後追撞系爭車輛(下稱系爭事故)。

二、原告受傷情形:
系爭事故導致原告陳皆宏因而受有頭痛之傷害；原告王惠滿受有腦震盪之傷害。

三、請求賠償的事實根據:
原告陳皆宏為治療系爭事故所生之傷勢，前往廣福診所醫院治療，合計醫療費用支出466元。系爭事故後，因系爭車輛經拖吊，當日即有返家之交通費用1,855元支出。原告陳皆宏於治療系爭傷勢期間支出交通費用700元。系爭車輛於修復期間，故而無法使用，故而上下班、快出會議拜訪客戶、接送小孩、數次前往調解、法院之交通費用12,820元。末原告陳皆宏因系爭車輛無法使用，故而有以租賃代步，期間一個月之費用為34,000元。系爭車輛因系爭事故致交易價格減損，經原告向第三方鑑定單位鑑定後，系爭車輛事故發生前價值約950,000元，修復後之價值為840,000元，顯見其市場交易之價值貶損為100,000元。原告陳皆宏因本件系爭事故，前後前往調解至須請假而有工作損失3,000元，嗣因此遭公司非自願離職，為此請求損失5,000元；上合計8,000元。系爭車輛受損後，原告被迫接受無法用車之不便、需另外於住家與公司重新設定、已規劃之行程取消，及額外支出心力安撫親戚家人，故請求精神慰撫金10,000元。而原告王惠滿則為治療系爭事故所生之傷勢，前後前往臺北市立聯合醫院(下稱聯合醫院)、全心中醫診所回診、追蹤等治療，為此之支出醫療費用13,532元。並因受有系爭傷勢，致有後續門診回診之必要，並因此多次前往醫院就診復健，合計迄今原告支出之交通費用為23,525元。又因為系爭腦震盪傷勢致受有相當精神上之痛苦，故請求精神慰撫金72,000元。"""

print(generate_lawsheet(tmp_prompt))

