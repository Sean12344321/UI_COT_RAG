import pandas as pd
import multiprocessing, os, sys
from groq import Groq
from ollama import chat, ChatResponse
# 將 KG_RAG 目錄添加到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "KG_RAG"))
df = pd.read_csv('dataset.csv')
df2 = pd.read_csv('dataset(no_law).csv')
inputs = df["模擬輸入內容"].tolist()[:-2]
template_output = df2["gpt-4o-mini-2024-07-18\n3000筆"].tolist()
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
# 單位原告
money_prompt_singlePerson = """請依照以下格式撰寫，輸出必須分為兩個部分：
1. 逐一列出原告的費用，將費用依功能進行歸類，例如：「醫療花費」、「精神慰撫金」、「工作損失」等，而非列出過於詳細的細項（如具體診所名稱）。
2. 最後統一撰寫「綜上所陳」段落，總結這位原告的損害賠償金額。
請注意：
- 僅列出金額大於 0 的項目，若某項費用為 0 元，請忽略該項目，不需顯示。

範例如下：
(一) {費用項目1}費用：台幣{費用金額1}元
   原告因{事故原因1}導致{後果或需要醫療的情況1}，產生相關費用或損失，共計{費用金額1}元。
(二) {費用項目2}費用：台幣{費用金額2}元
   原告因{事故原因2}導致{後果或需要醫療的情況2}，產生相關費用或損失，共計{費用金額2}元。
(三) 綜上所陳，應連帶賠償原告{受害者名稱1}之損害，包含{費用項目1}費用{費用金額1}元，{費用項目2}費用{費用金額2}元，共計{總金額}元；應連帶賠償原告{受害者名稱2}之損害，包含{費用項目1}費用{費用金額1}元，共計{總金額}元。
"""
#多位原告
money_prompt_multiplePeople = """請依照以下格式撰寫，輸出必須分為兩個部分：
1. 逐一列出每位原告的費用，將費用依功能進行歸類，例如：「醫療花費」、「精神慰撫金」、「工作損失」等，而非列出過於詳細的細項（如具體診所名稱）。
2. 最後統一撰寫「綜上所陳」段落，總結所有原告的損害賠償金額。
請注意：
- 僅列出金額大於 0 的項目，若某項費用為 0 元，請忽略該項目，不需顯示。
- 以下範例僅供參考，實際原告人數與項目可能不同，請完整依照實際資料格式呈現。

範例如下：
(一) 原告{受害者名稱1}:
1. {費用項目1}費用：台幣{費用金額1}元
   原告因{事故原因1}導致{後果或需要醫療的情況1}，產生相關費用或損失，共計{費用金額1}元。
2. {費用項目2}費用：台幣{費用金額2}元
   原告因{事故原因2}導致{後果或需要醫療的情況2}，產生相關費用或損失，共計{費用金額2}元。

(二) 原告{受害者名稱2}:
1. {費用項目1}費用：台幣{費用金額1}元
   原告因{事故原因1}導致{後果或需要醫療的情況1}，產生相關費用或損失，共計{費用金額1}元。
(若有有多位原告，請再列出{受害者名稱3}、{受害者名稱4}等)
(三) 綜上所陳，應連帶賠償原告{受害者名稱1}之損害，包含{費用項目1}費用{費用金額1}元，{費用項目2}費用{費用金額2}元，共計{總金額}元；應連帶賠償原告{受害者名稱2}之損害，包含{費用項目1}費用{費用金額1}元，共計{總金額}元。
"""
truth_prompt = """
請以專業的法律用語描述事故發生經過，並確保語言嚴謹，條理清晰。
以"緣被告/原告"開頭，不要有換行符號，把輸入整合成一個段落。
範例輸出:
緣被告於民國111年8月4日23時17分許，無照騎駛車牌號碼000-0000號普通重型機車，沿新北市三重區中正北路往蘆洲方向行駛，行經中正北路及中正北路312巷之交岔路口時，本應注意機車行駛時，駕駛人應注意車前狀況，並隨時採取必要之安全措施，而依當時天候晴，夜間有照明，柏油路面乾燥、無缺陷、無障礙物，視距良好，並無不能注意之情事，被告竟疏未注意及此，即貿然前行，適前方有原告林肜宇騎駛車牌號碼000-000號普通重型機車並搭載原告吳彩雲，沿同方向行駛，亦行經該處，雙方因而發生碰撞，致原告均人車倒地。被告已經鈞院113年度審交簡字第50號刑事簡易判決判處拘役40日有罪，足證被告針對本件事故之發生具有過失。
"""
tmp_input = """一、事故發生緣由:
 被告甲○○未領有適當駕駛執照，竟於110 年10月9 日14時3 分許，駕駛車牌號碼000-0000號自小客車（下稱肇事車輛）搭載被告乙○○，沿新竹市東區中華路1段由北往南行駛至中華路1段78巷口附近，本應注意車前狀況，隨時採取必要之安全措施，而依當時之情形，並無不能注意之情事，竟疏未注意及此，不慎追撞同向前方由原告丁○○所駕駛、搭載原告丙○○之車牌號碼000-0000號自小客車（下稱系爭車輛）。又被告甲○○上開不法行為，本案業經臺灣新竹地方檢察署（下稱新竹地檢署）檢察官偵查後提起公訴，並經鈞院刑事庭以111 年度竹交簡字第246 號刑事簡易判決判處有罪，是被告甲○○應負損害賠償責任。
 按「汽車駕駛人，未領有駕駛執照駕駛小型車者，處6,000元以上12,000元以下罰鍰，並當場禁止其駕駛；汽車所有人允許第1項第1款之違規駕駛人駕駛其汽車者，除依第1 項規定之罰鍰處罰外，並記該汽車違規紀錄1 次。」道路交通管理處罰條例第21條第1項第1款定有明文，而上開法律規定，應係保護他人之法律，依民法第184條第2項，違反時應推定其有過失；查被告乙○○為肇事車輛之所有權人，其於出借該車時本應查核借用人有無合格駕駛執照，但被告乙○○明知被告甲○○為未成年人，然其竟任意將系爭車輛出借予無駕駛執照之被告甲○○，已屬違反保護他人法律，自應認屬共同侵權行為人而負連帶賠償之責。
 
 二、原告受傷情形:
 原告丁○○、丙○○均受有頭部外傷之傷害。
 
 三、請求賠償的事實根據:
1.原告丁○○、丙○○請求之醫療費用部分：
 原告2人主張其等因本件事故受傷就醫治療，而分別支出醫療費用740元、400元，並且有提出中醫新竹分院之診斷證明書及醫療收據等件影本作為證據。
 
 2.原告丁○○、丙○○請求之交通費用：
 原告2人並主張其等因傷往返住家與中醫新竹分院所支出之交通費用分別為1,760元、880元，並提出預估車資網頁資料及GOOGLE地圖路線圖為證。
 
 3.原告丁○○、丙○○請求之不能工作之薪資損失部分：
 原告2人再主張其等請假治療傷勢，而無法工作1週，每月平均薪資各為35,000元、56,000元，分別受有薪資損失8,167元、13,067元。
 
 4.原告丁○○請求之系爭車輛修理費用部分：
 原告丁○○主張系爭車輛因本件車禍事故受損，支付修復費用98,424元，而系爭車輛所有人徐玉桂已將系爭車輛損害賠償請求權讓與原告丁○○，並有維修明細表、電子發票及債權讓與證明書為證據。
 
 5.原告丁○○、丙○○請求之精神慰撫金部分：
 查原告丁○○、丙○○因本件事故受有「頭部外傷」之傷害，堪認身心均受有痛苦，因此原告丁○○、丙○○分別向被告請求50,000元、30,000之精神慰撫金，請求鈞院衡量原告所受傷勢、本件事故發生原因及兩造財產資力等一切情狀，命被告賠償精神慰撫金50,000元、30,000予原告2人。
"""

def generate_response(input_data):
    """發送單次生成請求並回傳結果"""
    try:
        # chat_completion = client.chat.completions.create(
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": input_data,
        #         }
        #     ],
        #     model="llama-3.1-8b-instant",
        # )
        # return chat_completion.choices[0].message.content
        response: ChatResponse = chat(
            messages=[
                {
                    'role': 'user',
                    'content': input_data,
                },
            ],    
            model='llama3.1', 
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
    output = output[:-1] + "分別定有明文。查被告因上開侵權行為，致原告受有下列損害，依前揭規定，被告應負損害賠償責任："
    return output


def check_and_generate_lawsuit(input_data):
    if any(char in input_data["money"] for char in ['{', '}']):
        print("money has {}")
        return None
    if "(一)" not in input_data["money"]:
        print("money has no (一)")
        return None
    if '\n' in input_data["truth"]:
        print("truth has \n")
        return None
    input_data["money"] = input_data["money"].replace('*', '')
    input_data["money"] = input_data["money"].split("(一)", 1)[1]
    input_data["money"] = "(一)" + input_data["money"]

    return f"{input_data["truth"]}\n\n{input_data["law"]}\n\n{input_data["money"]}"

def worker(input_data):
    """處理單個生成請求並輸出結果"""
    from KG_RAG.KG_Generate import generate_legal_references, split_input
    data = split_input(input_data)
    laws = generate_legal_references(data["case_facts"], data["injury_details"])
    truth_input = f"""{data["case_facts"]}
{data["injury_details"]}
{truth_prompt}"""
    money_input = f"""輸入:
{data["compensation_request"]}
{money_prompt_multiplePeople}"""
    first_part = "一、" + generate_response(truth_input)
    second_part = convert_law(laws)
    third_part = generate_response(money_input)
    return {'truth': first_part, 'law': second_part, 'money': third_part}

def generate_output(input):
    inputs = [input] * 3
    with multiprocessing.Pool(processes=5) as pool:
        # 使用 map 同時對5個輸入進行處理
        lawsuits = pool.map(worker, inputs)
        result = ""
        for lawsuit_array in lawsuits:
            if check_and_generate_lawsuit(lawsuit_array) != None:
                print(check_and_generate_lawsuit(lawsuit_array))
                print('-' * 50)
        return result
generate_output(tmp_input)
