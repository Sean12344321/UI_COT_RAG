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
    with multiprocessing.Pool(processes=1) as pool:
        # 使用 map 同時對5個輸入進行處理
        lawsuits = pool.map(worker, inputs)
        for lawsuit_array in lawsuits:
            if check_and_generate_lawsuit(lawsuit_array) != None:
                print(check_and_generate_lawsuit(lawsuit_array))
                print('-' * 50)
                return check_and_generate_lawsuit(lawsuit_array)
        return "Error: 生成訴訟書失敗"