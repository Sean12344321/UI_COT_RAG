import pandas as pd
import multiprocessing, os, sys
from ollama import chat, ChatResponse
import re
os.chdir(os.path.dirname(__file__))
# 將 KG_RAG 目錄添加到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "KG_RAG"))
df = pd.read_csv('dataset.csv')
df2 = pd.read_csv('dataset(no_law).csv')
inputs = df["模擬輸入內容"].tolist()[:-2]
template_output = df2["gpt-4o-mini-2024-07-18\n3000筆"].tolist()
#output prompt
truth_score_prompt = """請根據以下三個版本的起訴狀內容，擷取最佳部分，生成最終版本。請確保：

1. 輸出為一整個段落，不要有換行，讓內容更加連貫。
2. 句子結構清晰、流暢，避免冗長或重複。
3. 事故敘述完整，包括事故過失、原告受傷情形。
提供的三個版本如下："""

money_score_prompt = """請根據賠償金額段落的完整性、合理性與法律適用性，評分該段落的品質，並輸出分數（0-100分），同時提供簡要評語。

評分標準：

表達清晰度（50%） - 是否有意義不明的文字，如「XX元」，或不合理的內容。
完整結尾（50%） - 是否輸出到一半就暫停，沒有完整結尾。
輸出格式：  
表達清晰度評語: XXX
完整結尾評語：XXX
分數：XX分"""
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
money_prompt_multiplePeople = """指示：
請根據輸入的賠償金額資料，生成賠償金額，格式需符合以下要求：

依照原告分類，先寫「（一）原告 X 部分」，再依序列出各項賠償金額。
逐項列出費用，每項費用需包含：
費用名稱（如「醫療費用」、「薪資損失」、「車損修理費用」等）。
具體金額與簡要原因。
綜合計算，最後加總每位原告的請求金額，並統整最終總賠償金額。
輸出格式範例：

（一）原告 X 部分：
1.醫療費用：XXX 元
原告 X 因本次事故受傷，前往 XXX 醫院治療，支出醫療費用 XXX 元。

2.車損修理費用：XXX 元
原告 X 所有之系爭車輛因本次事故受損，支出修復費用 XXX 元。

3.薪資損失：XXX 元
原告 X 因本次事故受傷無法工作，遭受薪資損失 XXX 元。

4.慰撫金：XXX 元
原告 X 於事故傷害，造成原告精神上極大痛苦，爰請求慰撫金 XXX 元。

（二）原告 Y 部分：
（依照相同格式列出）

（三）綜上所陳，被告應賠償原告 X 之損害，包含 醫療費用 XXX 元、車損修理費用 XXX 元及慰撫金 XXX 元，總計 XXX 元；應賠償原告 Y 之損害，包含 醫療費用 XXX 元、薪資損失 XXX 元及慰撫金 XXX 元，總計 XXX 元。兩原告合計請求賠償 XXX 元。

請嚴格按照此格式，確保邏輯清晰、條理分明。
"""
truth_prompt = """
請根據提供的交通事故基本資訊，生成一段法律起訴狀中「事實陳述」的完整段落，需符合正式法律文書格式，並包含以下要素：

1. 時間地點 - 事故發生的具體時間（民國年月日、時刻）與地點（城市、區域、道路名稱等）。
2. 當事人 - 涉事車輛駕駛人、乘客（如有）及受害者的身分描述（如「原告甲」）。
3. 車輛資訊 - 各方車輛之類型（小客車、機車等）與車牌號碼。
4. 事故過程 - 涉事駕駛之具體違規行為（如未遵守號誌、不讓幹道車先行等）。
5. 後果 - 事故造成的影響，包括人員受傷、財物損壞等。
輸出格式要求：

用正式法律文書語言表述，避免情緒化或模糊用詞。
敘述完整，邏輯清晰，條理分明。
不可遺漏事故發生的時間、地點、當事人、車輛資訊、事故過程與後果。
範例輸出（僅供參考，請根據具體情境變更內容）：
「一、緣被告於民國○○○年○月○日○時○分許，駕駛車牌號碼○○○○-○○號自用小客車，行經○○市○○區○○路段，於駛入幹道時，依規定應禮讓幹道車輛先行，惟被告疏未注意，貿然駛出，致與原告甲騎乘之車牌號碼○○○-○○○號普通重型機車發生碰撞，導致原告甲受有○○傷害，並造成車輛損壞。」

請依照上述要求，根據具體事故資訊生成完整事實陳述段落。
"""

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
            model='kenneth85/llama-3-taiwan:8b-instruct',
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

def generate_lawsheet(input_data):
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

def generate_money_score(input):
   money_input = f"""輸入:{input}
{money_score_prompt}"""
   money_output = generate_response(money_input)
   numbers = re.findall(r"\d+", money_output)
   return int(numbers[0])
def generate_best_lawsheet_among_three(input):
   inputs = [input] * 3
   with multiprocessing.Pool(processes=1) as pool:
      # 使用 map 同時對3個輸入進行處理
      lawsuits = pool.map(generate_lawsheet, inputs)
      final_truth = f"""{truth_score_prompt}
1. {lawsuits[0]['truth']}
2. {lawsuits[1]['truth']}
3. {lawsuits[2]['truth']}"""
      print(generate_response(final_truth))
      print(lawsuits[0]['law'])
      money_score = pool.map(generate_money_score, [lawsuits[0]['money'], lawsuits[1]['money'], lawsuits[2]['money']])
      best_lawsheet_index = money_score.index(max(money_score))
      print(lawsuits[best_lawsheet_index]['money'])
      return "Error: 生成訴訟書失敗"
    
tmp_prompt = """一、事故發生緣由:
被告於110年12月13日10時1分許，駕駛車號00-0000號自用小客車，沿臺中市西屯區港隆街由凱旋路往黎明路方向行駛至黎明路岔路口時，因疏於注意減速讓幹道車優先通行，即貿然行駛，致發生與李承祐所騎車號000-0000普通重型機車（下稱系爭機車，李承祐當時係騎系爭機車搭載賴秀雲，沿黎明路由東往西方向行駛）擦撞之車禍事故（下稱系爭車禍事故）。

二、原告受傷情形:
因本件事故導致賴秀雲受有右肘、右足、右側胸壁挫傷、右足撕裂傷等傷害；李承祐受有腰椎椎間盤突出之傷害。

三、請求賠償的事實根據:
原告賴秀雲主張因系爭車禍事故受有上開傷害，而前往臺中榮民總醫院（下稱臺中榮總）就醫花費9,160元，於高堂中醫診所花費4萬4628元，花費蕭永明骨科診所診療費1萬5350元、新奇美診所診療費4,100元。
並且因系爭車禍事故受傷，支出紗布及藥水衍生損失4,050元，並有發票及收據為證。
其因系爭車禍事故受傷，受有不能工作之損失總計10週，金額合計6萬2970元，並有臺中榮總診斷證明書、員工薪資單為證。
並因系爭車禍事故而導致系爭機車損壞，維修之零件金額為1萬8200元，工資金額為1萬800元，零件經計算折舊後，加計工資之維修費用結果為1萬2620元，，有系爭機車維修估價單為證。
而賴秀雲之眼鏡架因系爭車禍事故損壞，損失4,250元。
另查賴秀雲為00年00月00日出生，迄至系爭車禍事故發生時之110年12月13日，已年滿71歲，對照其因系爭車禍事故，受有右肘、右足、右側胸壁挫傷、右足撕裂傷等傷害，衡情應有搭車前往就醫之必要其因系爭車禍事故受傷至醫院治療，故一共支出計程車費用2萬1225元。
賴秀雲更因系爭車禍事故受傷後，受有勞動力減損2萬3622元。
末查賴秀雲為國小畢業，目前已退休，無收入，系爭車禍事故發生前月薪約2萬6000元，無存款、無汽車及不動產，因系爭車禍事故受有右肘、右足、右側胸壁挫傷、右足撕裂傷等傷害，造成精神痛苦，爰請求精神慰撫金30萬元。

原告李承祐主張因系爭車禍事故受有上開傷害，前往臺中榮總就診花費醫藥費用9,290元、高堂聯合中醫診所醫藥費用1萬2000元以及蕭永明骨科診所醫藥費用2萬3150元。
另外李承祐因傷復健需要購買護腰莢1,242元。
而李承祐因上開傷害而不良於行，需支出就醫之交通費用1萬9755元。
李承祐並主張系爭車禍事故經其送交臺中市車輛行車事故鑑定委員會、臺中市車輛行車事故鑑定覆議委員會進行鑑定、覆議，支出鑑價費用5,000元，請求被告應給付一半即2,500元，並有鑑價費之統一發票為證。
李承祐另外主張因受有上開傷害而無法上班，受有薪資損失57萬6000元。
李承祐因系爭車禍事故經鑑定勞動能力減損12%，其係00年00月00日生，迄至110年12月13日發生車禍時，為45歲，經扣除李承祐前所主張不能工作損失之期間後，至自112年2月起算至李承祐65歲強制退休之年齡止，原告可工作之期間應為18年9月又21日；又李承祐於系爭車禍事故時並無工作，是以最低基本工資作為計算之標準，而依照行政院勞動部公布之110年最低基本工資為每月2萬4000元，則李承祐每月之勞動能力減損之金額應為2,880元，依霍夫曼式計算法扣除中間利息（首期給付不扣除中間利息）核計其金額為新臺幣45萬8897元。
查李承祐為高職肄業，目前為自由職業，無固定收入，無存款、有汽機車各1部，無不動產，因系爭車禍事故受有腰椎椎間盤突出之傷害，受有精神上痛苦，爰請求精神慰撫金30萬元。"""

generate_best_lawsheet_among_three(tmp_prompt)

