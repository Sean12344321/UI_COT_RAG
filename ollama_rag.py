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
truth_prompt = """
請根據提供的交通事故基本資訊，生成一段法律起訴狀中「事實陳述」的完整段落，需符合正式法律文書格式，並包含以下要素:

1. 時間地點 - 事故發生的具體時間(民國年月日、時刻)與地點(城市、區域、道路名稱等)。
2. 當事人 - 涉事車輛駕駛人、乘客(如有)及受害者的身分描述(如「原告甲」)。
3. 車輛資訊 - 各方車輛之類型(小客車、機車等)與車牌號碼。
4. 事故過程 - 涉事駕駛之具體違規行為(如未遵守號誌、不讓幹道車先行等)。
5. 後果 - 事故造成的影響，包括人員受傷、財物損壞等。
輸出格式要求:
需使用緣開頭

請依照上述要求，根據具體事故資訊生成完整事實陳述段落。
"""

truth_score_prompt = """請根據以下三個版本的起訴狀內容，擷取最佳部分，生成最終版本。請確保:

1. 輸出為一整個段落，不要有換行，讓內容更加連貫。
2. 句子結構清晰、流暢，避免冗長或重複。
3. 事故敘述完整，包括事故過失、原告受傷情形。
4. 避免幻覺，若只有1個版本有提出某項事實，則該事實應被丟棄。
提供的三個版本如下:"""

money_score_prompt = """請根據賠償金額段落的完整性、合理性與法律適用性，評分該段落的品質，並輸出分數。
評分標準(總分100分):
- 若出現「XX元」等模糊表述，扣100分。  
- 是否有「綜上所陳」或其他合理總結句?若無，扣100分。
- 是否已。作結尾? 若無，扣100分。

### 輸出格式:
評語:是否出現模糊表述?是否有總結句?是否有結尾?
分數:XX分"""

# 單位原告
money_prompt_singlePerson = """請依照以下格式撰寫，輸出必須分為兩個部分:
1. 逐一列出原告的費用，將費用依功能進行歸類，例如:「醫療花費」、「精神慰撫金」、「工作損失」等，而非列出過於詳細的細項(如具體診所名稱)。
2. 最後統一撰寫「綜上所陳」段落，總結這位原告的損害賠償金額。
請注意:
- 僅列出金額大於 0 的項目，若某項費用為 0 元，請忽略該項目，不需顯示。

範例如下:
(一) {費用項目1}費用:台幣{費用金額1}元
   原告因{事故原因1}導致{後果或需要醫療的情況1}，產生相關費用或損失，共計{費用金額1}元。
(二) {費用項目2}費用:台幣{費用金額2}元
   原告因{事故原因2}導致{後果或需要醫療的情況2}，產生相關費用或損失，共計{費用金額2}元。
(三) 綜上所陳，應連帶賠償原告{受害者名稱1}之損害，包含{費用項目1}費用{費用金額1}元，{費用項目2}費用{費用金額2}元，共計{總金額}元；應連帶賠償原告{受害者名稱2}之損害，包含{費用項目1}費用{費用金額1}元，共計{總金額}元。
"""
#多位原告
money_prompt_multiplePeople = """指示:
請根據輸入的賠償金額資料，生成賠償金額，格式需符合以下要求:

依照原告分類，先寫「(一)原告 X 部分」，再依序列出各項賠償金額。
逐項列出費用，每項費用需包含:
費用名稱(如「醫療費用」、「薪資損失」、「車損修理費用」等)。
具體金額與簡要原因。
綜合計算，最後加總每位原告的請求金額，並統整最終總賠償金額。
輸出格式範例:

(一)原告 X 部分:
1.醫療費用:XXX 元
原告 X 因本次事故受傷，前往 XXX 醫院治療，支出醫療費用 XXX 元。

2.車損修理費用:XXX 元
原告 X 所有之系爭車輛因本次事故受損，支出修復費用 XXX 元。

3.薪資損失:XXX 元
原告 X 因本次事故受傷無法工作，遭受薪資損失 XXX 元。

4.慰撫金:XXX 元
原告 X 於事故傷害，造成原告精神上極大痛苦，爰請求慰撫金 XXX 元。

(二)原告 Y 部分:
(依照相同格式列出)

(三)綜上所陳，被告應賠償原告 X 之損害，包含 醫療費用 XXX 元、車損修理費用 XXX 元及慰撫金 XXX 元，總計 XXX 元；應賠償原告 Y 之損害，包含 醫療費用 XXX 元、薪資損失 XXX 元及慰撫金 XXX 元，總計 XXX 元。兩原告合計請求賠償 XXX 元。

請嚴格按照此格式，確保邏輯清晰、條理分明。
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
    truth_input = f"""輸入:
{data["case_facts"]}
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
    print("=" * 50)
    print(money_output)
    print("=" * 50)
    numbers = re.findall(r"\d+", money_output)
    return int(numbers[-1])
def generate_best_lawsheet_among_three(input):
   inputs = [input] * 3
   with multiprocessing.Pool(processes=1) as pool:
      # 使用 map 同時對3個輸入進行處理
      lawsuits = pool.map(generate_lawsheet, inputs)
      final_truth = f"""{truth_score_prompt}
1. {lawsuits[0]['truth']}
2. {lawsuits[1]['truth']}
3. {lawsuits[2]['truth']}"""
      lawsuits[0]['truth'] = generate_response(final_truth)
      print(final_truth)
      money_score = pool.map(generate_money_score, [lawsuits[0]['money'], lawsuits[1]['money'], lawsuits[2]['money']])
      best_lawsheet_index = money_score.index(max(money_score))
      print(money_score)
      return lawsuits[0]['truth'] + lawsuits[0]['law'] + lawsuits[best_lawsheet_index]['money']
    
tmp_prompt = """一、事故發生緣由:
被告於民國111年8月4日23時17分許，無照騎駛車牌號碼000-0000號普通重型機車，沿新北市三重區中正北路往蘆洲方向行駛，行經中正北路及中正北路312巷之交岔路口時，本應注意機車行駛時，駕駛人應注意車前狀況，並隨時採取必要之安全措施，而依當時天候晴，夜間有照明，柏油路面乾燥、無缺陷、無障礙物，視距良好，並無不能注意之情事，被告竟疏未注意及此，即貿然前行，適前方有原告林肜宇騎駛車牌號碼000-000號普通重型機車（下稱系爭機車）並搭載原告吳彩雲，沿同方向行駛，亦行經該處，雙方因而發生碰撞，致原告均人車倒地。而被告已經鈞院113年度審交簡字第50號刑事簡易判決判處拘役40日有罪，足證被告針對本件事故之發生具有過失。

二、原告受傷情形:
本件事故使原告林肜宇受有頭部、左側手肘、膝部、足部、雙側手部挫擦傷、左側第六肋骨及肩胛骨閉肩胛性骨折等傷害（下稱系爭傷害），並造成系爭機車受損。
另外原告吳彩雲則受有左側肩膀、手部、髖部及足部挫擦傷等傷害（下稱系爭傷勢）。

三、請求賠償的事實根據:
原告林肜宇主張因治療系爭傷害，支出醫療暨醫療耗材費用共2萬0185元，並提出新北市立聯合醫院急診醫療費用收據、固的診所自費醫療明細收據暨藥品明細收據、臺北榮民總醫院門診醫療費用明細收據暨急診醫療費用明細收據等作為證據。
並且根據臺北榮民總醫院111年10月28日診斷證明書記載：「病人因上述病症於111年8月5日、8月10日、8月24日、9月7日、9月28日、10月28日至本院門診，傷後需專人照顧兩個月及宜繼續門診治療追蹤複查。」顯示其需專人看護2個月，以每日看護2200元計算，費用為13萬2000元。
又因系爭傷害有2個月不能工作，事故發生前每月薪資為16萬2000元，故請求被告賠償其休養期間不能工作之損失為32萬4000元。
且因本件車禍足部受傷，回診必須搭乘計程車，已支付就醫交通費用1335元，並提出計程車乘車為證。
系爭機車受損修復費用一共7970元(均為零件)，有必榮機車材料行開立之估價單當作證據。
其因系爭傷害需要定期就醫回診，生活受到相當程度影響，受有相當之精神痛苦，故請求精神慰撫金20萬元。

原告吳彩雲主張因治療系爭傷勢，支出醫療費用3200元，並有新北市立聯合醫院急診醫療費用收據及臺北榮民總醫院門診醫療費用明細收據等作為證據。
其主張本件車禍發生前，任職於台鼎有限公司，每月收入薪資為3萬元，因系爭傷勢需休養3日無法工作，故向公司請假3日，以每月薪資3萬元計算受有薪資損失3000元，並有銀行存摺明細乙份作為證據。
其因系爭傷勢同樣需要就醫回診，承受身體不適及工作上受到相當程度影響，受有相當之精神痛苦，故請求精神慰撫金5萬元。"""

print(generate_best_lawsheet_among_three(tmp_prompt))

