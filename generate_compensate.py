from ollama import chat, ChatResponse
import multiprocessing, re
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
tmp_prompts = """三、請求賠償的事實根據:
原告陳皆宏為治療系爭事故所生之傷勢，前往廣福診所醫院治療，合計醫療費用支出466元。系爭事故後，因系爭車輛經拖吊，當日即有返家之交通費用1,855元支出。原告陳皆宏於治療系爭傷勢期間支出交通費用700元。系爭車輛於修復期間，故而無法使用，故而上下班、快出會議拜訪客戶、接送小孩、數次前往調解、法院之交通費用12,820元。末原告陳皆宏因系爭車輛無法使用，故而有以租賃代步，期間一個月之費用為34,000元。系爭車輛因系爭事故致交易價格減損，經原告向第三方鑑定單位鑑定後，系爭車輛事故發生前價值約950,000元，修復後之價值為840,000元，顯見其市場交易之價值貶損為100,000元。原告陳皆宏因本件系爭事故，前後前往調解至須請假而有工作損失3,000元，嗣因此遭公司非自願離職，為此請求損失5,000元；上合計8,000元。系爭車輛受損後，原告被迫接受無法用車之不便、需另外於住家與公司重新設定、已規劃之行程取消，及額外支出心力安撫親戚家人，故請求精神慰撫金10,000元。而原告王惠滿則為治療系爭事故所生之傷勢，前後前往臺北市立聯合醫院(下稱聯合醫院)、全心中醫診所回診、追蹤等治療，為此之支出醫療費用13,532元。並因受有系爭傷勢，致有後續門診回診之必要，並因此多次前往醫院就診復健，合計迄今原告支出之交通費用為23,525元。又因為系爭腦震盪傷勢致受有相當精神上之痛苦，故請求精神慰撫金72,000元。"""

money_abstract = """規則:
1. 完整列出所有原告及賠償項目，不得遺漏 任何原告 或 任何一項賠償項目。
2. 保持輸入順序，每位原告的賠償項目 順序應與輸入完全一致，不得調換。
3. 金額必須為阿拉伯數字格式（0-9），不得使用中文數字，並以 `XXX元` 格式呈現。
4. 每一項[元]後面及為。或！，不得額外生成補充資訊。
5. 不得新增分層細項（如「臺中榮總: 9,160元」），僅保留主要項目名稱與總金額。
輸出格式:
=======================
(一) 原告X部分:
1. [項目名稱]: XXX元
2. [項目名稱]: XXX元
3. [項目名稱]: XXX元

(二) 原告Y部分:
1. [項目名稱]: XXX元
2. [項目名稱]: XXX元
3. [項目名稱]: XXX元
=======================
"""
money_prompt_multiplePeople = """指示:
請根據輸入的賠償金額資料，生成賠償金額，格式需符合以下要求:

依照原告分類，先寫「(一)原告X部分」，再依序列出各項賠償金額。
逐項列出費用，每項費用需包含:
費用名稱(如「醫療費用」、「薪資損失」、「車損修理費用」等)。
具體金額與簡要原因。
確保每位原告都要有完整的賠償項目，不得遺漏任何一位原告或任何一項賠償項目。
輸出格式範例:
=================
(一)原告X部分:
1.醫療費用:XXX元
原告X因本次事故受傷，前往XXX醫院治療，支出醫療費用XXX元。
2.薪資損失:XXX元
原告X因受傷無法工作，造成薪資損失XXX元。
3. 精神慰撫金: XXX元
原告X因事故遭受精神痛苦，請求精神慰撫金XXX元。
(二)原告Y部分:
1.醫療費用:YYY元
原告Y因本次事故受傷，前往YYY醫院治療，支出醫療費用YYY元。
2.薪資損失:YYY元
原告Y因受傷無法工作，造成薪資損失YYY元。
3. 精神慰撫金: YYY元
原告Y因事故遭受精神痛苦，請求精神慰撫金YYY元。
=================
"""
money_output_check = """請依照以下步驟進行嚴格檢查，並針對每項細節提供判斷與原因：

賠償項目完整性檢查
-不得有未附原因的賠償項目。

輸出格式：
賠償項目檢查: 是/否, 原因: XXX

完整格式檢查
- 輸入的結尾必須是。或！，不得是輸入到一半的段落。  

輸出格式：
格式完整: 是/否, 原因: XXX

最終判決
如果所有檢查項目都為「是」，輸出: 判決結果: finished
如果有任一項為「否」，輸出: 判決結果: reset
"""

summary_output_generate = """根據輸入，生成以下賠償總額，嚴格遵照格式，不要有換行或空格。:
綜上所陳，被告應賠償原告X之損害，包含醫療費用XXX元...，總計XXX元；應賠償原告Y之損害，包含...，總計XXX元。原告合計請求總賠償XXX元。
"""

def remove_last_parenthesis_section(text):
   pattern = r"\(\w+\)(?!.*\(\w+\)).*"  # 找到最後一個 (X) 開頭的段落
   modified_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
   return modified_text

def combine_prompt_generate_lawsheet(input, prompt):
   money_input = f"""輸入:
{input}
{prompt}"""
   return generate_response(money_input)

def generate_compensate(input):
    while True:
        abstract = combine_prompt_generate_lawsheet(input, money_abstract)
        print(abstract)
        print("=" * 50)
        if '-' and '*' not in abstract:
            break
    while True:
        global money_prompt_multiplePeople
        money_prompt_multiplePeople += f"""\n請參考以下賠償項目進行填充:{abstract}"""
        output = combine_prompt_generate_lawsheet(input, money_prompt_multiplePeople)
        output = output.replace('*', '')
        print(output)
        print("=" * 50)
        judge_money = combine_prompt_generate_lawsheet(output, money_output_check)
        print(judge_money)
        print("=" * 50)
        if "reset" in judge_money:
            continue
        if "finished" in judge_money:
            break
    output = re.sub(r'(^=+|\n=+)', '', output) # remove all ===== signal
    summary_output = combine_prompt_generate_lawsheet(output, summary_output_generate)
    summary_output = re.sub(r'\n', '', summary_output) # remove all \n signals
    return output + '\n\n' + summary_output.replace('*', '') 

print(generate_compensate(tmp_prompts))