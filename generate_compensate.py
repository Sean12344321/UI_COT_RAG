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
 精神慰撫金：原告甲於系爭事故發生時年紀尚輕，因系爭事故受有左膝挫傷、擦傷及左手第五掌骨閉鎖性骨折之傷害，經多次回診治療，左手掌需以外物固定而無法彎曲，生活學業受有很大影響，且有永久影響左手握力與日常生活功能之虞。職故，原告確因系爭事故受有精神上痛苦，爰請求被告賠償精神慰撫金15萬元。"""

money_abstract = """規則:
1. 完整列出所有原告及賠償項目，不得遺漏 任何原告 或 任何一項賠償項目。
2. 保持輸入順序，每位原告的賠償項目 順序應與輸入完全一致，不得調換。
3. 金額必須為阿拉伯數字格式（0-9），不得使用中文數字，並以 `XXX元` 格式呈現。
4. 確保輸出格式與範例一致，不得更改格式或調整樣式。
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
money_output_check = """請確保輸入有符合格式:
====================
(一)原告X部分:
1.[賠償項目]:XXX元
賠償原因。
2.[賠償項目]:XXX元
賠償原因。
3.[賠償項目]:XXX元
賠償原因。
(二)原告Y部分:
1.[賠償項目]:XXX元
賠償原因。
2.[賠償項目]:XXX元
賠償原因。
3.[賠償項目]:XXX元
賠償原因。
====================
====================
若符合格式，生成結果請回報「finished」，若不符合格式，請回報「reset」。
"""

summary_output_generate = """根據輸入，生成以下賠償總額，嚴格遵照格式，不要有換行或空格。:
綜上所陳，被告應賠償原告X之損害，包含醫療費用XXX元...，總計XXX元；應賠償原告Y之損害，包含...，總計XXX元。原告合計請求總賠償XXX元。
"""

tmp_output = """(一)原告甲父部分:

1.醫療費用: 53,715元
原告甲父因本次事故受傷，前往部立基隆醫院及基隆長庚紀念醫院治療，支出醫療費用合計53,715元。

2.看護費用: 152,500元 
原告甲父因受有脾臟重度撕裂傷等傷害，需專人照顧3個月。住院期間以每日2,500元計算，休養期則以半日1,250元計算，看護費用合計152,500元。

3.交通費用: 965元
原告甲父因就醫需求，支出部分交通費用，合計965元，其中包括救護車及計程車費用。

4.工作收入損失: 335,070元 
原告甲父因受傷需休養3個月無法工作，造成薪資損失合計335,070元。

5.精神慰撫金: 800,000元
原告甲父因本次事故遭受身心痛苦，請求精神慰撫金800,000元。

6.B車修理及物品損失: 14,227元
原告甲父因本次事故造成B車維修費用支出7,977元，並且身上攜帶的手機、眼鏡、手錶及安全帽等物品受有損害，請求賠償損失6,250元，合計14,227元。

(二)原告甲母部分:
1.精神慰撫金: 50,000元

(三)原告甲部分:
1.精神慰撫金: 150,000元

(三)綜上所陳:
總計各原告的賠償金額為1,656,227元。"""

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
    abstract = combine_prompt_generate_lawsheet(input, money_abstract)
    print(abstract)
    print("=" * 50)
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

# print(generate_compensate(tmp_prompts))