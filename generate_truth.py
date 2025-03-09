from ollama import chat, ChatResponse
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

# 事實陳述摘要規則
truth_abstract = """規則:
1. 依據輸入的事實描述，生成結構清晰的事故摘要，完整保留所有關鍵資訊，不得遺漏。
2. 僅使用輸入中明確提供的資訊，**不得推測或補充未出現在輸入中的內容**（例如：刑事判決、天候條件、路況等）。
3. 以簡潔扼要的方式陳述內容，避免冗長敘述，確保資訊清楚易讀。
4. 若某個資訊缺失，則不輸出該項目，填入「無」或「不詳」**。

輸出格式：
=======================
[事故緣由]: [內容]
[當天環境]: [內容]
[傷勢情形]: [內容]
=======================
嚴格遵照上述規則，根據輸入資訊生成事故摘要。
"""

# 事實陳述生成提示詞（完整段落，並以「緣」開頭）
truth_prompt = """
嚴格遵照範例格式，根據輸入資訊生成事實陳述。
格式:
緣被告於民國 年 月 日，無照騎駛車牌號碼000-0000號普通重型機車，沿新北市三重區中正北路往蘆洲方向行駛，行經中正北路及中正北路000巷之交岔路口時，本應注意機車行駛時，駕駛人應注意車前狀況，並隨時採取必要之安全措施，而依當時天候晴，夜間有照明，柏油路面乾燥、無缺陷、無障礙物，視距良好，並無不能注意之情事，被告竟疏未注意及此，即貿然前行，適前方有原告林肜宇騎駛車牌號碼000-000號普通重型機車並搭載原告吳彩雲，沿同方向行駛，亦行經該處，雙方因而發生碰撞，致原告均人車倒地。
"""
# 事實陳述檢查提示詞
truth_output_check = """請依照以下步驟進行嚴格檢查，並針對每項細節提供判斷與原因：

客觀性檢查
-不能使用任何主觀評語，像是「我們」。

輸出格式：
客觀性: 是/否, 原因: XXX

內容範圍檢查
- 事實陳述部分不能涉及任何民法或賠償金額，僅限於陳述事實。  

輸出格式：
內容範圍: 是/否, 原因: XXX

最終判決
如果所有檢查項目都為「是」，輸出: 判決結果: finished
如果有任一項為「否」，輸出: 判決結果: reset
"""
tmp_prompt = """一、事故發生緣由:
被告於民國000年0月00日下午5時許，駕駛車牌號碼0000-00號自用小客車（下稱A車），自基隆市中正區碧砂漁港駛出欲右轉往中正路方向行駛之際，於該址路口（下稱系爭地點）本應注意行駛在閃紅燈之支線道車應暫停讓幹線道車先行，以避免危險或交通事故之發生，而依當時天候及路況，並無不能注意之情事，竟疏未注意上情而貿然前行，此時原告甲父騎乘車牌號碼000-000號普通重型機車（下稱B車）附載原告甲及甲母行駛至系爭地點，雙方因而發生碰撞（下稱系爭事故），原告甲、甲父、甲母因此分別受有身體傷害及財物損失。

二、原告受傷情形:
原告甲父因系爭事故受有脾臟重度撕裂傷合併腹內出血及休克、左側第6至第9肋骨骨折、左小腿及左腳踝大面積傷口、左側肺挫傷併少量血胸等傷害。
原告甲母因系爭事故受有四肢及臉部多處挫傷及擦傷。
原告甲因系爭事故受有左膝挫傷、擦傷及左手第五掌骨閉鎖性骨折之傷害。
"""
def combine_prompt_generate_fact(input, prompt):
    fact_input = f"""輸入:
{input}
{prompt}"""
    return generate_response(fact_input) 
def generate_fact_statement(input):
    abstract = combine_prompt_generate_fact(input, truth_abstract)
    print(abstract)
    print("=" * 50)

    while True:
        global truth_prompt
        truth_prompt += f"""\n參考事實摘要形成事實陳述{abstract}"""
        output = combine_prompt_generate_fact(input, truth_prompt)
        output = output.replace('*', '')
        output.replace("\n", "").replace("\r", "")
        print(output)
        print("=" * 50)
        judge_fact = combine_prompt_generate_fact(output, truth_output_check)
        print(judge_fact)
        print("=" * 50)
        if "reset" in judge_fact:
            continue
        if "finished" in judge_fact:
            break
    return "一、" + output