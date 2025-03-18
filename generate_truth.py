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
2. 僅使用輸入中明確提供的資訊，不得推測或補充未出現在輸入中的內容（例如：刑事判決、天候條件、路況等）。
3. 以簡潔扼要的方式陳述內容，避免冗長敘述，確保資訊清楚易讀。
4. 若某個資訊缺失，則不輸出該項目，填入「無」或「不詳」。

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
嚴格遵照範例格式生成事實陳述，不要有任何列點或是換行，並以「緣」開頭。
範例格式:
"""
# 事實陳述檢查提示詞
truth_output_check = """請依照以下步驟進行嚴格檢查，並針對每項細節提供判斷與原因：

正確性檢查
-輸入跟摘要內容是否一致，如事故緣由，受傷情形等資訊，並且沒有遺漏任何重要資訊。

輸出格式：
正確性: 是/否, 原因: XXX

內容範圍檢查
- 事實陳述部分不能涉及任何民法或賠償金額，僅限於陳述事實以及原告受傷情形。  

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
tmp_reference = """一、緣被告於民國110年5月8日14時7分許，駕駛車牌號碼000-0000號自用小客車（下稱A車），沿屏東縣東港鎮中正路1段由東往西方向行駛，行經該路段246號前時，本應注意在同一車道行駛時，與前車之間應保持隨時可以煞停之距離，且應注意車前狀況，並隨時採取必要之安全措施。而依當時天候晴、日間自然光線、柏油路面乾燥、無缺陷、無障礙物、視距良好等並無不能注意之情事，竟疏未注意即貿然直行，致追撞前方B車後車尾，造成連環追撞，導致原告所駕駛之系爭車輛受損，且原告因而受有顏面左側挫傷合併0.5公分撕裂傷、併腦震盪症狀等傷害。"""

def combine_prompt_generate_fact(input, prompt):
    fact_input = f"""輸入:
{input}
{prompt}"""
    return generate_response(fact_input) 

def generate_fact_statement(input, reference_fact):
    global truth_prompt, truth_output_check
    abstract = combine_prompt_generate_fact(input, truth_abstract)
    print(abstract)
    print("=" * 50)
    truth_output_check = f"\n摘要:{abstract}" + truth_output_check
    truth_prompt += reference_fact
    while True:
        output = combine_prompt_generate_fact(input, truth_prompt)
        output = output.replace('*', '').replace("\n", "").replace("\r", "")
        print(output)
        print("=" * 50)
        judge_fact = combine_prompt_generate_fact(output, truth_output_check)
        print(judge_fact)
        print("=" * 50)
        if "reset" in judge_fact or "二、" in output:
            continue
        if "finished" in judge_fact:
            break
    if "一、" not in output:
        output = "一、" + output
    return output
