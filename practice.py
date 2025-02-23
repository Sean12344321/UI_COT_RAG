from evaluate import load
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
            model='deepseek-r1:32b',
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"
tmp_prompts = """一、事故發生緣由:
 被告於民國000年0月00日下午5時許，駕駛車牌號碼0000-00號自用小客車（下稱A車），自基隆市中正區碧砂漁港駛出欲右轉往中正路方向行駛之際，於該址路口（下稱系爭地點）本應注意行駛在閃紅燈之支線道車應暫停讓幹線道車先行，以避免危險或交通事故之發生，而依當時天候及路況，並無不能注意之情事，竟疏未注意上情而貿然前行，此時原告甲父騎乘車牌號碼000-000號普通重型機車（下稱B車）附載原告甲及甲母行駛至系爭地點，雙方因而發生碰撞（下稱系爭事故），原告甲、甲父、甲母因此分別受有身體傷害及財物損失。
 
 二、原告受傷情形:
 原告甲父因系爭事故受有脾臟重度撕裂傷合併腹內出血及休克、左側第6至第9肋骨骨折、左小腿及左腳踝大面積傷口、左側肺挫傷併少量血胸等傷害。
 原告甲母因系爭事故受有四肢及臉部多處挫傷及擦傷。
 原告甲因系爭事故受有左膝挫傷、擦傷及左手第五掌骨閉鎖性骨折之傷害。
 
 三、請求賠償的事實根據:
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

money_prompt_multiplePeople = """指示:
請根據輸入的賠償金額資料，生成賠償金額，格式需符合以下要求:

依照原告分類，先寫「(一)原告X部分」，再依序列出各項賠償金額。
逐項列出費用，每項費用需包含:
費用名稱(如「醫療費用」、「薪資損失」、「車損修理費用」等)。
具體金額與簡要原因。
綜合計算，最後加總每位原告的請求金額，並統整最終總賠償金額。
輸出格式範例:

(一)原告X部分:
1.醫療費用:XXX元
原告X因本次事故受傷，前往XXX醫院治療，支出醫療費用XXX元。

2.車損修理費用:XXX元
原告X所有之系爭車輛因本次事故受損，支出修復費用XXX元。

3.薪資損失:XXX元
原告X因本次事故受傷無法工作，遭受薪資損失XXX元。

4.慰撫金:XXX元
原告X於事故傷害，造成原告精神上極大痛苦，爰請求慰撫金XXX元。

(二)原告Y部分:
(依照相同格式列出)

(三)綜上所陳:
被告應賠償原告X之損害，包含醫療費用XXX元...，總計XXX元；應賠償原告Y之損害，包含醫療費用...，總計XXX元。兩原告合計請求賠償XXX元。

嚴格按照此格式，確保邏輯清晰、條理分明。
"""
money_output_check="""
請一項一項仔細檢查輸入內容，確保以下事項:
若金額沒有明確數字，請生成「reset」。
若列出金額為0元，請生成「reset」。
若有出現意義不明的符號，像是'plaintiff'，請生成「reset」。
若無上述問題，請生成「finished」，以生成最終賠償金額。
"""
summary_output_generate = """
輸入的金額可能算錯，所以請一項一項把金額加總起來，並將正確的金額用下面風格生成:
綜上所陳，被告應賠償原告X之損害，包含...，總計XXX元；應賠償原告Y之損害，包含...，總計XXX元。兩原告合計請求賠償XXX元。
"""
tmp_output = """
(一)原告甲父部分:
1.醫療費用:53,715元
原告甲父因本次事故受傷，前往醫院治療，支出醫療費用53,715元。

2.看護費用:152,500元
原告甲父因本次事故需要專業看護，支出看護費用152,500元。

3.交通費用:965元
原告甲父因本次事故無法自行行動，支出交通費用965元。

4.薪資損失:335,070元
原告甲父因本次事故受傷無法工作，遭受薪資損失335,070元。

5.慰撫金:800,000元
原告甲父於事故中受傷，造成精神上極大痛苦，爰請求慰撫金800,000元。

6.車損及物品損失:14,227元
原告甲父所有之系爭車輛因本次事故受損，支出修復費用7,977元；並因本次事故造成手機、眼鏡、手錶、安全帽及衣服破損，受有價值相當於6,250元之損害，合計14,227元。

(二)原告甲母部分:
1.慰撫金:50,000元
原告甲母因本次事故受有四肢及臉部多處挫傷及擦傷，造成精神上痛苦，爰請求慰撫金50,000元。

(三)原告甲部分:
1.慰撫金:150,000元
原告甲於系爭事故中受有左膝挫傷、擦傷及左手第五掌骨閉鎖性骨折之傷害，造成生活學業影響及精神上痛苦，爰請求慰撫金150,000元。

(四)綜上所陳:
被告應賠償原告甲父之損害，包含醫療費用53,715元、看護費用152,500元、交通費用965元、薪資損失335,070元、慰撫金800,000元及車損及物品損失14,227元，總計582,714元；應賠償原告甲母之損害，包含慰撫金50,000元，總計50,000元；應賠償原告甲之損害，包含慰撫金150,000元。三原告合計請求賠償782,714元。"""



def remove_think_tags(text):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()  

def remove_last_parenthesis_section(text):
   pattern = r"\(\w+\)(?!.*\(\w+\)).*"  # 找到最後一個 (X) 開頭的段落
   modified_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
   return modified_text

def combine_prompt_generate_lawsheet(input, prompt):
   money_input = f"""輸入:
{input}
{prompt}"""
   return generate_response(money_input)

def generate_lawsheet(input):
   output = remove_think_tags(combine_prompt_generate_lawsheet(input, money_prompt_multiplePeople))
   print(output)
   print("=" * 50)
   while True:
      judge_money = combine_prompt_generate_lawsheet(output, money_output_check)
      print(judge_money)
      print("=" * 50)
      if "finished" in remove_think_tags(judge_money):
         break
      else: #reset or others
         output = remove_think_tags(combine_prompt_generate_lawsheet(input, money_prompt_multiplePeople))
      print(output)
      print("=" * 50)
   summary_output = combine_prompt_generate_lawsheet(output, summary_output_generate)
   print(summary_output)
   print("=" * 50)
   return remove_last_parenthesis_section(output) + '\n\n' + remove_think_tags(summary_output) 


def generate_best_lawsheet_among_three(input):
   inputs = [input] * 2
   with multiprocessing.Pool(processes=1) as pool:
      # 使用 map 同時對3個輸入進行處理
      lawsuits = pool.map(generate_lawsheet, inputs)
      
# generate_best_lawsheet_among_three(tmp_prompts)
print(generate_lawsheet(tmp_prompts))