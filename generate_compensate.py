from ollama import chat, ChatResponse
from utils import Tools
from KG_RAG_B.define_case_type import get_case_type
from chunk_RAG.ts_define_case_type import get_case_type as chunk_get_case_type
from chunk_RAG.ts_prompt import *
from evaluate import load
import re, time
user_input = """
一、事故發生緣由:
 被告於民國105年4月12日13時27分許，駕駛租賃小客車沿新北市某區某路往富國路方向行駛。行經福營路342號前時，被告跨越分向限制線欲繞越前方由原告所駕駛併排於路邊臨時停車後適欲起駛之車輛。被告為閃避對向來車，因而駕車自後追撞原告駕駛車輛左後車尾。當時天候晴朗、日間自然光線、柏油道路乾燥無缺陷或障礙物、視距良好，被告理應注意車前狀況及兩車並行之間隔，隨時採取必要之安全措施，但卻疏未注意而發生事故。
 
 二、原告受傷情形:
 原告因此車禍受有左膝挫傷、半月軟骨受傷等傷害。原告於105年5月2日、7日、7月16日、8月13日、8月29日至醫院門診就診，105年8月2日進行核磁共振造影檢查。根據醫院開立的診斷證明書，原告需休養1個月。
 
 三、請求賠償的事實根據:
 1. 醫療復健費用190元
 2. 車輛修復費用181,144元
 3. 交通費用4,500元
 4. 休養期間工作收入損失33,000元
 5. 慰撫金99,000元
 
 原告因傷不良於行，上下班須搭乘計程車，支出交通費用。原告的車輛因事故受損需要修理，修理費用包括工資費用88,774元和零件費用92,370元。原告需休養1個月無法工作，造成工作收入損失。
 
 原告學歷為高職畢業，目前從事打零工，日薪約900元，105年度所得124,719元，名下有動產。原告因本次事故受傷，不僅身體受損，還需經歷刑事偵審及民事訴訟過程，耗費大量時間和精神，因此請求精神慰撫金。
 
 綜上所述，原告依據民法第184條第1項前段、第191條之2、第193條第1項及第195條第1項前段之規定，請求被告賠償上述損害，總計317,834元及自起訴狀繕本送達翌日起至清償日止，按年息5%計算之利息。
"""
injuries = """  二、原告受傷情形:
 原告因此車禍受有左膝挫傷、半月軟骨受傷等傷害。原告於105年5月2日、7日、7月16日、8月13日、8月29日至醫院門診就診，105年8月2日進行核磁共振造影檢查。根據醫院開立的診斷證明書，原告需休養1個月。"""
compensation_facts = """三、請求賠償的事實根據:
 1. 醫療復健費用190元
 2. 車輛修復費用181,144元
 3. 交通費用4,500元
 4. 休養期間工作收入損失33,000元
 5. 慰撫金99,000元
 
 原告因傷不良於行，上下班須搭乘計程車，支出交通費用。原告的車輛因事故受損需要修理，修理費用包括工資費用88,774元和零件費用92,370元。原告需休養1個月無法工作，造成工作收入損失。
 
 原告學歷為高職畢業，目前從事打零工，日薪約900元，105年度所得124,719元，名下有動產。原告因本次事故受傷，不僅身體受損，還需經歷刑事偵審及民事訴訟過程，耗費大量時間和精神，因此請求精神慰撫金。
 
 綜上所述，原告依據民法第184條第1項前段、第191條之2、第193條第1項及第195條第1項前段之規定，請求被告賠償上述損害，總計317,834元及自起訴狀繕本送達翌日起至清償日止，按年息5%計算之利息。"""
references = ["""（一）醫療費用：146,363元
1. 原告莊士紘醫療費用：800元
2. 原告林淑玲醫療費用：145,563元

（二）交通費用：6,105元
原告林淑玲因本件車禍受有右側遠端股骨粉碎性骨折之傷害，需搭乘他人駕駛車輛接送往返醫院就醫，支出交通費用6,105元。

（三）看護費用：66,000元
原告林淑玲因本件車禍受有右側遠端股骨粉碎性骨折之傷害，需他人看護照顧一個月，以每日2,200元計算，共計66,000元。

（四）財物損害：454,869元
1. 系爭汽車修理費：438,069元
2. 原告莊士紘眼鏡毀損：6,800元

（五）工作損失：77,600元
1. 原告莊士紘工作損失：5,600元
2. 原告林淑玲工作損失：72,000元

（六）慰撫金：180,000元
1. 原告莊士紘：30,000元
2. 原告林淑玲：150,000元

（七）綜上所陳，被告應連帶賠償原告之損害，包含醫療費用146,363元、交通費用6,105元、看護費用66,000元、財物損害454,869元、工作損失77,600元及慰撫金180,000元，總計930,937元。
"""]
single_money_summary_prompt = """你是一個格式輸出助手，任務是將原告的賠償項目與金額，依照指定格式列出，完全依據輸入，不得遺漏、改寫、推論、補充。
【請務必遵守以下規則】：
1. 必須完整列出所有原告及其所有賠償項目，不得遺漏任何原告或任何一項賠償項目。
2. 保持輸入順序，不可顛倒。每位原告的賠償項目順序應與輸入一致。
3. 金額必須為阿拉伯數字格式（0-9），並以 `XXX元` 呈現。
4. 禁止出現加總過程、等號（=）、加號（+）或總金額文字，僅列出每筆明細金額。
5. 每一筆賠償項目名稱必須簡潔明確，不可包含日期或敘述性說明
6. 遵守格式規範，開頭使用「（一）」、「（二）」、「（三）」等標記，並以「元」結尾。

【輸出格式範例如下】：
（一）醫療費: 9160元
（二）交通費: 800元
（三）精神撫慰金: 50000元

【請根據使用者輸入內容，輸出相同格式的結果】：
"""
multiple_money_summary_prompt = """你是一個格式輸出助手，任務是將原告的賠償項目與金額，依照指定格式列出，完全依據輸入，不得遺漏、改寫、推論、補充。
【請務必遵守以下規則】：
1. 必須完整列出所有原告及其所有賠償項目，不得遺漏任何原告或任何一項賠償項目。
2. 每一個賠償項目皆需直接列出，並於項目後加上該原告姓名（例如：「醫療費用（原告甲部分）」），不得以子項或縮排方式呈現。
3. 保持輸入順序，不可顛倒。每位原告的賠償項目順序應與輸入一致。
4. 金額必須為阿拉伯數字格式（0-9），並以 `XXX元` 呈現。
5. 禁止出現加總過程、等號（=）、加號（+）或總金額文字，僅列出每筆明細金額。
6. 每一筆賠償項目名稱必須簡潔明確，不可包含日期或敘述性說明
7. 遵守格式規範，開頭使用「（一）」、「（二）」、「（三）」等標記，並以「元」結尾。

【輸出格式範例如下】：
（一）醫療費用（原告甲部分）: 1036元
（二）車損修理費用（原告甲部分）: 413300元
（三）醫療費用（原告乙部分）: 4862元
（四）薪資損失（原告乙部分）: 3225元

【請根據使用者輸入內容，輸出相同格式的結果】：
"""
amount_summary = """請根據輸入資訊擷取關鍵資訊，並遵循以下格式輸出：
============================
[賠償項目]: 
[賠償金額]: 
============================
請注意以下事項：
- 若有金額，請去除逗號並以阿拉伯數字表示。
- 如果金額或項目中包含 o 或 0 字眼，請照原樣輸出，不要擅自修改。
"""
final_amount_summary = """請根據輸入資訊擷取關鍵資訊，並遵循以下格式輸出：
============================
[總賠償金額]: 
============================
請注意以下事項：
- 若有金額，請去除逗號並以阿拉伯數字表示。
- 如果金額或項目中包含 o 或 0 字眼，請照原樣輸出，不要擅自修改。
"""
compensate_prompt = f"""
你是一位熟悉法律文書格式的語言模型。請從`輸入`中讀取內容，並以`生成格式`作為開頭，接續撰寫一段描述句，說明該筆費用的原因與金額使用情況。
⚠️請特別注意：
1. 你應該 **僅生成與 `生成格式` 所示金額相符的那一筆費用描述**。如果 `輸入` 中包含多筆金額或多位人員，請勿引用其他不相關的金額。
2. 輸出應包含 `生成格式` 開頭的標題行與一段敘述，兩者缺一不可。
3. 金額格式須為「#,###元」，務必與 `生成格式` 數字一致。
4. 第二行應僅為事實陳述，包括傷勢、用途、支出情形、金額等，不得包含任何法律條文、條號、責任、請求、結語或主觀評價，例如「被告應賠償」、「依據民法」等字句皆不得出現。
5. 請以「（」開頭。"""
item_compensate_check = """              
你是一位法律資訊助理，負責判斷兩個賠償項目是否可以視為語意與法律歸類上完全一致。請依照以下步驟進行判斷：

步驟一：判斷兩個項目是否屬於同一「法律分類」（例如：醫療支出、工作損失、財產損害、精神慰撫等）。若分類不同，直接輸出 reject。

步驟二：若分類相同，繼續比對兩者所涉及的實體是否一致（例如：「誤工損失」和「看護費」雖皆屬工作損失，但性質不同，仍輸出 reject）。

步驟三：若名稱不同但語意一致（如「醫療費」與「就醫費用」），並且法律分類與實體性質皆相同，則可輸出 accept。

步驟四：若兩項描述無明顯連結，或僅模糊表示為「損失」、「支出」等抽象詞，則輸出 reject。

⚠️ 請以保守原則為準，寧可誤殺也不要誤放。

請依照以下格式輸出：
===========================
[推理過程]:
法律分類：
實體性質是否一致：
語意判斷與對應分析：

[判決結果]:
若有任何一項不一致，則輸出「reject」；若分類、性質與語意皆一致，才輸出「accept」。
===========================

"""
money_to_number_prompt = """
你是一位專業的資訊處理助理，負責將輸入的賠償金額轉換為純阿拉伯數字（不含逗號、不含單位）。

請務必依照下列格式與規則作答：

【轉換規則】：
1. 若金額中有千分位逗號（,），請移除。
2. 若金額中含有單位（如「元」、「新台幣」、「NTD」），請去除單位。
3. 若金額包含中文數字或「萬」、「千」等中文單位，請進行正確的數字換算（例如：「42萬三千」→ 420000 + 3000 = 423000）。
4. 請完整展開換算過程以利檢查。
5. 請僅在最終答案欄輸出阿拉伯數字，不要加上單位或文字說明。
6. 若找不到任何有效金額，請僅輸出「0」。

請依照下列格式輸出：

===========================
[推理過程]:
1. 原始金額表達為：_____
2. 去除單位與千分位符號後為：_____
3. 最終轉換結果為純阿拉伯數字。

[最終答案]:
僅輸出轉換後的阿拉伯數字，不含逗號與單位。若無金額資訊，請輸出「0」。
===========================
"""
compensation_sum_prompt = """請參考以下「範例格式」，將給定的各筆損害賠償項目重新整理成總結句，格式須一致："""
labels = [
    '（一）', '（二）', '（三）', '（四）', '（五）', '（六）', '（七）', '（八）', '（九）', '（十）',
    '（十一）', '（十二）', '（十三）', '（十四）', '（十五）', '（十六）', '（十七）', '（十八）', '（十九）', '（二十）',
    '（二十一）', '（二十二）', '（二十三）', '（二十四）', '（二十五）', '（二十六）', '（二十七）', '（二十八）', '（二十九）', '（三十）'
]
tools = None
def get_exact_amount(money_content):
    numbers = re.findall(r'\d+', money_content)
    if len(numbers) == 0:
        return False
    return numbers[-1]
def check_and_generate_summary_items(text):
    text_array = []  # 去除多餘的空格和換行
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        line = line.rstrip()
        if len(line) >= 4 and line[0] == '（' and (line[2] == '）' or line[3] == '）') and '+' not in line and '=' not in line and line[-1] == '元':
            match = re.search(r'(\d+(,\d{3})*|\d+)元$', line)
            if match and match.group(0).replace(',', '') == '0元':
                print("money should not be 0")
                # yield tools.show_debug_to_UI("金額不應為0")
                return False
            match = re.search(r'（(.*?)）', line)
            if match.group(0) not in labels: # 括號裡面不是中文字
                print("（）should contain chinese")
                # yield tools.show_debug_to_UI("括號內應為中文")
                return False
            line = line[:match.start()] + labels[len(text_array)] + line[match.end():]
            line = re.sub(r'[。\.]', '', line)
            text_array.append(line)
        elif line != "":
            print("format error")
            # yield tools.show_debug_to_UI("格式錯誤")
            return False
    return text_array

def generate_summary(input, final = False):
    info_dict = {}
    time = 0
    if final == True:
        while time < 5:
            time += 1
            abstract = tools.combine_prompt_generate_response(input, final_amount_summary).replace("=", "")
            matches = re.findall(r"\[(總賠償金額)\]:\s*(.*)", abstract)
            # 轉換為字典
            info_dict = {k: v for k, v in matches}
            # 如果資訊不構成二個欄位則重新生成
            if len(info_dict) == 1:
                break
        if time >= 5:
            return False
    else:
        while time < 5:#生成至多5次
            time += 1
            abstract = tools.combine_prompt_generate_response(input, amount_summary).replace("=", "")
            matches = re.findall(r"\[(賠償項目|賠償金額)\]:\s*(.*)", abstract)
            # 轉換為字典
            info_dict = {k: v for k, v in matches}
            # 如果資訊不構成二個欄位則重新生成
            if len(info_dict) == 2:
                break
        if time >= 5:
            return False
    return info_dict

def generate_total_summary(input_text):
    """1. 取得原告姓名，判斷是單名還是多名原告來改變提示詞"""
    case_type = get_case_type(input_text)
    print(f"案件類型: {case_type}")
    # yield tools.show_debug_to_UI(f"案件類型: {case_type}")
    if case_type == "單純原被告各一" or case_type == "數名被告":
        prompt = single_money_summary_prompt
    else:
        prompt = multiple_money_summary_prompt
    """2. 取得賠償摘要"""
    summary = tools.combine_prompt_generate_response(input_text, prompt)
    judge = check_and_generate_summary_items(summary)
    while(judge == False):
        print(summary)
        print("格式錯誤，重新生成")
        print("=" * 50)
        # yield tools.show_debug_to_UI(f"{summary}\n格式錯誤，重新生成\n" + "=" * 50)
        summary = tools.combine_prompt_generate_response(input_text, prompt)
        judge = check_and_generate_summary_items(summary)
    print(summary, '\n', "=" * 50)
    # yield tools.show_debug_to_UI(f"賠償摘要:\n{summary}\n" + "=" * 50)
    return summary

    
def generate_reference_array(references):
    parts = re.split(r'（[一二三四五六七八九十]{1,3}）', references)[1:]
    results = []
    for i, part in enumerate(parts):
        lines = part.strip().splitlines()
        if (
            not lines[0].strip().endswith("元")
            or (
                part.count('\n') > 10
                and any(x in part for x in ['1.', '2.', '3.'])
            )
        ) and i != len(parts) - 1:
            lines = lines[1:]
            combined = "\n".join(lines)
            sections = re.split(r'\n(?=\d+\.\s)', combined)
            for section in sections:
                new_text = re.sub(r'^\d+\.', f'{labels[len(results)]}', section)
                if new_text != '' and new_text[0] != '（':
                    new_text = f'{labels[len(results)]} {new_text}'
                results.append(new_text)
        else:
            part = labels[len(results)] + part.strip()
            results.append(part)
    return results
def select_best_output_using_bert_score(input_text, outputs):
    if len(outputs) == 0:
        print("沒有可用的輸出")
        return ""
    bertscore = load("bertscore")
    print(outputs)
    results = bertscore.compute(
        predictions=outputs,
        references=[input_text] * len(outputs),  # repeat reference for each prediction
        lang="chinese",
        model_type="bert-base-chinese"
    )
    # 找出 F1 分數最高的輸出
    best_index = results["f1"].index(max(results["f1"]))
    return best_index

def compensate_iteration(user_input, references):
    summary = generate_total_summary(user_input)
    compensate_items = check_and_generate_summary_items(summary)
    reference_array = generate_reference_array(references)
    result = ""
    total_money = 0
    for i, item in enumerate(compensate_items):
        output = f"""==============================
{item}
[接續撰寫一段描述句]
==============================="""
        # 生成賠償項目
        retry_count = 0
        start_time = time.time()
        history_compensates = []
        input_compensate_prompt = compensate_prompt + f"\n\n生成格式:\n{output}"
        while True:
            if time.time() - start_time > 120:
                print(f"賠償項目{i+1}生成超過 120 秒，從過去{len(history_compensates)}次輸出中選擇最好的一個\n")
                yield tools.show_final_judge_to_UI(f'<span style="color:#4287f5;">賠償項目{i+1}生成超過 120 秒，從過去{len(history_compensates)}次輸出中選擇最好的一個</span>')
                final_index = select_best_output_using_bert_score(user_input, history_compensates)
                yield tools.show_final_judge_to_UI(f'<span style="color:#4287f5;">選擇第{final_index+1}筆作為賠償項目{i+1}生成內容</span>')
                response = history_compensates[final_index]
                break
            if retry_count >= 3:
                print(f"賠償項目{i+1}嘗試生成3次仍無法通過檢查，從過去3次輸出中選擇最好的一個\n")
                yield tools.show_final_judge_to_UI(f'<span style="color:#4287f5;">賠償項目{i+1}嘗試生成3次仍無法通過檢查，從過去3次輸出中選擇最好的一個"</span>')
                final_index = select_best_output_using_bert_score(user_input, history_compensates)
                yield tools.show_final_judge_to_UI(f'<span style="color:#4287f5;">選擇第{final_index+1}筆作為賠償項目{i+1}生成內容</span>')
                response = history_compensates[final_index]
                break
            response = tools.combine_prompt_generate_response(user_input, input_compensate_prompt).strip("=")
            first_sentence = response.strip().split('\n')[0].strip()
            other_sentences = '\n'.join(response.strip().split('\n')[1:]).strip()
            if len(other_sentences) and other_sentences[0] == '（' and other_sentences[-1] == '）':
                other_sentences = other_sentences[1:-1].strip()
                response = first_sentence + '\n' + other_sentences
            if len(other_sentences) == 0 or len(first_sentence) == 0 or first_sentence[0] != '（' or first_sentence[-1] != '元':
                print(response)
                print("格式錯誤，重新生成")
                print("=" * 50)
                # yield tools.show_debug_to_UI(f"{response}\n格式錯誤，重新生成\n" + "=" * 50)
            else:
                # 提取賠償項目以及金額
                summary_process_text = ""
                input_abs = generate_summary(item, final=False)
                output_abs = generate_summary(other_sentences, final=False)
                if input_abs == False or output_abs == False:
                    print("金額格式錯誤，重新生成")
                    # yield tools.show_final_judge_to_UI('<span style="color:#db272d;"金額格式錯誤，重新生成</span>')
                    continue
                print("輸出: ", other_sentences)
                summary_process_text += f"輸出: {other_sentences}<br><br>"
                print("輸入摘要:", input_abs)
                summary_process_text += f"輸入摘要:{input_abs}<br><br>"
                print("輸出摘要:", output_abs)
                summary_process_text += f"輸出摘要:{output_abs}<br><br>"
                yield tools.show_summary_to_UI(summary_process_text)

                judge_process_text = ""

                if input_abs['賠償項目'] != output_abs['賠償項目']:
                    processed_item_compensate_check = f"[{input_abs['賠償項目']}]，[{output_abs['賠償項目']}]" + item_compensate_check
                    item_response = tools.llm_generate_response(processed_item_compensate_check).replace('=', '')
                    print("賠償項目檢查:\n", item_response)

                    item_block = "賠償項目檢查<br>" + tools.wrap_debug_section(tools.remove_blank_lines(item_response), color="#d0c5bb", border="#d0c5bb")
                    judge_process_text += item_block
                else:
                    item_response = "accept"
                    print("輸入和輸出賠償項目一致，無需檢查")

                    item_block = "賠償項目檢查<br>" + tools.wrap_debug_section("輸入和輸出賠償項目一致，無需檢查", color="#d0c5bb", border="#d0c5bb")
                    judge_process_text += item_block

                money_response_1 = tools.combine_prompt_generate_response(input_abs["賠償金額"], money_to_number_prompt).replace('=', '')
                amount1 = get_exact_amount(money_response_1)
                if input_abs["賠償金額"] != output_abs["賠償金額"]:
                    money_response_2 = tools.combine_prompt_generate_response(output_abs["賠償金額"], money_to_number_prompt).replace('=', '')
                    amount2 = get_exact_amount(money_response_2)
                    money_block = f"輸入金額推理過程:<br>{tools.remove_blank_lines(money_response_1)}<br>輸出金額推理過程:<br>{tools.remove_blank_lines(money_response_2)}<br>"
                    print("金額檢查:\n", money_response_1, "\n", money_response_2)
                    if isinstance(amount1, str) and isinstance(amount2, str) and (not amount1.isdigit() or not amount2.isdigit()):
                        print("金額格式錯誤，重新生成")
                        money_block += "金額格式錯誤，重新生成"
                        money_response = "reject"
                    elif int(amount1) == int(amount2):
                        print("金額相同，通過檢查")
                        money_block += "金額相同，通過檢查"
                        money_response = "accept"
                    else:
                        print("金額不同，重新生成")
                        money_block += "金額不同，重新生成"
                        money_response = "reject"
                    judge_process_text += "賠償金額檢查<br>" + tools.wrap_debug_section(money_block, color="#b3d6c2", border="#b3d6c2")
                else:
                    print("輸入和輸出金額一致，無需檢查")
                    money_block = "賠償金額檢查<br>" + tools.wrap_debug_section("輸入和輸出金額一致，無需檢查", color="#b3d6c2", border="#b3d6c2")
                    money_response = "accept"
                    judge_process_text += money_block
                yield tools.show_debug_to_UI(judge_process_text)
                item_last_line = item_response.strip().split('\n')[-1].lower()
                money_last_line = money_response.strip().split('\n')[-1].lower()
                print("=" * 50)
                def contains_accept(text):
                    return "accept" in text or "接受" in text
                def contains_reject(text):
                    return "reject" in text or "拒絕" in text
                
                if contains_accept(item_last_line) and contains_accept(money_last_line) and not contains_reject(item_last_line) and not contains_reject(money_last_line):
                    yield tools.show_final_judge_to_UI('<span style="color:#2fd119;">最終判決結果: Accept</span>')
                    break
                elif contains_accept(item_response) and contains_accept(money_response) and not contains_reject(item_response) and not contains_reject(money_response):
                    yield tools.show_final_judge_to_UI('<span style="color:#2fd119;">最終判決結果: Accept</span>')
                    break
                else:
                    yield tools.show_final_judge_to_UI('<span style="color:#db272d;">最終判決結果: Reject</span>')
                    history_compensates.append(response)
                    retry_count += 1

        #去掉多餘的空白行
        lines = response.split('\n')
        non_empty_lines = [line for line in lines if line.strip() != '']
        total_money += int(amount1)
        print(amount1)
        response = '<br>'.join(non_empty_lines).strip()
        print(response)
        print("=" * 50)
        yield tools.show_result_to_UI(response)
        result += response + "\n\n"
    # 生成總結句
    processed_summary = summary + f"{labels[len(compensate_items)]}總計賠償金額: {total_money}元"
    print(processed_summary)
    print("=" * 50)
    # yield tools.show_debug_to_UI(f"賠償總結:\n{processed_summary}\n" + "=" * 50)
    processed_compensation_sum_prompt = processed_compensation_sum_prompt = compensation_sum_prompt + f"""
【範例格式】
{reference_array[-1]}

【賠償項目】
{processed_summary}

【請完成以下總結句】
請注意，總結句中的總金額應與最後一筆賠償項目「總計賠償金額」的金額一致（此例為{total_money}元），且應與前述各項賠償名稱與金額對應一致，僅進行格式統整，不需自行加減金額。
{labels[len(compensate_items)]}"""
    sum_response = ""
    print(processed_compensation_sum_prompt)
    retry_count = 0
    start_time = time.time()
    history_compensates = []
    while True:
        if time.time() - start_time > 120: 
            print(f"賠償項目{i+1}筆生成超過 120 秒，從過去{len(history_compensates)}次輸出中選擇最好的一個\n")
            yield tools.show_final_judge_to_UI(f'<span style="color:#4287f5;">賠償項目{i+1}筆生成超過 120 秒，從過去{len(history_compensates)}次輸出中選擇最好的一個</span>')
            final_index = select_best_output_using_bert_score(user_input, history_compensates)
            yield tools.show_final_judge_to_UI(f'<span style="color:#4287f5;">選擇第{final_index+1}筆作為賠償項目{i+1}生成內容</span>')
            sum_response = history_compensates[final_index]
            break
        if retry_count >= 3:
            print(f"賠償項目{i+1}嘗試生成3次仍無法通過檢查，從過去3次輸出中選擇最好的一個\n")
            yield tools.show_final_judge_to_UI(f'<span style="color:#4287f5;">賠償項目{i+1}嘗試生成3次仍無法通過檢查，從過去3次輸出中選擇最好的一個\n</span>')
            final_index = select_best_output_using_bert_score(user_input, history_compensates)
            yield tools.show_final_judge_to_UI(f'<span style="color:#4287f5;">選擇第{final_index+1}筆作為賠償項目{i+1}生成內容</span>')
            sum_response = history_compensates[final_index]
            break
        summary_process_text = ""
        sum_response = tools.llm_generate_response(processed_compensation_sum_prompt)
        print(sum_response)
        print("=" * 50)
        input_abs = generate_summary(f"{labels[len(compensate_items)]}總計賠償金額: {total_money}元", final=True)
        output_abs = generate_summary(sum_response, final=True)
        if input_abs == False or output_abs == False:
            print("金額格式錯誤，重新生成")
            # yield tools.show_final_judge_to_UI('<span style="color:#db272d;">金額格式錯誤，重新生成</span>')
            continue
        print("輸出: ", sum_response)
        summary_process_text += f"輸出: {sum_response}<br><br>"
        print("輸入摘要:", input_abs)
        summary_process_text += f"輸入摘要:{input_abs}<br><br>"
        print("輸出摘要:", output_abs)
        summary_process_text += f"輸出摘要:{output_abs}<br><br>"
        yield tools.show_summary_to_UI(summary_process_text)
        judge_process_text = ""
        money_response_1 = tools.combine_prompt_generate_response(input_abs["總賠償金額"], money_to_number_prompt).replace('=', '')
        amount1 = get_exact_amount(money_response_1)
        if input_abs["總賠償金額"] != output_abs["總賠償金額"]:
            money_response_2 = tools.combine_prompt_generate_response(output_abs["總賠償金額"], money_to_number_prompt).replace('=', '')
            amount2 = get_exact_amount(money_response_2)
            print("金額檢查:\n", money_response_1, "\n", money_response_2)
            money_block = f"輸入金額推理過程:<br>{tools.remove_blank_lines(money_response_1)}<br>輸出金額推理過程:<br>{tools.remove_blank_lines(money_response_2)}<br>"
            if isinstance(amount1, str) and isinstance(amount2, str) and (not amount1.isdigit() or not amount2.isdigit()):
                print("金額格式錯誤，重新生成")
                money_block += "金額格式錯誤，重新生成"
                money_response = "reject"
            elif int(amount1) == int(amount2):
                print("金額相同，通過檢查")
                money_block += "金額相同，通過檢查"
                money_response = "accept"
            else:
                print("金額不同，重新生成")
                money_block += "金額不同，重新生成"
                money_response = "reject"
            judge_process_text += "賠償金額檢查<br>" + tools.wrap_debug_section(money_block, color="#d0c5bb", border="#d0c5bb")
        else:
            print("輸入和輸出金額一致，無需檢查")
            money_block = "賠償金額檢查<br>" + tools.wrap_debug_section("輸入和輸出金額一致，無需檢查", color="#d0c5bb", border="#d0c5bb")
            money_response = "accept"
            judge_process_text += money_block
        yield tools.show_debug_to_UI(judge_process_text)
        item_last_line = item_response.strip().split('\n')[-1].lower()
        money_last_line = money_response.strip().split('\n')[-1].lower()
        
        def contains_accept(text):
            return "accept" in text or "接受" in text
        def contains_reject(text):
            return "reject" in text or "拒絕" in text
        
        if contains_accept(item_last_line) and contains_accept(money_last_line) and not contains_reject(item_last_line) and not contains_reject(money_last_line):
            yield tools.show_final_judge_to_UI('<span style="color:#2fd119;">最終判決結果: Accept</span>')
            break
        elif contains_accept(item_response) and contains_accept(money_response) and not contains_reject(item_response) and not contains_reject(money_response):
            yield tools.show_final_judge_to_UI('<span style="color:#2fd119;">最終判決結果: Accept</span>')
            break
        else:
            yield tools.show_final_judge_to_UI('<span style="color:#db272d;">最終判決結果: Reject</span>')
            history_compensates.append(sum_response)
            retry_count += 1
        print("=" * 50)
    if len(sum_response) > 0 and sum_response[0] != '（':
        sum_response = labels[len(compensate_items)] + sum_response
    result += sum_response
    yield tools.show_final_judge_to_UI('<span style="color:gray; font-weight:bold;">生成結束</span>')
    yield tools.show_result_to_UI(sum_response)
    print(result)
    return result

def generate_compensate(user_input, references, passed_tools):
    # 生成賠償項目
    global tools
    tools = passed_tools
    result = None
    size = len(references)
    id = 0
    while result == None:#生成錯誤
        print("參考資料:\n", references[(id) % size])
        result = yield from compensate_iteration(user_input, references[(id) % size])
        id += 1
    return result

def generate_simple_compensate(input_text, injuries, compensation_facts, references, passed_tools):
    global tools
    tools = passed_tools
    # 生成賠償項目
    case_type, plaintiffs_info = chunk_get_case_type(input_text)
    reference_array = generate_reference_array(references[0])
    is_multiple_plaintiffs = any(x in case_type for x in ["數名原告", "原被告皆數名"])
    if is_multiple_plaintiffs:
        prompt = get_compensation_prompt_part1_multiple_plaintiffs(injuries, compensation_facts, plaintiffs_info=plaintiffs_info)
    else:
        prompt = get_compensation_prompt_part1_single_plaintiff(injuries, compensation_facts, plaintiffs_info=plaintiffs_info)
    result_part_1 = tools.retrieval_system.clean_compensation_part(tools.llm_generate_response(prompt))
    print(result_part_1)
    print("=" * 50)
    yield result_part_1.replace('\n', '<br>')
    processed_compensation_sum_prompt = compensation_sum_prompt + f"""
    【範例格式】
    {reference_array[-1]}

    【賠償項目】
    {result_part_1}

    【請完成以下總結句】"""
    result_part_2 = tools.llm_generate_response(processed_compensation_sum_prompt)
    print(result_part_1 + '\n\n' + result_part_2)
    yield result_part_2

    return result_part_1 + result_part_2
if __name__ == "__main__":
    start_time = time.time()
    tools = Tools("kenneth85/llama-3-taiwan:8b-instruct-dpo")
    # for part, reference, summary, log, final_judge in generate_compensate(user_input, references, tools):
    #     pass
    for part in generate_simple_compensate(user_input, injuries, compensation_facts, references, tools):
        pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n執行時間: {hours}h {minutes}m {seconds}s")