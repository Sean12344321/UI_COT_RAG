from ollama import chat, ChatResponse
from utils import Tools
import re
single_money_summary = """你是一個格式輸出助手，任務是將原告的賠償項目與金額，依照指定格式列出，完全依據輸入，不得遺漏、改寫、推論、補充。
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
multiple_money_summary = """你是一個格式輸出助手，任務是將原告的賠償項目與金額，依照指定格式列出，完全依據輸入，不得遺漏、改寫、推論、補充。
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
detect_name = """你是一個資訊擷取助手，請根據下列輸入的民事訴狀內容，判斷是否有提及特定原告姓名。如果有，請列出該姓名；如果沒有出現任何原告姓名，則統一輸出為「原告」。

【任務規則】：
1. 請優先尋找明確出現的人名（例如：「林小明」、「甲○○」等），並以其作為原告名稱。
2. 如果沒有明確人名，則統一使用「原告」作為名稱。
3. 不需解釋或重述內容，只需輸出一個人名或「原告」。
4. 若出現多名原告（如「陳○○及林○○」），請分別列出所有人名，並且用頓號分隔。

【請根據使用者輸入內容進行判斷並輸出原告名稱】：
"""
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
reference = """（一）醫療費用：146,363元
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
"""
compensate_prompt = f"""
你是一位熟悉法律文書格式的語言模型。請從`輸入`中讀取內容，並以`生成格式`作為開頭，接續撰寫一段描述句，說明該筆費用的原因與金額使用情況。
⚠️請特別注意：
1. 你應該 **僅生成與 `生成格式` 所示金額相符的那一筆費用描述**。如果 `輸入` 中包含多筆金額或多位人員，請勿引用其他不相關的金額。
2. 輸出應包含 `生成格式` 開頭的標題行與一段敘述，兩者缺一不可。
3. 金額格式須為「#,###元」，務必與 `生成格式` 數字一致。
4. 第二行應僅為事實陳述，包括傷勢、用途、支出情形等，不得包含任何法律條文、條號、責任、請求、結語或主觀評價，例如「被告應賠償」、「依據民法」等字句皆不得出現。
5. 請以「（」開頭。"""

compensate_head_check = f"""
請比較上述兩筆資料，判斷它們是否代表相同的賠償項目。你需要依據以下三項標準進行比對：

1. 【項目標號】是否一致（例如：（三））。
2. 【賠償項目名稱】是否一致（例如：慰撫金、機車修理費等）。
3. 【整體金額標題行】是否描述相同的費用內容。

如果三者都一致，則視為「相同的賠償項目」。若有任何一項不同，請說明是哪些項目不同，並判定為「不同的賠償項目」。
請依照以下格式生成輸出：
===========================
[推理過程]:
(請寫下每一項檢查的判斷過程)

[判決結果]:
(填入accept 或 reject)
===========================
⚠️ 注意：只要任何一項為「否」，就必須輸出「reject」。
不要同時出現 accept 和 reject，只能選一個。
"""
compensation_sum_prompt = """請參考以下「範例格式」，將給定的各筆損害賠償項目重新整理成總結句，格式須一致："""
labels = [
    '（一）', '（二）', '（三）', '（四）', '（五）', '（六）', '（七）', '（八）', '（九）', '（十）',
    '（十一）', '（十二）', '（十三）', '（十四）', '（十五）', '（十六）', '（十七）', '（十八）', '（十九）', '（二十）',
    '（二十一）', '（二十二）', '（二十三）', '（二十四）', '（二十五）', '（二十六）', '（二十七）', '（二十八）', '（二十九）', '（三十）'
]
def check_and_generate_summary_items(text):
    text_array = []  # 去除多餘的空格和換行
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        line = line.rstrip()
        if len(line) >= 4 and line[0] == '（' and (line[2] == '）' or line[3] == '）') and '+' not in line and '=' not in line and line[-1] == '元':
            match = re.search(r'(\d+(,\d{3})*|\d+)元$', line)
            if match and match.group(0).replace(',', '') == '0元':
                print("money should not be 0")
                return False
            match = re.search(r'（(.*?)）', line)
            if match.group(0) not in labels: # 括號裡面不是中文字
                print("（）should contain chinese")
                return False
            line = line[:match.start()] + labels[len(text_array)] + line[match.end():]
            line = re.sub(r'[。\.]', '', line)
            text_array.append(line)
        elif line != "":
            print("format error")
            return False
    return text_array

def generate_compensate_summary(input):
    """1. 取得原告姓名，判斷是單名還是多名原告來改變提示詞"""
    name = Tools.combine_prompt_generate_response(input, detect_name)
    while '\n' in name:
        print("姓名格式錯誤，重新生成")
        print(name)
        name = Tools.combine_prompt_generate_response(input, detect_name)
    print(name)
    if '、' in name:
        prompt = multiple_money_summary
    else:
        prompt = single_money_summary
    """2. 取得賠償摘要"""
    summary = Tools.combine_prompt_generate_response(input, prompt)
    while(check_and_generate_summary_items(summary) == False):
        print("格式錯誤，重新生成")
        print(summary)
        print("=" * 50)
        summary = Tools.combine_prompt_generate_response(input, prompt)
    print(summary)
    print("=" * 50)
    return summary

    
def generate_reference_array(reference):
    parts = re.split(r'（[一二三四五六七八九十]{1,3}）', reference)[1:]
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


def compensate_iteration(user_input, reference):
    summary = generate_compensate_summary(user_input)
    compensate_items = check_and_generate_summary_items(summary)
    reference_array = generate_reference_array(reference)
    result = ""
    for i, item in enumerate(compensate_items):
        output = f"""==============================
    {item}
    [接續撰寫一段描述句]
    ==============================="""
        retry_count = 0
        input_compensate_prompt = compensate_prompt + f"\n\n生成格式:\n{output}"
        while True:
            response = Tools.combine_prompt_generate_response(user_input, input_compensate_prompt)
            first_sentence = response.strip().split('\n')[0].strip()    
            if first_sentence[0] != '（' or first_sentence[-1] != '元':
                print("格式錯誤，重新生成")
                print(response)
                print("=" * 50)
            else:
                input_compensate_head_check = f"段落一:{first_sentence}\n段落二{item}" + compensate_head_check
                compensate_response = Tools.llm_generate_response(input_compensate_head_check)
                print(compensate_response)
                print("=" * 50)
                if "accept" in compensate_response or "Accept" in compensate_response:
                    break
                else:
                    print("COT檢測錯誤，重新生成")
                    print(response)
                    print("=" * 50)
                    retry_count += 1
            if retry_count >= 7:
                print(f"第{i+1}筆 item 嘗試超過 7 次仍無法通過檢查，跳過處理並重新生成整體 text。\n")
                return None
        #去掉多餘的空白行
        lines = response.split('\n')
        non_empty_lines = [line for line in lines if line.strip() != '']
        response = '\n'.join(non_empty_lines).strip()
        print(response)
        print("=" * 50)
        result += response + "\n\n"
    # 生成總結句
    global compensation_sum_prompt
    input_compensation_sum_prompt = compensation_sum_prompt + f"""【範例格式】\n{reference_array[-1]}\n\n【賠償項目】\n{summary}【請完成以下總結句】\n{labels[len(compensate_items)]}"""
    sum_response = Tools.llm_generate_response(input_compensation_sum_prompt)
    if sum_response[0] != '（':
        sum_response = labels[len(compensate_items)] + sum_response
    result += sum_response
    return result

def generate_compensate(user_input, references):
    # 生成賠償項目
    result = None
    id = 0
    while result == None:#生成錯誤
        print(references[(id) % 5])
        result = compensate_iteration(user_input, references[(id) % 5])
        id += 1
    return result

if __name__ == "__main__":
    generate_compensate(user_input, reference)