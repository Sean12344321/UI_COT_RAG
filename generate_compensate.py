from ollama import chat, ChatResponse
from utils import Tools
from KG_RAG_B.define_case_type import get_case_type
import re, time
user_input = """
一、事故發生緣由:
 被告在民國112年10月27日15時許，駕駛車牌號碼000-0000號自用小客車，沿基隆市安樂區麥金路直行往基金一路方向行駛，行經麥金路與安國橋路口要右轉時，本應注意轉彎車應讓直行車先行，以避免危險或交通事故之發生，而當時是具自然光線之日間、柏油路面未積水、無障礙物且視距良好，並無不能注意之情事，但被告仍疏於注意，即貿然右轉彎，因而與直行至該路口之原告發生碰撞，導致原告遭被告駕駛之汽車撞擊倒地（下稱系爭車禍）。
 
 二、原告受傷情形:
 原告因為被告引起的這場車禍受有頭部外傷、腦部創傷性腦損傷、創傷性蜘蛛膜下出血、視力模糊、視神經受損、外斜視、垂直性斜視等傷害，其後並有壓力後創傷症候群、憂鬱症、適應性失眠症等後遺症。
 
 三、請求賠償的事實根據:
按民法第184條第1項前段、第191條之2本文、第193條第1項、第195條第1項前段，請求下列損害
 （一）醫療費用部分合計8萬3,016元
 1、住院開刀醫療費用
 原告因系爭車禍於112年10月27日經急診入院，並於同日接受顱內監測器放置手術，後來於000年00月00日出院，期間之手術及住診費用共計支出新臺幣5萬3,122元，有基隆長庚紀念醫院（以下簡稱基隆長庚）復健科費用收據可以證明。
 2、復健科、腦神經外科、神經科診療費用
 原告於系爭車禍發生時因頭部遭受撞擊而有頭部外傷、創傷性蜘蛛膜下出血等傷勢，並進而造成創傷性腦損傷，須接受復健治療，因此支出有一般性高壓氧治療費用共1萬0500元；神經科診療費用2次共802元、腦神經外科診療費用8次共3,520元，合計共1萬4,822元，有基隆長庚復健科之診斷證明書，與復健科、腦神經外科、神經科費用收據可以證明。
 3、眼科診療費用
 原告因系爭車禍致視神經受損視力模糊，有外斜視、垂直斜視之傷勢，須至基隆長庚眼科回診接受治療，共二次支出有診療費1,280元。且因視神經受損，原告平日間為聚焦看清物體，眼部肌肉須特別用力，導致眼睛較一般人更容易感到疲勞，原告因而另至基隆市瑞光眼科看診2次接受治療，支出有看診費400元，以上費用合計1,680元，此有基隆長庚眼科診斷證明書、費用收據，與瑞光眼科費用收據可以證明。
 4、整形外科診療費用
 原告因系爭車禍而人車倒地，臉部與路面發生摩擦，致有臉部擦傷、上唇肥厚性疤痕色素沉澱，因而於112年12月20日、113年1月24日、113年5月28日三度至基隆長庚接受治療，支出有看診費用1萬1,832元，以及證明前開損害所用之診斷及證明書費100元，共計1萬1,932元，此有基隆長庚之整形外科診斷證明書與費用收據可以證明。
 5、精神科與中醫診所診療費用
 原告因系爭車禍腦部受傷，致有壓力後創傷症候群、憂鬱症、適應性失眠症等症狀。平日除常有偏頭痛之症狀，夜間更因患有憂鬱症而失眠無法正常入睡，即便入睡後亦常於半夜驚醒，故原告於113年2月14日至基隆長庚精神科看診接受治療，支出有診用510元；113年4月26日至000年0月0日間。五度至基隆陳字斌中醫診所接受睡眠障礙症之治療並接受中醫藥劑調養，因而支出有看診費850元，以及證明前開損害所用之診斷及證明書費100元，共計1,460元，此有基隆長庚與陳字斌中醫診所之診斷書與醫療費用收據可以證明。
 
 （二）交通費部分215元
 原告於113年5月28日因須至基隆長庚醫院整形外科回診，來回一共支出計程車交通費用215元。
 
 （三）看護費用部分共計52萬5,000元
 本件原告因系爭車禍受傷入院接受手術開刀治療後，112年12月18日經基隆長庚診斷「出院後宜休養3個月以上，並需專人24小時照護」；113年4月25日回診後，醫師認為因原告「創傷後視力及記憶力恢復不全，情緒不穩定，需專人長時間陪護照顧至少2個月」；113年6月13日回診後，醫師仍認「須專人長時間陪護照顧至少2個月以上」，此有歷次之診斷證明書可證明。從而原告共計有7個月之時間需由專人照護陪伴，又雖原告並未聘僱看護照顧，而是由其婆婆與配偶輪流代為照顧，然而依照最高法院94年度台上字第1543號裁判意旨，縱原告因親屬關係而無現實之看護費支出，仍應比照一般看護情形，認定原告受有相當於看護費之損害，故原告參考基隆長庚紀念醫院112年招募全日看護工照顧服務員之徵人啟事作為行情價標準，認為一般長時間專人照顧服務員，每日薪資約為2,500至3,000元之間，而本件原告因系爭車禍受傷在家休養，診斷證明書並載明須專人24小時照護，原告因未實際聘僱看護而是由婆婆代為照顧，因此僅以最低價2,500元計算每日看護費用，請求7個月共計52萬5,000元【計算式：2,500元×30日×7個月＝52萬5,000元】。
 
 （四）無法家務勞動之損失部分共計18萬9,080元
 原告雖然是家庭主婦，並無現實之工作收入資料可為參考，然依最高法院92年度台上字第1626號判決意旨，原告於家中處理家務之勞動能力，應能以另僱他人代勞而支出之報酬予以評價，故原告所受無法從事家務勞動之損失，自應比照基本工資而為計算。本件事故發生於112年10月27日，原告因系爭車禍而受傷，112年12月18日經基隆長庚醫院診斷出院後「宜休養3個月以上，並需專人24小時照護」；113年4月25日回診後，醫師認因原告「創傷後視力及記憶力恢復不全，情緒不穩定，需專人長時間陪護照顧至少2個月」，依行政院勞動部公布於112年1月1日起實施之基本工資為每月2萬6,400元；113年1月1日起實施之基本工資為每月2萬7,470元作為基準計算，原告此7個月無法從事家務勞動之工作損失應為18萬9,080元【計算式：2萬6,400元×3月＋2萬7,470元×4月=18萬9,080元】。
 
 （五）慰撫金部分130萬元
 被告駕駛自用小客車，於路口右轉彎時未注意右側騎乘普通重型機車直行之原告，竟於路口先偏左後又貿然右轉而撞擊原告，致原告人車倒地。原告於系爭車禍發生後，因頭部遭受撞擊而有腦内出血，隨即接受腦部緊急手術，於醫院中昏迷十數日才清醒，並住院將近1個月，然即便於出院後亦因創傷性腦損傷、憂鬱症、記憶力衰退、視力受損造成原告一般生活有諸多不便，除時有偏頭痛外，夜間更因而無法正常入眠，且事發距今已逾八個月，原告仍須定期回診接受治療，對於身心靈造成莫大痛苦。因原告所受之腦部傷勢為一長期而持續之傷害，勢必對於未來之生活造成影響，且原告目前擔任家庭主婦，尚有二名未成年子女須扶養照顧，然配偶因工作性質時常出差不在國内，本次因系爭車禍致無法照顧未成年子女與從事家務勞動，家中原先平靜之生活頓時大亂，因此原告主張請求慰撫金130萬元。
"""
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
compensation_amount_prompt = """你是一位專業的法律文書助理，負責閱讀交通事故民事起訴狀中的賠償敘述。
你的任務是從【賠償敘述】中，抽取出文中請求的「總賠償金額」（單位為新台幣元）。
請務必遵循以下流程：

1. 逐行閱讀敘述，列出每一筆出現的金額及其對應的描述。
2. 判斷哪些金額是屬於賠償項目的金額。
3. 將這些賠償金額相加，計算出「總賠償金額」。
4. 最後，回覆以下格式：
- 推理過程：（請簡單列出找到的金額及加總過程）
- 最終答案：（只填數字，不加任何單位或文字）

【賠償敘述】：
"""
extract_amount_prompt = """你是一位專業的資訊處理助理，負責從輸入的短句中提取賠償金額。

請務必遵循以下步驟：
1. 推理過程：請列出你是如何從句子中找到金額的，例如指出金額的原始表達、是否有逗號需要去除、是否需要中文轉阿拉伯數字等。
2. 最終答案：只回覆提取後的阿拉伯數字，不要出現逗號（,）。

注意事項：
- 如果金額中有逗號，請去除後輸出完整連續的數字。
- 如果金額用中文數字表示（如「十萬元」、「二十五萬」），請換算成阿拉伯數字後輸出。
- 如果找不到任何金額，請最終答案回覆「0」。

輸入句子如下：
"""
compensate_prompt = f"""
你是一位熟悉法律文書格式的語言模型。請從`輸入`中讀取內容，並以`生成格式`作為開頭，接續撰寫一段描述句，說明該筆費用的原因與金額使用情況。
⚠️請特別注意：
1. 你應該 **僅生成與 `生成格式` 所示金額相符的那一筆費用描述**。如果 `輸入` 中包含多筆金額或多位人員，請勿引用其他不相關的金額。
2. 輸出應包含 `生成格式` 開頭的標題行與一段敘述，兩者缺一不可。
3. 金額格式須為「#,###元」，務必與 `生成格式` 數字一致。
4. 第二行應僅為事實陳述，包括傷勢、用途、支出情形等，不得包含任何法律條文、條號、責任、請求、結語或主觀評價，例如「被告應賠償」、「依據民法」等字句皆不得出現。
5. 請以「（」開頭。"""

compensation_sum_prompt = """請參考以下「範例格式」，將給定的各筆損害賠償項目重新整理成總結句，格式須一致："""
labels = [
    '（一）', '（二）', '（三）', '（四）', '（五）', '（六）', '（七）', '（八）', '（九）', '（十）',
    '（十一）', '（十二）', '（十三）', '（十四）', '（十五）', '（十六）', '（十七）', '（十八）', '（十九）', '（二十）',
    '（二十一）', '（二十二）', '（二十三）', '（二十四）', '（二十五）', '（二十六）', '（二十七）', '（二十八）', '（二十九）', '（三十）'
]
tools = None
def get_exact_amount(extract_amount_prompt):
    response = tools.llm_generate_response(extract_amount_prompt)
    print("金額提取過程:\n", response)
    yield tools.show_debug_to_UI(f"金額提取過程:\n{response}")
    numbers = re.findall(r'\d+', response)
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
                yield tools.show_debug_to_UI("金額不應為0")
                return False
            match = re.search(r'（(.*?)）', line)
            if match.group(0) not in labels: # 括號裡面不是中文字
                print("（）should contain chinese")
                yield tools.show_debug_to_UI("括號內應為中文")
                return False
            line = line[:match.start()] + labels[len(text_array)] + line[match.end():]
            line = re.sub(r'[。\.]', '', line)
            text_array.append(line)
        elif line != "":
            print("format error")
            yield tools.show_debug_to_UI("格式錯誤")
            return False
    return text_array

def generate_compensate_summary(input_text):
    """1. 取得原告姓名，判斷是單名還是多名原告來改變提示詞"""
    case_type = get_case_type(input_text)
    print(f"案件類型: {case_type}")
    yield tools.show_debug_to_UI(f"案件類型: {case_type}")
    if case_type == "單純原被告各一" or case_type == "數名被告":
        prompt = single_money_summary_prompt
    else:
        prompt = multiple_money_summary_prompt
    """2. 取得賠償摘要"""
    summary = tools.combine_prompt_generate_response(input_text, prompt)
    judge = yield from check_and_generate_summary_items(summary)
    while(judge == False):
        print(summary)
        print("格式錯誤，重新生成")
        print("=" * 50)
        yield tools.show_debug_to_UI(f"{summary}\n格式錯誤，重新生成\n" + "=" * 50)
        summary = tools.combine_prompt_generate_response(input_text, prompt)
        judge = yield from check_and_generate_summary_items(summary)
    print(summary, '\n', "=" * 50)
    yield tools.show_debug_to_UI(f"賠償摘要:\n{summary}\n" + "=" * 50)
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


def compensate_iteration(user_input, references):
    summary = yield from generate_compensate_summary(user_input)
    compensate_items = yield from check_and_generate_summary_items(summary)
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
        input_compensate_prompt = compensate_prompt + f"\n\n生成格式:\n{output}"
        while True:
            response = tools.combine_prompt_generate_response(user_input, input_compensate_prompt)
            first_sentence = response.strip().split('\n')[0].strip()    
            other_sentences = '\n'.join(response.strip().split('\n')[1:]).strip()
            if first_sentence[0] != '（' or first_sentence[-1] != '元':
                print(response)
                print("格式錯誤，重新生成")
                print("=" * 50)
                yield tools.show_debug_to_UI(f"{response}\n格式錯誤，重新生成\n" + "=" * 50)
            else:
                processed_compensation_amount_prompt = compensation_amount_prompt + other_sentences
                compensate_response = tools.llm_generate_response(processed_compensation_amount_prompt)
                print(f"賠償描述:\n{other_sentences}")
                print(f"賠償金額推理過程:\n{compensate_response}")
                yield tools.show_debug_to_UI(f"賠償描述:\n{other_sentences}\n賠償金額推理過程:\n{compensate_response}")
                # 提取金額
                processed_extract_amount_prompt = extract_amount_prompt + first_sentence
                amount1 = yield from get_exact_amount(processed_extract_amount_prompt)
                processed_extract_amount_prompt = extract_amount_prompt + compensate_response.split('\n')[-1]
                amount2 = yield from get_exact_amount(processed_extract_amount_prompt)
                # 判斷金額是否相同
                print(f"原本句子:{first_sentence}, 提取金額:{amount1}")
                print(f"生成句子:{compensate_response.split('\n')[-1]}, 提取金額:{amount2}")
                yield tools.show_debug_to_UI(f"原本句子:{first_sentence}, 提取金額:{amount1}\n生成句子:{compensate_response.split('\n')[-1]}, 提取金額:{amount2}")
                if not amount1.isdigit() or not amount2.isdigit():
                    print("金額格式錯誤，重新生成")
                    yield tools.show_debug_to_UI("金額格式錯誤，重新生成")
                    retry_count += 1
                elif amount1 == amount2:
                    print("金額相同，通過檢查")
                    yield tools.show_debug_to_UI("金額相同，通過檢查")
                    total_money += int(amount1)
                    break
                else:
                    print("金額不同，重新生成")
                    yield tools.show_debug_to_UI("金額不同，重新生成")
                    retry_count += 1
                print("=" * 50)
                yield tools.show_debug_to_UI("=" * 50)
            if retry_count >= 7:
                print("賠償項目嘗試超過 7 次仍無法通過檢查，跳過處理並重新生成整體 text。\n")
                yield tools.show_debug_to_UI("賠償項目嘗試超過 7 次仍無法通過檢查，跳過處理並重新生成整體 text。\n")
                return None
        #去掉多餘的空白行
        lines = response.split('\n')
        non_empty_lines = [line for line in lines if line.strip() != '']
        response = '\n'.join(non_empty_lines).strip()
        print(response)
        print("=" * 50)
        yield tools.show_result_to_UI(response)
        result += response + "\n\n"
    # 生成總結句
    processed_summary = summary + f"\n{labels[len(compensate_items)]}總計賠償金額: {total_money}元"
    print(processed_summary)
    print("=" * 50)
    yield tools.show_debug_to_UI(f"賠償總結:\n{processed_summary}\n" + "=" * 50)
    processed_compensation_sum_prompt = compensation_sum_prompt + f"""【範例格式】\n{reference_array[-1]}\n\n【賠償項目】\n{processed_summary}【請完成以下總結句】\n{labels[len(compensate_items)]}"""
    retry_count = 0
    while True:
        sum_response = tools.llm_generate_response(processed_compensation_sum_prompt)
        processed_compensation_amount_prompt = compensation_amount_prompt + sum_response
        compensate_response = tools.llm_generate_response(processed_compensation_amount_prompt)
        # 提取金額
        processed_extract_amount_prompt = extract_amount_prompt + compensate_response.split('\n')[-1]
        amount = yield from get_exact_amount(processed_extract_amount_prompt)
        print(sum_response)
        print(f"賠償金額推理過程:{compensate_response}")
        print(f"賠償金額:{amount}, 加總金額:{total_money}")
        yield tools.show_debug_to_UI(f"賠償金額推理過程:\n{compensate_response}\n賠償金額:{amount}, 加總金額:{total_money}")
        # 判斷金額是否相同
        if not amount.isdigit():
            print("金額格式錯誤，重新生成")
            yield tools.show_debug_to_UI("金額格式錯誤，重新生成")
            retry_count += 1
        elif int(amount) == int(total_money):
            print("金額相同，通過檢查")
            yield tools.show_debug_to_UI("金額相同，通過檢查")
            break
        else:
            print("金額不同，重新生成")
            yield tools.show_debug_to_UI("金額不同，重新生成")
            retry_count += 1
        print("=" * 50)
        yield tools.show_debug_to_UI("=" * 50)
        if retry_count >= 7:
            print("賠償項目嘗試超過 7 次仍無法通過檢查，跳過處理並重新生成整體 text。\n")
            yield tools.show_debug_to_UI("賠償項目嘗試超過 7 次仍無法通過檢查，跳過處理並重新生成整體 text。\n")
            return None
    if sum_response[0] != '（':
        sum_response = labels[len(compensate_items)] + sum_response
    result += sum_response
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

if __name__ == "__main__":
    start_time = time.time()
    tools = Tools("kenneth85/llama-3-taiwan:8b-instruct-dpo")
    for part, reference, log in generate_compensate(user_input, references, tools):
        # print(f"生成的內容:\n{part}")
        # print(f"參考資料:\n{reference}")
        # print(f"推理紀錄:\n{log}")
        pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n執行時間: {hours}h {minutes}m {seconds}s")