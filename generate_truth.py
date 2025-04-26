from utils import Tools
import re, time
# 事實陳述摘要
truth_summary = """請根據輸入資訊擷取關鍵資訊，並遵循以下格式輸出：
============================
[案發時間]:預設無
[案發地點]:預設無
============================
如果沒有相關資訊，生成預設內容。
如果輸入有o或0字眼，請照抄輸出，不要擅自修改。
"""
# 生成事實陳述提示詞
truth_prompt = """
嚴格遵照範例格式生成事實陳述，不要參考範例格式內容，不要有任何列點或是換行，並以「緣」開頭。
"""
# 事實陳述檢查
address_truth_check = """
你要比較括號中的兩個地址是否在實質上屬於同一地點。請依以下規則判斷：

步驟一：如果任一方的地址為「無」、「未記載」、「不詳」或明顯缺漏，則直接判斷為「reject」，無需進行後續比對。
步驟二：比對「城市」、「地區」、「道路名稱」是否一致。若一致則進行下一步；若有不同，則直接輸出「reject」。
步驟三：門牌號碼若完全相同，則視為一致；若有小幅差異（例如門牌差距在50號以內），仍可視為一致；
步驟四：如果描述中出現「附近」、「對面」、「旁邊」等模糊詞，可容許小範圍差異，視為一致。
步驟五：依上述檢查結果，若整體可認定為同一地點，則輸出「accept」；若有明確不同，則輸出「reject」。

請依照以下格式生成輸出：
===========================
[推理過程]:
(請逐項說明城市、地區、道路名稱與門牌號碼的比對結果，以及你的判斷依據)

[判決結果]:
(只填 accept 或 reject，不能同時出現)
===========================
"""
time_truth_check = """
請你比對括號中的兩個時間是否完全一致，包括以下 4 個項目：

1. 民國年份是否相同
2. 月份是否相同
3. 日期是否相同
4. 時間（例如上午8時15分或晚間10時）是否相同

請按照以下格式回答：

===========================
[推理過程]:
逐項比較：
1. 年份: 
2. 月份:
3. 日期: 
4. 時間: 

[判決結果]:
===========================
⚠️ 注意：只要任何一項為「否」，就必須輸出「reject」。
不要同時出現 accept 和 reject，只能選一個。
"""
user_input = """一、事故發生緣由:
 被告於民國000年0月00日下午5時許，駕駛車牌號碼0000-00號自用小客車（下稱A車），自基隆市中正區碧砂漁港駛出欲右轉往中正路方向行駛之際，於該址路口（下稱系爭地點）本應注意行駛在閃紅燈之支線道車應暫停讓幹線道車先行，以避免危險或交通事故之發生，而依當時天候及路況，並無不能注意之情事，竟疏未注意上情而貿然前行，此時原告甲父騎乘車牌號碼000-000號普通重型機車（下稱B車）附載原告甲及甲母行駛至系爭地點，雙方因而發生碰撞（下稱系爭事故），原告甲、甲父、甲母因此分別受有身體傷害及財物損失。
 
 二、原告受傷情形:
 原告甲父因系爭事故受有脾臟重度撕裂傷合併腹內出血及休克、左側第6至第9肋骨骨折、左小腿及左腳踝大面積傷口、左側肺挫傷併少量血胸等傷害。
 原告甲母因系爭事故受有四肢及臉部多處挫傷及擦傷。
 原告甲因系爭事故受有左膝挫傷、擦傷及左手第五掌骨閉鎖性骨折之傷害。
"""
#暫時的參考資料，實際使用時會從資料庫中獲取
rag_references = ['一、緣被告明知其駕駛執照經吊扣，不得駕駛自用小貨車行駛於道路，仍於102年5月31日上午10時15分許，駕駛車號00-0000號自用小貨車，沿雲林縣斗南鎮臺一線由北往南方向行駛，途經同路段南向239.2公里處時，其原應注意汽車行駛變換車道時，讓直行車先行，並注意安全距離。依當時天候晴、日間自然光線、柏油路面乾燥無缺陷、無障礙物，並無不能注意之情事，竟疏於注意及此，適有原告騎乘000-000號普通重型機車同向行駛於被告後方機車道上，致2車發生擦撞，造成原告受傷。', '一、緣被告於民國108年5月31日8時45分許，駕駛其所有車牌號碼000-0000號自小客貨車，沿彰化縣彰化市公園路1段由北往南方向行駛，行經該路段412號前之無號誌交岔路口，欲右轉公園路1段往西方向行駛時，疏未注意車前狀況及讓直行車先行，貿然右轉，適原告騎乘車牌號碼000-000號普通重型機車，沿彰化市公園路1段由東往西方向直行至該路口，兩車因而發生碰撞，原告人車倒地。', '一、緣被告於民國107年7月30日下午,駕駛其所有車牌號碼000-00號自用小貨車,沿彰化縣芳苑鄉海埔路往南行駛,嗣於同日17時5分許,行經海埔路與功湖路無號誌交岔路口時,疏未注意車前狀況,讓直行車先行,竟貿然左轉駛往功湖路,適原告騎乘車牌號碼000-000號普通重型機車,沿功湖路往西方向直行,兩車於上開路口發生碰撞,致原告人車倒地,並受有頭部外傷、左頸部挫傷、左腕擦傷及雙膝挫傷等傷害。', '一、緣被告於民國107年9月7日下午16時50分許，駕駛000-0000號營業大貨車沿臺南市○○區○道○號公路由北向南行駛於道路中線車道，至該路段南向328公里700公尺處，本應注意變換車道時應讓直行車先行，並注意安全距離。當時天候雖為陰天，惟日間有自然光線，柏油路面乾燥、無缺陷、無障礙物、視距良好，並無不能注意之情形，依被告駕駛能力亦應能注意，竟貿然向右變換至外側車道，致擦撞同向平行行駛於外側車道、由原告駕駛之00-0000號自小客車左前方，致該車受撞後衝撞右側路旁護欄反轉逆向才停止。', '一、緣被告於民國108年5月31日7時58分許，駕駛車牌號碼000-0000號自用小客貨車，沿屏東縣恆春鎮中正路239巷39弄由北往南方向行駛，行經該巷與同巷39弄時，被告原應注意轉彎車應讓直行車先行，竟疏未注意而貿然右轉，碰撞由原告騎乘車牌號碼為000-0000號普通重型機車。且當時路口正前方設有反射鏡，該反射鏡反射範圍即為被告左側之原告直行而來之道路情況，顯係特別設置供被告行向之車輛於駛入該路口欲右轉時，注意左側原告行向之來車使用，被告依其行向駛至該路口欲右轉時，客觀上並無不能注意左側有原告來車之情事。']

def generate_simple_fact_statement(input, reference_fact):
    """"直接生成, 沒有用chain of though以及summary, 只有結合Input跟rag的reference"""
    global truth_prompt
    truth_prompt = f"\n範例格式:{reference_fact}" + truth_prompt
    input = Tools.remove_input_specific_part(input)
    return "一、" + Tools.combine_prompt_generate_response(input, truth_prompt)

def generate_summary(input):
    info_dict = {}
    time = 0
    while time < 5:#生成至多5次
        time += 1
        abstract = Tools.combine_prompt_generate_response(input, truth_summary)
        matches = re.findall(r"\[(案發時間|案發地點)\]:\s*(.*)", abstract)
        # 轉換為字典
        info_dict = {k: v for k, v in matches}
        # 如果資訊不構成二個欄位則重新生成
        if len(info_dict) == 2:
            break
    if time >= 5:
        return False
    return info_dict

def check_input_output_content(input, output):
    # 檢查輸入和輸出內容是否一致
    global address_truth_check
    global time_truth_check
    print("輸出:\n", output)
    input_abs = generate_summary(input)
    output_abs = generate_summary(output)
    if input_abs == False or output_abs == False:
        return False
    print("輸入摘要:\n", input_abs)
    print("輸出摘要:\n", output_abs)
    processed_address_truth_check = f"[{input_abs["案發地點"]}]，[{output_abs["案發地點"]}]" + address_truth_check
    processed_time_truth_check = f"[{input_abs["案發時間"]}]，[{output_abs["案發時間"]}]" + time_truth_check
    address_response = Tools.llm_generate_response(processed_address_truth_check).replace('=', '')
    time_response = Tools.llm_generate_response(processed_time_truth_check).replace('=', '')
    print("地址檢查:\n", address_response)
    print("時間檢查:\n", time_response)
    address_last_line = address_response.strip().split('\n')[-1].lower()
    time_last_line = time_response.strip().split('\n')[-1].lower()
    
    def contains_accept(text):
        return "accept" in text or "接受" in text
    def contains_reject(text):
        return "reject" in text or "拒絕" in text
    
    if contains_accept(address_last_line) and contains_accept(time_last_line) and not contains_reject(address_last_line) and not contains_reject(time_last_line):
        return True
    elif contains_accept(address_response) and contains_accept(time_response) and not contains_reject(address_response) and not contains_reject(time_response):
        return True
    else:
        return False

def generate_fact_statement(input, reference_facts):
    global truth_prompt
    size = len(reference_facts)
    cnt = 0 #生成次數
    while True:
        if cnt % 5 == 0:
            print("參考輸出:\n", reference_facts[int(cnt / 5)%size])
        processed_truth_prompt = f"\n範例格式:{reference_facts[int(cnt / 5)%size]}" + truth_prompt
        input = Tools.remove_input_specific_part(input)
        result = Tools.combine_prompt_generate_response(input, processed_truth_prompt)
        cnt += 1
        #避免提前做結尾或者分段
        if "綜上所述" in result or '\n' in result or "二、" in result or "三、" in result or ("民法" in result and "民法" not in input):
            print(result)
            print("生成格式不符合要求，重新生成")
        #如果輸入和輸出的擷取內容一致則跳出
        elif check_input_output_content(input, result) == False:
            print("判斷輸入和輸出的時間地點或內容不一致，重新生成")
        else:
            break
        print("=" * 50)
    if "一、" not in result:
        result = "一、" + result
    return result
if __name__ == "__main__":
    start_time = time.time()    
    print(generate_fact_statement(user_input, rag_references))
    print("=" * 50)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n執行時間: {hours}h {minutes}m {seconds}s")