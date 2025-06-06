from utils import Tools
import re, time
from evaluate import load
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
你要比較括號中的兩個地址是否完全相同，包括城市、地區、路名及門牌號碼等細節。
請依照以下步驟進行逐項檢查：

步驟一：比對「城市」、「地區」、「道路名稱」是否一致。
步驟二：如果門牌號碼有提供，則需完全一致；若未提供門牌號碼，則忽略此項。
步驟三：如果差異僅在「附近」、「對面」、「旁邊」等模糊詞，仍視為一致。
步驟四：若有明確差異（例如不同路名、地區或城市），則輸出「reject」。
步驟五：只有當所有項目符合上述條件時，才輸出「accept」。

請依照以下格式生成輸出：
===========================
[推理過程]:
城市: 
地區:
道路名稱:
門牌號碼:

[判決結果]:
若有任何一項不一致，則輸出「reject」；若全部一致，則輸出「accept」。
===========================
"""
time_truth_check = """
請比對括號中的兩個時間，依照以下步驟進行逐項檢查：

步驟一：比對民國年份，數字相同視為一致。
步驟二：比對月份，數字相同視為一致。
步驟三：比對日期，數字相同視為一致。
步驟四：比對時間（上午/下午/晚上、小時與分鐘），若換算後的數字相同，即視為一致。
步驟五：只有當所有項目符合上述條件時，才輸出「accept」。

請依照以下格式輸出：
===========================
[推理過程]:
年份:
月份:
日期:
時間:

[判決結果]:
若有任何一項不一致，則輸出「reject」；若全部一致，則輸出「accept」。
===========================
"""
user_input = """一、事故發生緣由:
 被告於000年0月00日下午7時許，駕駛車號000-0000號自小貨車，沿雲林縣水林鄉海埔村雲157線（由北往南方向）行駛，行經雲林縣○○鄉○○村○000○○○○○道路○○號誌交岔路口時，疏未注意轉彎車應暫停讓直行車先行，即貿然左轉，適原告騎乘車號000-000號普通重型機車，沿雲林縣水林鄉雲157線（由南往北方向）行駛至該處時，見被告所駕駛之自小貨車突然左轉時閃避不及，原告所騎乘機車車頭與被告所駕駛自小貨車右側車身發生碰撞，原告因而人車倒地。
 按左轉彎時，應距交岔路口三十公尺前顯示方向燈或手勢，換入內側車道或左轉車道，行至交岔路口中心處左轉，並不得占用來車道搶先左轉；汽車行駛至無號誌之交岔路口，轉彎車應暫停讓直行車先行，道路交通安全規則第102條第1項第5、7款分別有明文。被告駕駛車輛，自應注意遵守上開交通規定，且當時天候為晴天，晨或暮光，路面乾燥無缺陷，無障礙物及視距良好，並無不能注意之情事，而被告駕車行經交岔路口時，卻疏未注意轉彎車應暫停讓直行車先行，因此被告應負過失侵權行為之責任。
 
 二、原告受傷情形:
 原告因為此次交通事故受有左側脛骨遠端骨折、左側腓骨外踝骨折、左側腎臟撕裂、左下眼瞼及鼻淚管撕裂、舌部撕裂、左耳鼓膜破損、左側第4第5第6第7肋骨骨折等傷害。
"""

#暫時的參考資料，實際使用時會從資料庫中獲取
rag_references = ['一、緣被告明知其駕駛執照經吊扣，不得駕駛自用小貨車行駛於道路，仍於102年5月31日上午10時15分許，駕駛車號00-0000號自用小貨車，沿雲林縣斗南鎮臺一線由北往南方向行駛，途經同路段南向239.2公里處時，其原應注意汽車行駛變換車道時，讓直行車先行，並注意安全距離。依當時天候晴、日間自然光線、柏油路面乾燥無缺陷、無障礙物，並無不能注意之情事，竟疏於注意及此，適有原告騎乘000-000號普通重型機車同向行駛於被告後方機車道上，致2車發生擦撞，造成原告受傷。', '一、緣被告於民國108年5月31日8時45分許，駕駛其所有車牌號碼000-0000號自小客貨車，沿彰化縣彰化市公園路1段由北往南方向行駛，行經該路段412號前之無號誌交岔路口，欲右轉公園路1段往西方向行駛時，疏未注意車前狀況及讓直行車先行，貿然右轉，適原告騎乘車牌號碼000-000號普通重型機車，沿彰化市公園路1段由東往西方向直行至該路口，兩車因而發生碰撞，原告人車倒地。', '一、緣被告於民國107年7月30日下午,駕駛其所有車牌號碼000-00號自用小貨車,沿彰化縣芳苑鄉海埔路往南行駛,嗣於同日17時5分許,行經海埔路與功湖路無號誌交岔路口時,疏未注意車前狀況,讓直行車先行,竟貿然左轉駛往功湖路,適原告騎乘車牌號碼000-000號普通重型機車,沿功湖路往西方向直行,兩車於上開路口發生碰撞,致原告人車倒地,並受有頭部外傷、左頸部挫傷、左腕擦傷及雙膝挫傷等傷害。', '一、緣被告於民國107年9月7日下午16時50分許，駕駛000-0000號營業大貨車沿臺南市○○區○道○號公路由北向南行駛於道路中線車道，至該路段南向328公里700公尺處，本應注意變換車道時應讓直行車先行，並注意安全距離。當時天候雖為陰天，惟日間有自然光線，柏油路面乾燥、無缺陷、無障礙物、視距良好，並無不能注意之情形，依被告駕駛能力亦應能注意，竟貿然向右變換至外側車道，致擦撞同向平行行駛於外側車道、由原告駕駛之00-0000號自小客車左前方，致該車受撞後衝撞右側路旁護欄反轉逆向才停止。', '一、緣被告於民國108年5月31日7時58分許，駕駛車牌號碼000-0000號自用小客貨車，沿屏東縣恆春鎮中正路239巷39弄由北往南方向行駛，行經該巷與同巷39弄時，被告原應注意轉彎車應讓直行車先行，竟疏未注意而貿然右轉，碰撞由原告騎乘車牌號碼為000-0000號普通重型機車。且當時路口正前方設有反射鏡，該反射鏡反射範圍即為被告左側之原告直行而來之道路情況，顯係特別設置供被告行向之車輛於駛入該路口欲右轉時，注意左側原告行向之來車使用，被告依其行向駛至該路口欲右轉時，客觀上並無不能注意左側有原告來車之情事。']
tools = None

def generate_simple_fact_statement(input, reference_fact, tools):
    """"直接生成, 沒有用chain of though以及summary, 只有結合Input跟rag的reference"""
    global truth_prompt
    truth_prompt = f"\n範例格式:{reference_fact}" + truth_prompt
    input = tools.remove_input_specific_part(input)
    result = tools.combine_prompt_generate_response(input, truth_prompt).replace('\n', '')
    if "一、" not in result:
        result = "一、" + result    
    print("輸出:\n", result)
    yield result
    return result

def generate_summary(input):
    info_dict = {}
    time = 0
    while time < 5:#生成至多5次
        time += 1
        abstract = tools.combine_prompt_generate_response(input, truth_summary)
        matches = re.findall(r"\[(案發地點|案發時間)\]:\s*(.*)", abstract)
        # 轉換為字典
        info_dict = {k: v for k, v in matches}
        # 如果資訊不構成二個欄位則重新生成
        if len(info_dict) == 2:
            break
    if time >= 5:
        return False
    ordered_keys = ["案發地點", "案發時間"]
    reordered = {key: info_dict[key] for key in ordered_keys if key in info_dict}

    return reordered

def check_input_output_content(input, output):
    # 檢查輸入和輸出內容是否一致
    global address_truth_check
    global time_truth_check
    print("輸出:\n", output)
    input_abs = generate_summary(input)
    output_abs = generate_summary(output)
    if input_abs == False or output_abs == False:
        return False
    
    summary_process_text = "輸出:{" + output + "}<br><br>"
    print("輸入摘要:", input_abs)
    summary_process_text += f"輸入摘要:{input_abs}<br><br>"
    print("輸出摘要:", output_abs)
    summary_process_text += f"輸出摘要:{output_abs}<br><br>"
    yield tools.show_summary_to_UI(summary_process_text)

    # judge_process_text = "輸出:<br>" + tools.wrap_debug_section(output, color="#7281e8", border="#7281e8") 
    judge_process_text = ""
    if input_abs["案發地點"] != output_abs["案發地點"]:
        processed_address_truth_check = f"[{input_abs['案發地點']}]，[{output_abs['案發地點']}]" + address_truth_check
        address_response = tools.llm_generate_response(processed_address_truth_check).replace('=', '')
        print("地址檢查:\n", address_response)

        address_block = "地址檢查<br>" + tools.wrap_debug_section(tools.remove_blank_lines(address_response), color="#d0c5bb", border="#d0c5bb")
        judge_process_text += address_block
    else:
        address_response = "accept"
        print("輸入和輸出地址一致，無需檢查")
        address_block = "地址檢查<br>" + tools.wrap_debug_section("輸入和輸出地址一致，無需檢查", color="#d0c5bb", border="#d0c5bb")
        judge_process_text += address_block

    if input_abs["案發時間"] != output_abs["案發時間"]:
        processed_time_truth_check = f"[{input_abs['案發時間']}]，[{output_abs['案發時間']}]" + time_truth_check
        time_response = tools.llm_generate_response(processed_time_truth_check).replace('=', '')
        print("時間檢查:\n", time_response)

        time_block = "時間檢查<br>" + tools.wrap_debug_section(tools.remove_blank_lines(time_response), color="#b3d6c2", border="#b3d6c2")
        judge_process_text += time_block
    else:
        time_response = "accept"
        print("輸入和輸出時間一致，無需檢查")

        time_block = "時間檢查<br>" + tools.wrap_debug_section("輸入和輸出時間一致，無需檢查", color="#b3d6c2", border="#b3d6c2")
        judge_process_text += time_block
    yield tools.show_debug_to_UI(judge_process_text)
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

def generate_fact_statement(input, reference_facts, passed_tools):
    global truth_prompt
    global tools
    tools = passed_tools
    size = len(reference_facts)
    cnt = 0 #生成次數
    result = ""
    history_logs = []
    start_time = time.time()   
    while True:
        if time.time() - start_time > 120:
            print(f"事實陳述生成超過2分鐘，從過去{len(history_logs)}次輸出中選擇最好的一個")
            yield tools.show_final_judge_to_UI(f'<span style="color:#4287f5;">事實陳述生成超過2分鐘，從過去{len(history_logs)}次輸出中選擇最好的一個</span>')
            final_index = select_best_output_using_bert_score(input, history_logs)
            yield tools.show_final_judge_to_UI(f'<span style="color:#4287f5;">選擇第{final_index+1}筆輸出作為事實陳述生成</span>')
            result = history_logs[final_index]
            break
        if cnt == 3:
            print(f"事實陳述嘗試生成3次仍無法通過檢查，從過去3次輸出中選擇最好的一個\n")
            yield tools.show_final_judge_to_UI(f'<span style="color:#4287f5;">事實陳述嘗試生成3次仍無法通過檢查，從過去3次輸出中選擇最好的一個</span>')
            final_index = select_best_output_using_bert_score(input, history_logs)
            yield tools.show_final_judge_to_UI(f'<span style="color:#4287f5;">選擇第{final_index+1}筆輸出作為事實陳述生成</span>')
            result = history_logs[final_index]
            break
        if cnt % 3 == 0:
            print("參考輸出:\n", reference_facts[int(cnt / 3)%size])
            # yield tools.show_debug_to_UI(f"參考輸出:\n, {reference_facts[int(cnt / 5)%size]}")
        processed_truth_prompt = f"\n範例格式:{reference_facts[int(cnt / 3)%size]}" + truth_prompt
        input = tools.remove_input_specific_part(input)
        result = tools.combine_prompt_generate_response(input, processed_truth_prompt).strip('\n')
        #避免提前做結尾或者分段
        if "綜上所述" in result or '\n' in result or "二、" in result or "三、" in result or ("民法" in result and "民法" not in input):
            print("輸出:\n", result, "\n生成格式不符合要求，重新生成")
            # yield tools.show_debug_to_UI(f"輸出:\n{result}\n生成格式不符合要求，重新生成")
        else:
            check_result = yield from check_input_output_content(input, result)
            print("check_result:", check_result)
            if check_result == False:
                print("判斷輸入和輸出的時間地點或內容不一致，重新生成")
                yield tools.show_final_judge_to_UI('<span style="color:#db272d;">最終判決結果: Reject</span>')
                cnt += 1
                history_logs.append(result)
            else:
                print("判斷輸入和輸出的時間地點或內容一致，生成完成")
                yield tools.show_final_judge_to_UI('<span style="color:#2fd119;">最終判決結果: Accept</span>')
                break
        print("=" * 50)
        # yield tools.show_debug_to_UI("=" * 50)
    if "一、" not in result:
        result = "一、" + result
    yield tools.show_result_to_UI(result)
    print(result)
    return result
if __name__ == "__main__":
    start_time = time.time()   
    tools = Tools("kenneth85/llama-3-taiwan:8b-instruct-dpo")
    response = "" 
    for part, reference, summary, log, final_judge in generate_fact_statement(user_input, rag_references, tools):
        pass
    # for part in generate_simple_fact_statement(user_input, rag_references, tools):
    #     pass
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n執行時間: {hours}h {minutes}m {seconds}s")