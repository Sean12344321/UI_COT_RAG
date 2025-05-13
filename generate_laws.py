import os, sys
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "chunk_RAG"))
from chunk_RAG.ts_define_case_type import get_case_type
from utils import Tools
def check_and_generate_laws(user_input, tools, k_value, laws = []):
        retrieval_system = tools.retrieval_system   
        progress_text = """"""
        
        # Count law occurrences
        law_counts = retrieval_system.count_law_occurrences(laws)
        progress_text += "相似案例法條出現頻率:\n"
        for law, count in sorted(law_counts.items(), key=lambda x: x[1], reverse=True):
            progress_text += f"法條 {law}: 出現 {count} 次\n"
        # Determine j based on k value
        k_value = len(laws)
        j_values = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2}
        j = j_values.get(k_value, 1)
        # progress_text += f"根據 k={k_value} 設置法條保留閾值 j={j}\n"
        
        # Filter laws by occurrence threshold
        filtered_law_numbers = retrieval_system.filter_laws_by_occurrence(law_counts, j)
        progress_text += f"符合出現次數 >= {j} 的法條: {filtered_law_numbers}\n"
        
        yield tools.show_debug_to_UI(progress_text.replace("\n", "<br>"))
        progress_text = ""
        # Add law check logic from original code
        # progress_text += "\n進行法條適用性檢查...\n"
        # progress_text += "使用關鍵詞映射生成可能適用的法條...\n"
        user_query = retrieval_system.split_user_query(user_input)
        keyword_laws = retrieval_system.get_laws_by_keyword_mapping(
            user_query['accident_facts'], 
            user_query['injuries'],
            user_query['compensation_facts']
        )
        # progress_text += f"關鍵詞映射生成的法條: {keyword_laws}\n"
        # Compare with filtered laws
        missing_laws = [law for law in keyword_laws if law not in filtered_law_numbers]
        extra_laws = [law for law in filtered_law_numbers if law not in keyword_laws]
        progress_text += f"可能缺少的法條: {missing_laws}\n"
        progress_text += f"可能多餘的法條: {extra_laws}\n"
        yield tools.show_debug_to_UI(progress_text.replace("\n", "<br>"))
        progress_text = ""
        # Check each missing law
        for law_number in missing_laws:      
            progress_text = f"\n檢查缺少的法條 {law_number}...\n"   
            yield tools.show_debug_to_UI(progress_text.replace("\n", "<br>"))
            progress_text = ""
            # Get law content from Neo4j
            law_content = ""
            with retrieval_system.neo4j_driver.session() as session:
                query = """
                MATCH (l:law_node {number: $number})
                RETURN l.content AS content
                """
                result = session.run(query, number=law_number)
                record = result.single()
                if record and record.get("content"):
                    law_content = record["content"]
            
            if not law_content:
                progress_text += tools.wrap_debug_section(f"無法獲取法條 {law_number} 的內容，跳過檢查\n", color='#d0c5bb', border='#d0c5bb')
                yield tools.show_debug_to_UI(progress_text.replace("\n", "<br>"))
                progress_text = ""
                continue
            
            # Check if the law is applicable
            check_result = retrieval_system.check_law_content(
                user_query['accident_facts'],
                user_query['injuries'],
                law_number,
                law_content
            )
            
            progress_text += f"法條 {law_number} 檢查結果: {check_result['result']}\n"
            progress_text += tools.wrap_debug_section(f"原因: {check_result['reason']}\n", color='#d0c5bb', border='#d0c5bb')
            
            # Add to filtered laws if applicable
            if check_result['result'] == 'pass':
                progress_text += f"添加法條 {law_number} 到適用法條列表\n"
                filtered_law_numbers.append(law_number)
                # Sort the list again
                filtered_law_numbers = sorted(filtered_law_numbers)
            
            yield tools.show_debug_to_UI(progress_text.replace("\n", "<br>"))
        # Check each extra law
        for law_number in extra_laws:
            progress_text += f"\n檢查可能多餘的法條 {law_number}...\n"
            
            yield tools.show_debug_to_UI(progress_text.replace("\n", "<br>"))
            progress_text = ""
            # Get law content
            law_content = ""
            with retrieval_system.neo4j_driver.session() as session:
                query = """
                MATCH (l:law_node {number: $number})
                RETURN l.content AS content
                """
                result = session.run(query, number=law_number)
                record = result.single()
                if record and record.get("content"):
                    law_content = record["content"]
            
            if not law_content:
                progress_text += tools.wrap_debug_section(f"無法獲取法條 {law_number} 的內容，跳過檢查\n", color='#d0c5bb', border='#d0c5bb')
                yield tools.show_debug_to_UI(progress_text.replace("\n", "<br>"))
                progress_text = ""
                continue
            
            # Check if the law is applicable
            check_result = retrieval_system.check_law_content(
                user_query['accident_facts'],
                user_query['injuries'],
                law_number,
                law_content
            )
            
            progress_text += f"法條 {law_number} 檢查結果: {check_result['result']}\n"
            progress_text += tools.wrap_debug_section(f"原因: {check_result['reason']}\n", color='#d0c5bb', border='#d0c5bb')
            
            # Remove from filtered laws if not applicable
            if check_result['result'] == 'fail':
                progress_text += f"從適用法條列表中移除法條 {law_number}\n"
                filtered_law_numbers.remove(law_number)
            yield tools.show_debug_to_UI(progress_text.replace("\n", "<br>"))
            progress_text = ""
        # Filter out duplicates and sort
        filtered_law_numbers = sorted(list(set(filtered_law_numbers)))
        progress_text = f"\n最終適用法條列表: {filtered_law_numbers}\n\n"
        yield tools.show_debug_to_UI(progress_text.replace("\n", "<br>"))
        # Generate the final law section
        law_section = "二、按「"
        if filtered_law_numbers:
            # Get law contents
            law_contents = retrieval_system.get_law_contents(filtered_law_numbers)
            
            for i, law in enumerate(law_contents):
                content = law["content"]
                if "：" in content:
                    content = content.split("：")[1].strip()
                elif ":" in content:
                    content = content.split(":")[1].strip()
                
                if i > 0:
                    law_section += "、「"
                law_section += content
                law_section += "」"
            
            law_section += "民法第"
            for i, law in enumerate(law_contents):
                if i > 0:
                    law_section += "、第"
                law_section += law["number"]
                law_section += "條"
            
            law_section += "分別定有明文。查被告因上開侵權行為，使原告受有下列損害，依前揭規定，被告應負損害賠償責任："
        else:
            law_section += "NO LAW"
        yield tools.show_result_to_UI(law_section.replace("\n", "<br>"))

user_input = """一、事故發生緣由:
被告於108年9月12日早上8時36分許，騎乘車牌號碼000-000號普通重型機車，沿新北市中和區華中橋行駛時，本應注意變換車道時，應讓直行車先行，並注意安全距離，竟疏未注意而貿然變換車道，致擦撞右側原告所騎乘之車牌000-0000號普通重型機車（下稱系爭車輛），造成原告人車倒地。又被告因上開過失駕駛行為，經鈞院109年度審交易字第672號、臺灣高等法院109年度交上易字第412號判決，均認犯過失致重傷罪，足證被告針對本件事故之發生具有過失。

二、原告受傷情形:
本件事故使原告受有右鎖骨骨折、四肢多處挫傷、右側遠端鎖骨骨折、右側股骨外髁軟骨骨折、右臂神經叢損傷及右上肢失去功能等重傷害（下稱系爭傷害）。

三、請求賠償的事實根據:
按民法第184條第1項前段、第191條之2本文、第193條第1項、第195條第1項前段，請求下列損害
原告主張自系爭車禍發生已支出聯合醫院醫療費用460元、臺大醫院醫療費用8萬1,356元、長庚醫院醫療費用2,290元、高雄義大醫院108年11月22日至109年4月24日醫療費用47萬6,103元、高雄義大醫院109年7月17日至111年12月16日醫療費用5萬3,804元、高雄義大醫院112年3月10日至112年6月9日醫療費用4,280元、義大癌治療醫院112年9月1日醫療費用50元、義大癌治療醫院112年11月24日醫療費用50元，以上均有門診收據可證，以及原告因有持續復健需求，故支出物理矯正治療費用4萬5,500元，並有永春物理治療所治療收據可證。原告因系爭車禍事件所受之傷害，導致其生活無法自理，於108年9月12日至臺大醫院急診入院住院至108年9月26日（共計15日），再於108年11月24日至義大醫院住院進行左側第七頸椎神經至右手上臂神經顯微移植術、右胸肋間神經轉移至右上臂肌皮神經顯微移植術，於108年12月7日出院（共計14日），且根據義大醫院診斷證明書所載原告出院後「宜有專人協助生活照顧至少3個月」，故自車禍發生起由專人及家人照護共計178日，以全日照護費用每日2,200元計算，看護費用一共39萬1,600元。原告前往高雄義大醫院就醫支出之交通費用、高速公路過路費用、油資共4萬9,299元。系爭車禍導致機車受損，修理費用為5萬0,250元，且均為零件費用，此有山維修估價單為證。另查原告住院起至出院後，共需專人協助生活照料至少3個月已如上述，是原告因系爭事故不能工作之日數應為前開醫囑所載之期間即系爭事故發生日108年9月12日至109年3月7日，共計176日，又原告於系爭事故發生前之平均每月薪資為4萬9,791元，因此原告因系爭傷害後無法工作，而受有176日薪資損失即29萬2,160元（計算式：1,660元×176日＝29萬2,160元）。另外原告因本件事故經醫院鑑定勞動能力減損76%，查原告係00年0月00日出生，其勞動能力減損應自事發翌日（即108年9月13日）起算至其法定退休年齡65歲（即152年4月23日），共計43年7月10日。是依每月3萬元勞動能力減損76％計算，原告每月減損勞動能力之損失為2萬2,800元（計算式：3萬元×76％＝2萬2,800元），依霍夫曼式計算法扣除中間利息（首期給付不扣除中間利息）其勞動力減損金額為新臺幣633萬9,232元。並且原告至少需復健5年，未來需支出復健費用51萬元。原告另請求其支出診斷證明書費用、影印費用、鑑定報告書費用共計1萬2,966元。查本件原告於事發時為21歲，職業為工程師，卻因被告之過失行為致受有系爭傷害，右臂神經叢損傷與右上肢失去功能，目前右上肢嚴重萎縮且領有身心障礙手冊，已嚴重影響生活及未來職涯，已如前述，足見原告身心確受有相當程度痛苦，因此請求慰撫金50萬元。
"""

if __name__ == "__main__":
    # Initialize the tools and retrieval system
    tools = Tools("kenneth85/llama-3-taiwan:8b-instruct-dpo")
    # Set the k_value for the search
    k_value = 3  # Example value, adjust as needed

    # Call the check_laws function with user input
    for part, ref, summary, audit, final_judge in check_and_generate_laws(user_input, tools, k_value):
        print(audit)
        if part != "":
            print(part)