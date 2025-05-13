from ollama import chat, ChatResponse
import re, os, sys
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "chunk_RAG"))
sys.path.append(os.path.join(os.path.dirname(__file__), "KG_RAG_B"))
from chunk_RAG.ts_retrieval_system import RetrievalSystem
class Tools:
    def __init__(self, model):
        self.model = model
        self.retrieval_system = RetrievalSystem()   
    def llm_generate_response(self, input_data):
        """
        use LLM to generate response
        Args:
            input_data (str): input data to generate response
        Returns:
            response (str): generated response
        """
        try:
            response: ChatResponse = chat(
                messages=[
                    {
                        'role': 'user',
                        'content': input_data,
                    },
                ],
                model=self.model,
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {e}"
    
    def split_user_input(self, user_input):
        """
        split user input by 一、二 and 三
        Args:
            user_input (str): input to split
        Returns:
            Dictionary of sections(case_facts, injury_details, compensation_request)
        """
        sections = re.split(r"(一、|二、|三、)", user_input)
        input_dict = {
            "case_facts": sections[2].strip(),
            "injury_details": sections[4].strip(),
            "compensation_request": sections[6].strip()
        }
        return input_dict

    def split_user_output(self, output):
        """
        split lawsheet by 一、二 and 綜上所陳
        Args:
            output (str): output to split
        Returns:
            Dictionary of sections(fact, law, compensation)
        """
        sections = output.split('二、')
        if len(sections) != 2:
            print("警告: 無法正確識別文本標記。請確保格式為「一、」開頭，然後有「二、」和(一)")
            return False
        if '（一）' in sections[1]:
            sub_sections = sections[1].split('（一）')
        elif '(一)' in sections[1]:
            sub_sections = sections[1].split('(一)')
        else:
            sub_sections = [sections[1]] 
        if len(sub_sections) != 2:
            print("警告: 無法正確識別文本標記。請確保格式為「二、」和「（一）」")
            return False
        reference_fact = sections[0].strip()
        reference_law = '二、' + sub_sections[0].strip()
        reference_compensation = ' (一) ' + sub_sections[1].strip()
        return {
            "fact": reference_fact,
            "law": reference_law,
            "compensation": reference_compensation
        }
    
    def remove_input_specific_part(self, input):
        """
        remove「一、事故發生緣由:」and「二、原告受傷情形:」
        Args:
            input (str): input text to clean
        Returns:
            str: cleaned text
        """
        parts = re.split(r'一、事故發生緣由[:：]', input)
        if len(parts) > 1:
            input = parts[1]  # 把後段當作新的 input
        # 再切割「二、原告受傷情形：」
        parts = re.split(r'二、原告受傷情形[:：]', input)
        if len(parts) > 1:
            input = parts[0] + parts[1]  # 把後段當作最終的 text（繼續保留後續內容）
        return input.strip().replace('\n', '')
    
    def combine_prompt_generate_response(self, input, prompt):
        """
        combine input and prompt to generate response
        Args:
            input (str): input text
            prompt (str): prompt text
        Returns:
            str: generated fact
        """
        fact_input = f"""輸入:
        {input}
        {prompt}"""
        return self.llm_generate_response(fact_input)
    
    def extract_cases_to_laws(self, cases: list) -> dict:
        print("[DEBUG] laws before extraction:", cases)
        print("[DEBUG] type of laws:", type(cases))
        laws = []
        laws_content = [{"law_number": "184", "content": "第184條:因故意或過失，不法侵害他人之權利者，負損害賠償責任。故意以背於善良風俗之方法，加損害於他人者亦同。 違反保護他人之法律，致生損害於他人者，負賠償責任。但能證明其行為無過失者，不在此限。"},{"law_number": "185", "content": "第185條:數人共同不法侵害他人之權利者，連帶負損害賠償責任。不能知其中孰為加害人者亦同。 造意人及幫助人，視為共同行為人。"}, {"law_number": "187", "content": "第187條:無行為能力人或限制行為能力人，不法侵害他人之權利者，以行為時有識別能力為限，與其法定代理人連帶負損害賠償責任。行為時無識別能力者，由其法定代理人負損害賠償責任。 前項情形，法定代理人如其監督並未疏懈，或縱加以相當之監督，而仍不免發生損害者，不負賠償責任。 如不能依前二項規定受損害賠償時，法院因被害人之聲請，得斟酌行為人及其法定代理人與被害人之經濟狀況，令行為人或其法定代理人為全部或一部之損害賠償。 前項規定，於其他之人，在無意識或精神錯亂中所為之行為致第三人受損害時，準用之。"}, {"law_number": "188", "content": "第188條:受僱人因執行職務，不法侵害他人之權利者，由僱用人與行為人連帶負損害賠償責任。但選任受僱人及監督其職務之執行，已盡相當之注意或縱加以相當之注意而仍不免發生損害者，僱用人不負賠償責任。 如被害人依前項但書之規定，不能受損害賠償時，法院因其聲請，得斟酌僱用人與被害人之經濟狀況，令僱用人為全部或一部之損害賠償。 僱用人賠償損害時，對於為侵權行為之受僱人，有求償權。 前項損害之發生，如別有應負責任之人時，賠償損害之所有人，對於該應負責者，有求償權。"}, {"law_number": "191-2", "content": "第191-2條:汽車、機車或其他非依軌道行駛之動力車輛，在使用中加損害於他人者，駕駛人應賠償因此所生之損害。但於防止損害之發生，已盡相當之注意者，不在此限。"}, {"law_number": "193", "content": "第193條:不法侵害他人之身體或健康者，對於被害人因此喪失或減少勞動能力或增加生活上之需要時，應負損害賠償責任。 前項損害賠償，法院得因當事人之聲請，定為支付定期金。但須命加害人提出擔保。"}, {"law_number": "195", "content": "第195條:不法侵害他人之身體、健康、名譽、自由、信用、隱私、貞操，或不法侵害其他人格法益而情節重大者，被害人雖非財產上之損害，亦得請求賠償相當之金額。其名譽被侵害者，並得請求回復名譽之適當處分。 前項請求權，不得讓與或繼承。但以金額賠償之請求權已依契約承諾，或已起訴者，不在此限。 前二項規定，於不法侵害他人基於父、母、子、女或配偶關係之身分法益而情節重大者，準用之。"}, {"law_number": "213", "content": "第213條:負損害賠償責任者，除法律另有規定或契約另有訂定外，應回復他方損害發生前之原狀。 因回復原狀而應給付金錢者，自損害發生時起，加給利息。 第一項情形，債權人得請求支付回復原狀所必要之費用，以代回復原狀。"}, {"law_number": "216", "content": "第216條:損害賠償，除法律另有規定或契約另有訂定外，應以填補債權人所受損害及所失利益為限。 依通常情形，或依已定之計劃、設備或其他特別情事，可得預期之利益，視為所失利益。"}, {"law_number": "217", "content": "第217條:損害之發生或擴大，被害人與有過失者，法院得減輕賠償金額，或免除之。重大之損害原因，為債務人所不及知，而被害人不預促其注意或怠於避免或減少損害者，為與有過失。 前二項之規定，於被害人之代理人或使用人與有過失者，準用之。"}]
        for input in cases:
            if "第184條" in input:
                laws.append({"law_number": laws_content[0]["law_number"], "content": laws_content[0]["content"]})
            if "第185條" in input:
                laws.append({"law_number": laws_content[1]["law_number"], "content": laws_content[1]["content"]})
            if "第187條" in input:
                laws.append({"law_number": laws_content[2]["law_number"], "content": laws_content[2]["content"]})
            if "第188條" in input:
                laws.append({"law_number": laws_content[3]["law_number"], "content": laws_content[3]["content"]})
            if "第191條" in input:
                laws.append({"law_number": laws_content[4]["law_number"], "content": laws_content[4]["content"]})
            if "第193條" in input:
                laws.append({"law_number": laws_content[5]["law_number"], "content": laws_content[5]["content"]})
            if "第195條" in input:
                laws.append({"law_number": laws_content[6]["law_number"], "content": laws_content[6]["content"]})
            if "第213條" in input:
                laws.append({"law_number": laws_content[7]["law_number"], "content": laws_content[7]["content"]})
            if "第216條" in input:
                laws.append({"law_number": laws_content[8]["law_number"], "content": laws_content[8]["content"]})
            if "第217條" in input:
                laws.append({"law_number": laws_content[9]["law_number"], "content": laws_content[9]["content"]})
        return laws
    
    def generate_laws(self, laws, threshold):
        law_counts = self.retrieval_system.count_law_occurrences(laws)
        filtered_law_numbers = self.retrieval_system.filter_laws_by_occurrence(law_counts, threshold)
        law_contents = []
        if filtered_law_numbers:
            law_contents = self.retrieval_system.get_law_contents(filtered_law_numbers)
        law_section = "二、按「"
        if law_contents:
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
        return law_section

    def show_result_to_UI(self, result_data):
        """
        generate to UI
        Args:
            result_data (str): generated result
        Returns:
            str: formatted string for UI
        """
        return result_data, '', '', '', ''
    
    def show_reference_to_UI(self, reference_data):
        """
        generate to UI
        Args:
            reference_data (str): generated result
            debug_data (str): debug data
        Returns:
            str: formatted string for UI
        """
        return '', reference_data, '', '', ''
    
    def show_summary_to_UI(self, summary_data):
        """
        generate to UI
        Args:
            summary_data (str): generated result
        Returns:
            str: formatted string for UI
        """
        return '', '', summary_data, '', ''

    def show_debug_to_UI(self, debug_data):
        """
        generate to UI
        Args:
            debug_data (str): debug data
        Returns:
            str: formatted string for UI
        """
        return '', '', '', debug_data, ''
    
    def show_final_judge_to_UI(self, final_judge_data):
        """
        generate to UI
        Args:
            final_judge_data (str): final judge data
        Returns:
            str: formatted string for UI
        """
        return '','', '', '', final_judge_data
    def wrap_debug_section(self, content, color="#f9f9f9", border="#ccc", font_color="#000"):
        """
        Wrap debug section with HTML
        Args:
            content (str): Content of the section
            color (str): Background color of the section
            border (str): Border color of the section
            font_color (str): Font color of the section
        Returns:
            str: HTML formatted string  
        """
        return f"""<div style="background-color:{color}; color:{font_color}; border:1px solid {border}; padding:8px; border-radius:5px;">{content}</div>"""
    def remove_blank_lines(self, text):
        lines = text.splitlines()
        non_blank_lines = [line for line in lines if line.strip() != '']
        return '\n'.join(non_blank_lines)