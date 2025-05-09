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
        sub_sections = sections[1].split('（一）')
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
    
    def generate_laws(self, case_ids, threshold):
        laws = self.retrieval_system.get_laws_from_neo4j(case_ids)
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