from ollama import chat, ChatResponse
import re

class Tools:
    def llm_generate_response(input_data):
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
                model='kenneth85/llama-3-taiwan:8b-instruct-dpo',
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {e}"
    
    def split_user_input(user_input):
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

    def split_user_output(output):
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
    
    def remove_input_specific_part(input):
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
    def combine_prompt_generate_response(input, prompt):
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
        return Tools.llm_generate_response(fact_input) 