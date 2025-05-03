import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
def get_case_type(sim_input: str) -> str:
    """
    根據模擬輸入文本 sim_input 判斷案件的類型。
    會依據原被告的人數，以及是否涉及未成年人、僱用人責任或動物責任等因素來組合案型說明。

    Args:
        sim_input (str): 用戶輸入的案件描述文字。

    Returns:
        str: 案件的類型描述（例如 "數名被告+§187未成年案型"）
    """
    case_info = generate_filter(sim_input)
    # 分割姓名列表
    # 正則表達式提取原告和被告姓名
    pattern = r"原告:([\u4e00-\u9fa5A-Za-z0-9○·．,、]+)"
    plaintiff_match = re.search(pattern, case_info)
    pattern = r"被告:([\u4e00-\u9fa5A-Za-z0-9○·．,、]+)"
    defendant_match = re.search(pattern, case_info)

    plaintiffs = re.split(r"[,、]", plaintiff_match.group(1)) if plaintiff_match else []
    defendants = re.split(r"[,、]", defendant_match.group(1)) if defendant_match else []

    # 去除空格
    plaintiffs = [name.strip() for name in plaintiffs]
    defendants = [name.strip() for name in defendants]

    #print("原告:", plaintiffs)
    #print("被告:", defendants)

    case_type=""
    p=len(plaintiffs)
    d=len(defendants)
    # 根據人數分類基本案型
    if p<=1 and d<=1:
        case_type="單純原被告各一"
    elif p>1 and d<=1:
        case_type="數名原告"
    elif p<=1 and d>1:
        case_type="數名被告"
    elif p>1 and d>1:
        case_type="原被告皆數名"

    match = re.search(r'被告是否為未成年人(.*?)被告是否為受僱人(.*?)車禍是否由動物造成(.*)', case_info, re.S)

    if match.group(1).strip()[1] =="是":
        case_type += "+§187未成年案型"
        return case_type
    if match.group(2).strip()[1] =="是":
        case_type += "+§188僱用人案型"
        return case_type
    if match.group(3).strip()[1] =="是":
        case_type += "+§190動物案型"
        return case_type
    
    return case_type
# 主函式：根據模擬輸入，回傳清洗後的描述（包含原被告姓名、是否為未成年、是否為受僱人、是否由動物造成）
def generate_filter(sim_input: str) -> str:
    match = re.search(r'一、(.*?)二、(.*?)三、(.*)', sim_input, re.S)
    if match:
        user_input = match.group(1).strip()
    else:
        raise ValueError(f"未能從輸入中擷取 case 類型內容，輸入為：{sim_input}")
    filted=get_people(user_input)+"\n"+get_187(user_input)+"\n"+get_188(user_input)+"\n"+get_190(user_input)+"\n"
    return filted
# 判斷是否為未成年人 (§187)
def get_187(user_input: str) -> str:
    llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct-dpo-q8_0",
                    temperature=0,
                    keep_alive=0,
                    )
    # 創建 LLMChain
    # 定義提示模板
    prompt_template = PromptTemplate(
        input_variables=["reason"],
        template="""
    請你幫我從以下車禍案件的事故詳情中判斷被告是否為未成年人，並只能用以下格式輸出:
    被告是否為未成年人:(是/否)

    以下是本起車禍的事故詳情：
    {reason}
    備註:
    如果未提及被告的年齡就判斷為否
    你只需要告訴我被告是不是未成年人，請依照格式輸出，不要輸出其他多餘的內容
    輸出時記得按照格式在是或否前加上:"被告是否為未成年人"
    """
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    # 傳入數據生成起訴書
    filtered_input = llm_chain.run({
        "reason" : user_input
    })
    #print(filtered_input)
    return filtered_input
# 判斷是否為受僱人 (§188)
def get_188(user_input: str) -> str:
    llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct-dpo-q8_0",
                    temperature=0,
                    keep_alive=0,
                    )
    # 創建 LLMChain
    # 定義提示模板
    prompt_template = PromptTemplate(
        input_variables=["reason"],
        template="""
    請你幫我從以下車禍案件的事故詳情中判斷被告在車禍發生時是否為正在執行職務的受僱人，並只能用以下格式輸出:
    被告是否為受僱人:(是/否)

    以下是本起車禍的事故詳情：
    {reason}
    備註:
    如果未提及被告是否為正在執行職務的受僱人就判斷為否
    你只需要告訴我被告是不是受僱人，請依照格式輸出，不要輸出其他多餘的內容
    輸出時記得按照格式在是或否前加上:"被告是否為受僱人:"
    """
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    # 傳入數據生成起訴書
    filtered_input = llm_chain.run({
        "reason" : user_input
    })
    #print(filtered_input)
    return filtered_input
# 判斷是否為動物造成 (§190)
def get_190(user_input: str) -> str:

    llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct-dpo-q8_0",
                    temperature=0,
                    keep_alive=0,
                    )
    # 創建 LLMChain
    # 定義提示模板
    prompt_template = PromptTemplate(
        input_variables=["reason"],
        template="""
    請你幫我從以下車禍案件的事故詳情中判斷車禍是否由動物造成，並只能用以下格式輸出:
    車禍是否由動物造成:(是/否)

    以下是本起車禍的事故詳情：
    {reason}
    備註:
    如果未提及車禍是否由動物造成就判斷為否
    你只需要告訴我車禍是否由動物造成，請依照格式輸出，不要輸出其他多餘的內容
    輸出時記得按照格式在是或否前加上:"車禍是否由動物造成"
    """
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    # 傳入數據生成起訴書
    filtered_input = llm_chain.run({
        "reason" : user_input
    })
    #print(filtered_input)
    return filtered_input
# 擷取原告與被告姓名
def get_people(user_input: str) -> str:
    llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct-dpo-q8_0",
                    temperature=0,
                    keep_alive=0,
                    )
    # 創建 LLMChain
    # 定義提示模板
    prompt_template = PromptTemplate(
        input_variables=["reason"],
        template="""
    請你幫我從以下車禍案件的事故詳情中提取並列出所有原告和被告的姓名，並只能用以下格式輸出:
    原告:原告1,原告2...
    被告:被告1,被告2...

    以下是本起車禍的事故詳情：
    {reason}
    備註:
    如果未提及原告或被告的姓名或代稱需寫為"未提及"
    你只需要列出原告和被告的姓名，請不要輸出其他多餘的內容
    """
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    # 傳入數據生成起訴書
    filtered_input = llm_chain.run({
        "reason" : user_input
    })
    #print(filtered_input)
    return filtered_input
#print(generate_filter(user_input))