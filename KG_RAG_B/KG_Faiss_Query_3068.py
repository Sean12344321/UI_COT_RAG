from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from functools import lru_cache

import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
# 加載 .env 配置
load_dotenv()

# Neo4j 配置
uri = os.getenv("NEO4J_URI_3068")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD_3068")
driver = GraphDatabase.driver(uri, auth=(username, password))

# 索引保存路徑
INDEX_PATH = "case_index_3068"

MAX_CACHE_SIZE = 5  # 最多 cache 幾個索引
# 初始化嵌入模型
model = SentenceTransformer("shibing624/text2vec-base-chinese")

def build_faiss_indexes() -> Dict[str, Tuple[faiss.IndexHNSWFlat, List[str], List[str]]]:
    """
    從 Neo4j 數據庫中構建每個 case_type 的 FAISS 索引並保存到磁盤。

    Returns:
        Dict[str, Tuple[faiss.IndexHNSWFlat, List[str], List[str]]]: 每個 case_type 對應的 FAISS 索引，案件 ID 列表，事故緣由文本列表。
    """
    with driver.session() as session:
        # 查詢所有事故發生緣由節點的 ID、文本、嵌入和案件類型
        results = session.run("MATCH (f:事故發生緣由) RETURN f.case_id AS id, f.text AS text, f.embedding AS embedding")
        print(results)
        # 根據 case_type 分組
        data_by_type = {}
        for record in results:
            case_id = record["id"]
            case_type = get_type_for_case(case_id)
            print(case_type)
            if case_type not in data_by_type:
                data_by_type[case_type] = {'embeddings': [], 'case_ids': [], 'reason_texts': []}
            data_by_type[case_type]['case_ids'].append(case_id)
            data_by_type[case_type]['reason_texts'].append(record["text"])
            data_by_type[case_type]['embeddings'].append(np.array(record["embedding"], dtype="float32"))
        
    indexes = {}
    cnt = 0
    for case_type, data in data_by_type.items():
        print(cnt)
        cnt += 1
        embeddings = data['embeddings']
        if not embeddings:
            continue
        print(len(embeddings))
        # 構建 FAISS HNSW 索引
        dimension = len(embeddings[0])
        M = 32  # HNSW 的參數，決定連接數量
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = 200  # 構建時的 ef 值
        index.hnsw.efSearch = 100  # 查詢時的 ef 值
        index.add(np.array(embeddings))  # 添加嵌入向量

        # 創建存儲目錄（如果不存在）
        os.makedirs(INDEX_PATH, exist_ok=True)

        # 保存索引到磁盤
        index_path = os.path.join(INDEX_PATH, f"{case_type}_index.faiss")
        faiss.write_index(index, index_path)
        metadata_path = os.path.join(INDEX_PATH, f"{case_type}_metadata.npy")
        with open(metadata_path, "wb") as f:
            np.save(f, {"case_ids": data['case_ids'], "reason_texts": data['reason_texts']})

        indexes[case_type] = (index, data['case_ids'], data['reason_texts'])

    return indexes

# 使用 LRU cache，最多保留 5 個索引在記憶體中
@lru_cache(maxsize=MAX_CACHE_SIZE)
def load_faiss_index_cached(case_type: str) -> Tuple[faiss.IndexHNSWFlat, List[str], List[str]]:
    index_path = os.path.join(INDEX_PATH, f"{case_type}_index.faiss")
    metadata_path = os.path.join(INDEX_PATH, f"{case_type}_metadata.npy")
    print(index_path)
    print(metadata_path)
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        index = faiss.read_index(index_path)
        metadata = np.load(metadata_path, allow_pickle=True).item()
        return index, metadata["case_ids"], metadata["reason_texts"]
    else:
        indexes = build_faiss_indexes()
        return indexes.get(case_type, (None, [], []))

def query_faiss(input_text: str, case_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    在指定 case_type 的 FAISS 索引中查詢最相似的案件。

    Args:
        input_text (str): 用戶輸入的文本。
        case_type (str): 案件類型。
        top_k (int): 返回的最相似事實數量。

    Returns:
        List[Dict[str, Any]]: 包含最相似事實的 ID、文本和距離的列表。
    """
    print("Encoding input...")
    query_embedding = np.array([model.encode(input_text)], dtype="float32")
    print("Embedding shape:", query_embedding.shape)

    print(f"Loading FAISS index for: {case_type}")
    sim_inputs = []
    index, case_ids, reason_texts = load_faiss_index_cached(case_type)
    if index is None:
        print("Index is None.")
        return []
    print("Performing FAISS search...")
    distances, indices = index.search(query_embedding, top_k)
    print("Search complete.")
    print("Distances:", distances)
    print("Indices:", indices)
    for i, indice in enumerate(indices[0]):
        sim_inputs.append({
            "id": case_ids[indice],
            "text": reason_texts[indice],
            "distance": distances[0][i]
        })
    print("使用faiss搜尋結果:", sim_inputs)
    return sim_inputs
def query_simulation(input_text,top_k):
    # 1. 查詢最相近的 "模擬輸入"
    case_type=get_case_type(input_text)
    print(f"案件類型: {case_type}")
    print(f"在faiss中查詢{top_k}個模擬輸入")
    sim_inputs = query_faiss(input_text, case_type, top_k=top_k)
    # 2. 查詢對應的 "模擬輸出"
    print("在neo4j中找到對應的起訴狀")
    results = []
    for sim_input in sim_inputs:
        results.append({"case_id": sim_input["id"],
                        "case_text": get_simoutput_case(int(sim_input["id"])),
        })
    return results
###
#if __name__ == "__main__":
input_txt = """一、事故發生緣由:
    被告於民國109年9月20日18時30分許，駕駛車牌號碼000-000號自用小客車（A車），行經新北市○○區○○街0段000號前處時，在劃有分向限制的路段，貿然跨越雙黃線逕自迴轉駛入對向車道。此時，原告正駕駛車牌號碼000-0000號普通重型機車（系爭機車）行駛，因被告的違規行為而發生碰撞。事發當時路況良好，被告應注意而未注意，導致此次事故的發生。

    二、原告受傷情形:
    原告因此次碰撞受有左膝部挫傷、左側肩關節扭傷之傷害（系爭傷害）。根據亞東醫院診斷證明書顯示，原告需休養復健4至8週，並需門診複查。原告因傷勢多次就醫，造成生活上的不便，並影響其工作。原告在事故發生後，需要持續就醫治療，並進行復健，對其日常生活和工作能力造成顯著影響。

    三、請求賠償的事實根據:
    1. 醫療費用：原告因系爭傷害支出醫療費用11,067元，有仁愛醫院、亞東醫院、林口長庚醫院、敏盛綜合醫院、健雄診所等醫療機構開立的診斷證明書及醫療費用收據為證。
    2. 交通費用：原告因就醫支出交通費用380元，有停車費用收據為證。
    3. 工作損失：原告因系爭事故受有不能工作之損失24,400元，有診斷證明書及薪資表為證。原告需休養復健4至8週，影響其工作能力。
    4. 機車維修費用：系爭機車因事故受損，支出修復費用共計10,850元，有估價單為證。
    5. 預估醫療費用：原告主張因系爭事故而需支出未來開刀醫療費用50,000元。
    6. 精神慰撫金：原告因系爭傷害多次就醫，造成生活上的不便，請求精神慰撫金300,000元。

    綜上所述，原告依侵權行為之法律關係，請求被告賠償上述各項損失，總計396,697元及法定利息。原告請求自起訴狀繕本送達翌日起至清償日止，按週年利率5%計算之利息。原告認為被告應對此次事故負全部責任，故請求全額賠償。"""
#   print(query_simulation(input_txt, 3))
###

def find_case_type_by_case_id(tx, case_id):
    query = (
        "MATCH (t:案件類型)-[:所屬案件]->(c:案件 {case_id: $case_id}) "
        "RETURN t.name AS case_type"
    )
    result = tx.run(query, case_id=case_id)
    record = result.single()
    if record:
        return record["case_type"]
    else:
        return None


def get_type_for_case(case_id):
    with driver.session() as session:
        case_type = session.execute_read(find_case_type_by_case_id, case_id)
        return case_type
    
def get_simoutput_case(case_id):
    with driver.session() as session:
        simoutput = session.execute_read(find_simoutput_by_case_id, case_id)
        return simoutput
def find_simoutput_by_case_id(tx, case_id):
    query = (
        "MATCH (s:模擬輸出 {case_id: $case_id})"
        "RETURN s.text AS simoutput"
    )
    result = tx.run(query, case_id=case_id)
    record = result.single()
    if record:
        return record["simoutput"]
    else:
        return None
    
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