from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from functools import lru_cache
from Neo4j_Query import get_type_for_case,get_simoutput_case
from define_case_type import get_case_type
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

if __name__ == "__main__":
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
    print(query_simulation(input_txt, 3))