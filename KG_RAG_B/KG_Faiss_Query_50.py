from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from functools import lru_cache
from Neo4j_Query import get_type_for_case
# 加載 .env 配置
load_dotenv()

# Neo4j 配置
uri = os.getenv("NEO4J_URI_50")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD_50")
driver = GraphDatabase.driver(uri, auth=(username, password))

# 索引保存路徑
INDEX_PATH = "case_index_50"
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
        
        # 根據 case_type 分組
        data_by_type = {}
        for record in results:
            case_id = record["id"]
            case_type = get_type_for_case(case_id)
            if case_type not in data_by_type:
                data_by_type[case_type] = {'embeddings': [], 'case_ids': [], 'reason_texts': []}
            data_by_type[case_type]['case_ids'].append(case_id)
            data_by_type[case_type]['reason_texts'].append(record["text"])
            data_by_type[case_type]['embeddings'].append(np.array(record["embedding"], dtype="float32"))

    indexes = {}
    for case_type, data in data_by_type.items():
        embeddings = data['embeddings']
        if not embeddings:
            continue

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
    query_embedding = np.array([model.encode(input_text)], dtype="float32")
    index, case_ids, reason_texts = load_faiss_index_cached(case_type)

    if index is None:
        return []

    distances, indices = index.search(query_embedding, top_k)
    results = []

    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "id": case_ids[idx],
            "text": reason_texts[idx],
            "distance": dist
        })
    return results


user_input="""
text: "事故發生緣由:
 原告於106年8月5日凌晨騎乘車牌號碼000-000號普通重型機車（下稱A車），沿臺北市松山區南京東路5段由西往東方向之第4車道行駛，因訴外人康家誠將車牌號碼000-0000號自用小客車（下稱C車）違規停放於同路段116號前紅線處，遮蔽原告視線，致無法看見斜停於C車前方王俊傑所駕駛且正在執行拖吊機車業務之車牌號碼000-0000號拖吊車（下稱B車）將車尾拖板起落架放下，且王俊傑亦未於車後適當位置設置警告標誌，導致原告行經C車後，始發現B車之起落架放置於原告行進路線上，不及閃避而發生擦撞（下稱系爭事故）。
 又王俊傑受僱於王俊楠即楠德車業工作室，而B車車身上漆有「TMS」字樣（即全鋒公司之英文名稱縮寫），且係因執行拖吊業務而發生系爭事故，王俊楠、全鋒公司均應分別就王俊傑之侵權行為負僱用人責任。"
"""