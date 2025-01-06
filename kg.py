from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import os, re, faiss
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
load_dotenv()

# Neo4j 配置
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))

# 索引保存路徑
INDEX_PATH = "fact_index_hnsw.faiss"

# 初始化嵌入模型
model = SentenceTransformer("shibing624/text2vec-base-chinese")

# 構建 FAISS 索引
def build_faiss_index():
    with driver.session() as session:
        results = session.run("MATCH (f:Fact) RETURN f.id AS id, f.text AS text, f.embedding AS embedding")
        embeddings = []
        fact_ids = []
        fact_texts = []

        for record in results:
            fact_ids.append(record["id"])
            fact_texts.append(record["text"])
            embeddings.append(np.array(record["embedding"], dtype="float32"))

    dimension = len(embeddings[0])
    M = 32
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 100
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)
    with open("fact_metadata_hnsw.npy", "wb") as f:
        np.save(f, {"fact_ids": fact_ids, "fact_texts": fact_texts})

    return index, fact_ids, fact_texts

# 加載 FAISS 索引
def load_faiss_index():
    if os.path.exists(INDEX_PATH) and os.path.exists("fact_metadata_hnsw.npy"):
        index = faiss.read_index(INDEX_PATH)
        metadata = np.load("fact_metadata_hnsw.npy", allow_pickle=True).item()
        return index, metadata["fact_ids"], metadata["fact_texts"]
    else:
        return build_faiss_index()

# 查詢 FAISS
def query_faiss(input_text, top_k=5):
    query_embedding = np.array([model.encode(input_text)], dtype="float32")
    index, fact_ids, fact_texts = load_faiss_index()
    print(len(query_embedding[0]))
    print(index.d)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "id": fact_ids[idx],
            "text": fact_texts[idx],
            "distance": dist
        })
    return results

# 查詢引用法條
def get_statutes_for_case(fact_id):
    with driver.session() as session:
        results = session.run(
            """
            MATCH (c:Case)-[:案件事實]->(f:Fact {id: $fact_id})
            MATCH (c)-[:案件相關法條]->(l:LegalReference)
            MATCH (l)-[:引用法條]->(s:Statute)
            RETURN c.id AS case_id, collect(s.id) AS statutes
            """,
            fact_id=fact_id
        )
        return [{"case_id": record["case_id"], "statutes": record["statutes"]} for record in results]

# 構建 FAISS 索引
def build_faiss_index():
    with driver.session() as session:
        results = session.run("MATCH (f:Fact) RETURN f.id AS id, f.text AS text, f.embedding AS embedding")
        embeddings = []
        fact_ids = []
        fact_texts = []

        for record in results:
            fact_ids.append(record["id"])
            fact_texts.append(record["text"])
            embeddings.append(np.array(record["embedding"], dtype="float32"))

    dimension = len(embeddings[0])
    M = 32
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 100
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)
    with open("fact_metadata_hnsw.npy", "wb") as f:
        np.save(f, {"fact_ids": fact_ids, "fact_texts": fact_texts})

    return index, fact_ids, fact_texts

# 函數：生成引用的法條
def generate_legal_references(case_facts, injury_details):
    input_text = f"{case_facts} {injury_details}"
    similar_facts = query_faiss(input_text, top_k=5)
    statutes_set = set()

    for fact in similar_facts:
        fact_id = fact["id"]
        statutes_info = get_statutes_for_case(fact_id)
        for info in statutes_info:
            statutes_set.update(info["statutes"])

    legal_references = "\n".join(sorted(statutes_set))
    return legal_references

# 定義提示模板
prompt_template = PromptTemplate(
    input_variables=["case_facts", "injury_details", "compensation_request", "legal_references"],
    template="""
你是一個台灣原告律師，你要撰寫一份車禍起訴狀，請根據下列格式進行輸出，並確保每個段落內容完整：
（一）事實概述：完整描述事故經過，事件結果及要求賠償盡量越詳細越好
（二）法律依據：先對每一條我給你的引用法條做判斷，如過確定在這起案件中要引用，列出所有相關法律條文，並對每一條文做出詳細解釋與應用。
  模板：
  - 民法第xxx條第x項：「...法律條文...」。
    - 案件中的應用：本條適用於 [事實情節]，因為 [具體行為] 屬於 [法條描述的範疇]，因此 [解釋為何負責賠償]。
（三）損害項目：列出所有損害項目的金額，並說明對應事實。
  模板：
    損害項目名稱： [損害項目描述]
    金額： [金額數字] 元
    事實根據： [描述此損害項目的原因和依據]
（四）總賠償金額：需要將每一項目的金額列出來並總結所有損害項目，計算總額，並簡述賠償請求的依據。
  模板:
    損害項目總覽：
    總賠償金額： [總金額] 元
    賠償依據：
    依據 [法律條文] 規定，本案中 [被告行為] 對原告造成 [描述損害]，被告應負賠償責任。總賠償金額為 [總金額] 元。
### 案件事實：
{case_facts}
### 受傷情形：
{injury_details}
### 引用法條：
{legal_references}
### 賠償請求：
{compensation_request}
"""
)
user_input="""
一、事故發生緣由:
民國97年10月27日17時37分左右，原告騎乘車牌號碼FZX-900號重型機車，在新莊市○○街往中正路方向行駛。當時有一輛不明小客車跨越分向線，原告為閃避而向右偏駛。同時，對向又有一輛由訴外人黃筱娟駕駛的車輛也跨越分向線迎面而來。原告為避免碰撞，在行車路線受到干擾的情況下，於17時37分5秒左右，原告機車的把手煞車握桿擦撞到被告所有的車牌號碼5265-GX號自用小貨車。該貨車由被告僱用的吳進成違規停放在新莊市○○街41巷巷口，停在劃有紅色實線的禁止臨時停車處，且車尾突出到明中街往中正路方向的車道內。這導致原告的機車失控，人車倒地。

二、原告受傷情形:
原告因這起事故受有右側額葉顱內出血、右側額葉硬膜出血、顱骨骨折、第6、7頸椎棘突骨折、癲癇症等傷害。原告主張這些傷害造成他腦部受傷併發癲癇，無法勝任原來的工作，且以他的學歷和工作經歷，也無法從事簡單的內勤工作。原告認為這符合勞工保險殘廢給付標準中的「精神遺存顯著障害，終身祇能從事輕便工作者」，殘廢等級為7級，喪失勞動能力的比率應為69.21%。

三、請求賠償的事實根據:
原告請求被告賠償的項目包括:
1. 勞動能力減少的損害300萬元：原告主張因腦部受傷併發癲癇，無法勝任原工作，也無法從事簡單內勤工作。
2. 精神慰撫金100萬元：原告主張正值壯年卻因車禍造成腦部受傷併發癲癇，須終身服藥控制病情，造成社會功能下降，身心受創、痛苦不堪。

原告認為被告應為其僱用的吳進成的侵權行為負損害賠償責任。原告強調自己在行駛過程中並無過失，是為了閃避對向侵入車道的汽車才向右偏轉，無法閃避被告違規停放且突出於車道上的貨車。原告主張被告的違規停車行為與車禍有直接因果關係，應負全部責任。
"""

def split_input(user_input):
    sections = re.split(r"(一、|二、|三、)", user_input)
    input_dict = {
        "case_facts": sections[2].strip(),
        "injury_details": sections[4].strip(),
        "compensation_request": sections[6].strip()
    }
    return input_dict

def generate_lawsuit(user_input):
    input_data=split_input(user_input)
    legal_references = generate_legal_references(input_data["case_facts"], input_data["injury_details"])
    input_data["legal_references"] = legal_references
    llm = OllamaLLM(model="kenneth85/llama-3-taiwan:8b-instruct-dpo",
                    temperature=0.1,
                    keep_alive=0,
                    num_predict=50000,
                    )
    # 創建 LLMChain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    print(legal_references)
    # 傳入數據生成起訴書
    lawsuit_draft = llm_chain.run({
        "case_facts": input_data["case_facts"],
        "injury_details": input_data["injury_details"],
        "legal_references": legal_references,
        "compensation_request": input_data["compensation_request"]
    })
    print(lawsuit_draft)
    return lawsuit_draft
generate_lawsuit(user_input)