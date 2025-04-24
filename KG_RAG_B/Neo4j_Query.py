from dotenv import load_dotenv
from neo4j import GraphDatabase
import os
import re
# 加載 .env 配置
load_dotenv()

# Neo4j 配置
uri = os.getenv("NEO4J_URI_3068")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD_3068")
driver = GraphDatabase.driver(uri, auth=(username, password))

def find_statute_by_case_id(tx, case_id):
    query = (
        "MATCH (s:法條 {case_id: $case_id})"
        "RETURN s.text AS statute"
    )
    result = tx.run(query, case_id=case_id)
    record = result.single()
    if record:
        return record["statute"]
    else:
        return None
    
def find_siminput_by_case_id(tx, case_id):
    query = (
        "MATCH (s:模擬輸入 {case_id: $case_id})"
        "RETURN s.text AS siminput"
    )
    result = tx.run(query, case_id=case_id)
    record = result.single()
    if record:
        return record["siminput"]
    else:
        return None
    
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

def get_statude_case(case_id):
    with driver.session() as session:
        statude = session.execute_read(find_statute_by_case_id, case_id)
        return statude
    
def get_siminput_case(case_id):
    with driver.session() as session:
        siminput = session.execute_read(find_siminput_by_case_id, case_id)
        return siminput
    
def get_simoutput_case(case_id):
    with driver.session() as session:
        simoutput = session.execute_read(find_simoutput_by_case_id, case_id)
        return simoutput

# 函數：將法條格式標準化
def normalize_statute_reference(reference):
    # 將 "第191條之2" 轉換為 "191-2條"
    normalized = re.sub(r"條之(\d+)", r"-\1條", reference)
    return normalized

def get_statute_id(legal_text):
    # 找出所有引用的法條
    references = re.findall(r"第(\d+-?\d*條之?\d*)", legal_text)
    statute_ids=[]
    for ref in references:
        # 標準化引用格式
        normalized_ref = normalize_statute_reference(ref)
        statute_ids.append(f"民法第{normalized_ref}")
    return statute_ids

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