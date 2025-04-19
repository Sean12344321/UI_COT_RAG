from neo4j import GraphDatabase

# 設定連線資訊
uri = "neo4j+s://e3da4770.databases.neo4j.io"
user = "neo4j"
password = "4Y977JBsxLjj0KeqPKfgA84qxSa7y8ahzUznkpGf8G4"

# 建立驅動器
driver = GraphDatabase.driver(uri, auth=(user, password))

# 刪除所有節點與關係
def delete_all(tx):
    tx.run("MATCH (n) DETACH DELETE n")

with driver.session() as session:
    session.write_transaction(delete_all)

print("✅ 所有資料已刪除")

driver.close()
