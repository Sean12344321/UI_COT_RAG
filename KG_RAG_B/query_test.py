import pandas as pd
from define_case_type import get_case_type
from KG_Faiss_Query_3068 import query_faiss
df = pd.read_excel("data_50.xlsx")
for index, row in df.iterrows():
    sim_input = row.iloc[0]
    case_type=get_case_type(sim_input)
    cases=query_faiss(sim_input,case_type)
    ids=[]
    for case in cases:
        ids.append(case["id"])
    with open("test_result.txt", "a", encoding="utf-8") as f:
        f.write(f"{ids}\n")
