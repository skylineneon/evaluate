import pandas as pd

# 创建一个包含2列100行数据的DataFrame
data = {"Column1": range(0, 40), "Column2": range(0, 40)}
df = pd.DataFrame(data)

# 将DataFrame保存为Excel文件
df.to_excel(
    "/DATA/LLM/gaojiale/llm_project/evaluate/Edit_test/output.xlsx", index=False
)
