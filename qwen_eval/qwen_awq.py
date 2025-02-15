import json
import pandas as pd
import requests
import time
from tqdm import tqdm  # 导入tqdm库

HEADERS = {"Content-Type": "application/json"}

INFERENCE_PARAMS = {
    "stream": False,
    "n": 1,
    "temperature": 0,
    "top_p": 0.8,
    "top_k": 2,
    "min_p": 0,
    "messages": [],
    "model": "Qwen2.5-14B-Instruct",
}

system_prompt = """
你是一个乐于助人的人工智能助手，可以优秀的完成用户的问题，并给出答案。
"""


def use_model_generate_sql(question):
    INFERENCE_PARAMS["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    response = requests.post(
        url="http://101.230.251.254:8511/v1/chat/completions",
        headers=HEADERS,
        data=json.dumps(INFERENCE_PARAMS),
    ).json()["choices"][0]["message"]["content"]

    # 处理response，只保留</think>后面的部分
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
    return response


# while True:
#     input_content = input("输入内容：")
#     if input_content == "break":
#         break
#     start = time.time()
#     print(use_model_generate_sql(input_content))
#     print(f"本次回答用时：{time.time() - start}")
#     print("\n\n")
# 创建问题列表
questions = [
    "小王、小张、小赵三个人是好朋友，他们中间其中一个人下海经商，一个人考上了重点大学，一个人参军了。此外他们还知道以下条件：小赵的年龄比士兵的大；大学生的年龄比小张小；小王的年龄和大学生的年龄不一样。请推出这三个人中谁是商人？谁是大学生？谁是士兵？",
    "甲、乙、丙三个人在一起做作业，有一道数学题比较难，当他们三个人都把自己的解法说出来以后，甲说：“我做错了。”乙说：“甲做对了。”丙说：“我做错了。”在一旁的丁看到他们的答案并听了她们的意见后说：“你们三个人中有一个人做对了，有一个人说对了。”请问，他们三人中到底谁做对了？",
    "求解所有在 [0,2π) 区间内的 x 值，使得 sinx+sin2x+sin3x=0。",
    "一个三棱柱的上底和下底为两个等腰直角三角形，每个等腰三角形的直角边长为16。直棱柱的高度等于等腰直角三角形的斜边长度。求直棱柱的表面积。",
    "从集合 {1,2,3,…,2020} 中选出 225 个不同的数，组成一个递增的等差数列。我需要找出这样的数列有多少个。",
    "有集合 A={1,2,3,…,2019}。对于 A 的每一个非空子集，我们需要计算其所有元素的乘积的倒数，然后将所有这些倒数相加。我们需要找到这个总和",
    # 添加更多问题
]

# 可以读取文件
# df_questions = pd.read_excel("../ds_qwen_32b_evaluate.xlsx", usecols=[0])
# questions = df_questions.iloc[1:, 0].tolist()

results = []

# 使用tqdm来显示进度条
for question in tqdm(questions, desc="处理问题", unit="个"):
    start = time.time()
    answer = use_model_generate_sql(question)
    duration = time.time() - start
    results.append({"题目": question, "输出的答案": answer, "耗时": duration})

# 将结果保存到Excel表格
df = pd.DataFrame(results)
df.to_excel("/DATA/LLM/gaojiale/llm_project/deepseek_evaluate/Excel/qwen_32b_AWQ_evaluate.xlsx", index=False)
