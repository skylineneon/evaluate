import json
import requests
import time
import pandas as pd
from tqdm import tqdm  # 导入tqdm库

# 请求头
HEADERS = {"Content-Type": "application/json"}
# 请求体
INFERENCE_PARAMS = {
    "stream": False,
    "n": 1,
    "temperature": 0,
    "top_p": 0.8,
    "top_k": 2,
    "min_p": 0,
    "messages": [],
    "model": "DeepSeek-R1-Distill-Qwen-32B",
}

system_prompt = """
يرجى الإجابة على أسئلة المستخدمين باللغة العربية
"""


def use_model_generate_sql(question):
    INFERENCE_PARAMS["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    response = requests.post(
        url="http://192.168.0.20:8501/v1/chat/completions",
        headers=HEADERS,
        data=json.dumps(INFERENCE_PARAMS),
    ).json()["choices"][0]["message"]["content"]

    return response


# 创建问题列表
questions = [
    '؟كان هناك ثلاثة أشخاص يعملون معًا على إتمام الواجبات المقررة، وكانت هناك مسألة رياضية صعبة. بعد أن قدم كل منهم طريقته في الحل، قال الشخص الأول: "لقد أجبت عليه خطأً." وقال الشخص الثاني: ‘الشخص الأول حلها بشكل صحيح." وقال الشخص الثالث: "أجبت عليه خطأً." ثم نظر الشخص الرابع إلى إجاباتهم واستمع إلى آرائهم وقال: ‘من بينكم ثلاثة، هناك شخص واحد حلها بشكل صحيح، وهناك شخص واحد على حق." فالسؤال هو: من بين الثلاثة من لديه الإجابة الصحيحة؟',
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
df.to_excel(
    "/DATA/LLM/gaojiale/llm_project/deepseek_evaluate/Excel/ds_qwen_32b_evaluate.xlsx",
    index=False,
)
