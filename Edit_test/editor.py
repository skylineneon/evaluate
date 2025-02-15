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
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": 2,
    "min_p": 0,
    "messages": [],
    "model": "Qwen2.5-14B-Instruct",
}

system_prompt = """
帮我按照下面的要求对问题进行转换\n
要求：
1、替换问题中的时间，转换成某年某月某日或者某年某月至某月的数字形式\n
2、时间不要全部和原问题中的时间一样，更改20%的问题的年份和月份\n
3、替换的年份确保大部分时间集中在2020年到2025年，同时还要有一些2020年之前的年份，并避免使用括号解释“去年”“前年”等表述\n
4、只给我最终修改后的问题，不要序号\n
5、输出格式要：年份在前，问题在后\n
6、一个问题只转换成一个问题，不要输出多个\n
7、你可以所以编造时间，不需要与原问题全保持一致\n
"""


def call_api(question):
    INFERENCE_PARAMS["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    response = requests.post(
        url="http://101.230.251.254:8511/v1/chat/completions",
        headers=HEADERS,
        data=json.dumps(INFERENCE_PARAMS),
    ).json()["choices"][0]["message"]["content"]

    return response


def load_questions(file_path):
    df_questions = pd.read_excel(file_path, usecols=[0])
    questions = df_questions.iloc[1:, 0].tolist()  # 范围是左闭右开
    return questions


def process_and_save(questions, output_file):
    results = []
    # 使用tqdm来显示进度条
    for question in tqdm(questions, desc="处理问题", unit="个"):
        start = time.time()
        try:
            answer = call_api(question)
        except Exception as e:
            print(f"Error processing question: {question}. Skipping to next question.")
            continue
        duration = time.time() - start
        results.append({"Column1": question, "Column2": answer})

    # 将结果保存到Excel表格
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)


def main():
    input_file = "/DATA/LLM/gaojiale/llm_project/evaluate/Edit_test/output.xlsx"
    output_file = "/DATA/LLM/gaojiale/llm_project/evaluate/Edit_test/output.xlsx"
    questions = load_questions(input_file)
    process_and_save(questions, output_file)


if __name__ == "__main__":
    main()
