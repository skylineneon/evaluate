from openai import OpenAI

url = "http://192.168.0.147:8509/v1/"
api_key = "1234"

client = OpenAI(base_url=url, api_key=api_key)


# 发送非流式输出的请求
messages = [
    {"role": "system", "content": "YoU are a helpful assistant."},
    {"role": "user", "content": "今天星期几？"},
]
response = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-32B",
    messages=messages,
    stream=False,
)
# answer
content = response.choices[0].message.content
print(content)
