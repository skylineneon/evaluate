from openai import OpenAI

url = "http://192.168.0.20:8051/v1"
api_key = "token-abc123"

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
# Long COT
reasoning_content = response.choices[0].message.reasoning_content

print(reasoning_content)
print(content)
