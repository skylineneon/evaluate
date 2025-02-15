from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 假设 vllm 部署的 API 地址
vllm_api_base = "http://192.168.0.20:8511/v1"
# 这里的 API key 可以是任意非空字符串，因为我们使用的是私有部署
api_key = "qwen2.5:32b"

# 初始化 OpenAI 语言模型
llm = ChatOpenAI(
    api_key=api_key,
    model="qwen2.5:32b",
    base_url=vllm_api_base,
    temperature=0.7
    
)
messages = [
    SystemMessage(content="Translate the following from English into English"),
    HumanMessage(content="你好"),
]

print(llm.invoke(messages).content)
