from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from langchain.llms import OpenAI  # 导入LangChain的OpenAI类

# 加载配置文件
with open('/DATA/LLM/gaojiale/llm_project/llm_chat/config/config.json', 'r') as f:
    config = json.load(f)

# 初始化FastAPI应用
app = FastAPI()

# 定义请求体模型
class QueryRequest(BaseModel):
    query: str

# 定义响应体模型
class QueryResponse(BaseModel):
    response: str

# 初始化LangChain的OpenAI客户端
llm = OpenAI(
    openai_api_base=config["qwen_json_data"]["URL_Qwen"],
    openai_api_key="",  # 假设不需要API密钥，如果有需要请在这里设置
    model_name=config["qwen_json_data"]["model"],
    temperature=config["qwen_json_data"]["temperature"],
    streaming=config["qwen_json_data"]["stream"]
)

# 调用Qwen API进行推理
def call_qwen_api(query):
    try:
        # 使用LangChain的OpenAI客户端调用API
        response = llm(query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 定义FastAPI路由
@app.post("/query/", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        # 调用Qwen API
        result = call_qwen_api(request.query)
        # 提取响应内容
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))