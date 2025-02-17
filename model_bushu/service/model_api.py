import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from fastapi import FastAPI, Request
import uvicorn
from model_init import Model_Server

app = FastAPI()


@app.post('/nlp/qwen2_72B_awq')
async def chat(request: Request):
    result = await model_service.model_infer(request)
    return result


if __name__ == '__main__':
    qwen2_72B_path = '/DATA/LLM_model/Qwen/Qwen2___5-72B-Instruct-AWQ'
    model_service = Model_Server(qwen2_72B_path)
    uvicorn.run(app=app, host='127.0.0.1', port=8091)
