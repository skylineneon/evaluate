from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel

app = FastAPI()

# 加载模型
embedding_model_path = '/DATA/LLM_zhangfeng/model/embedding/zpoint_large_embedding_zh'
embedding_model = BGEM3FlagModel(embedding_model_path, use_fp16=True, device='cuda:3')

class EmbeddingRequest(BaseModel):
    input: list

class EmbeddingResponse(BaseModel):
    data: list

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    try:
        embeddings = embedding_model.encode(request.input, batch_size=12, max_length=1024)['dense_vecs']
        response_data = [{"object": "embedding", "embedding": emb.tolist(), "index": idx} for idx, emb in enumerate(embeddings)]
        return {"data": response_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8510)