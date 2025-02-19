#!/bin/bash

# 设置要使用的 GPU 编号
export CUDA_VISIBLE_DEVICES=2

# 设置 API 密钥
API_KEY="sk-1234567890"
export VLLM_API_KEY=$API_KEY

# 检查模型路径是否存在
MODEL_PATH="/DATA/LLM_model/deepseek/DeepSeek-R1-Distill-Qwen-1.5B"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model path $MODEL_PATH does not exist"
    exit 1
fi

# 启动 OpenAI API 服务
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "DeepSeek-R1-Distill-Qwen-1.5B" \
    --host 0.0.0.0 \
    --port 8990 \
    --tensor-parallel-size 1 \
    --max-model-len 512 \
    --enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization 0.2 \
    
    
    