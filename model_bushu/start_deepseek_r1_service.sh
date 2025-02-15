#!/bin/bash

# 设置要使用的 GPU 编号
export CUDA_VISIBLE_DEVICES=0

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
    --port 8051 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 20 \
    
    