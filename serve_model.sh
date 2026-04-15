#!/bin/bash
# serve_model.sh — 启动 vLLM 推理服务
#
# 用法: bash serve_model.sh
# 服务启动后监听 http://localhost:8000/v1（OpenAI 兼容接口）
#
# 前置条件:
#   conda activate rag_project   # 或对应 venv
#   pip install vllm

set -euo pipefail

MODEL="/data/zhaoyaoxi/huggingface_cache/Qwen2.5-7B-Instruct" 
PORT=8000
TP=2           # Tensor Parallel Size（双卡）
GPU_MEM=0.80   # 每张卡的显存占用上限（90%）
GPUS="5,6"     # 经检测空闲的 GPU（2026-04-04），其余卡被其他同学占用

echo "========================================"
echo " Starting vLLM server"
echo "   Model : $MODEL"
echo "   Port  : $PORT"
echo "   TP    : $TP"
echo "   GPUs  : $GPUS"
echo "========================================"

CUDA_VISIBLE_DEVICES=$GPUS python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size $TP \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEM \
    --dtype bfloat16 \
    --trust-remote-code \
    --served-model-name "Qwen2.5-7B-Instruct"
