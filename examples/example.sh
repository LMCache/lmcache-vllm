#!/bin/bash

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export VLLM_PORT=12345
export VLLM_TP=1
export VLLM_PP=1
export VLLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# vllm instance
VLLM_LOGGING_LEVEL=DEBUG VLLM_DISTRIBUTED_KV_ROLE=both CUDA_VISIBLE_DEVICES=0 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $VLLM_MODEL_NAME \
    --port 8100 \
    --max-model-len 5000 \
    --gpu-memory-utilization 0.5 2>&1 | tee vllm_output.log &


# lmc driver
python entrypoints.py 
# sleep 10
# wait_for_server 8100