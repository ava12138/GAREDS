#!/bin/bash
CUDA_VISIBLE_DEVICES=7 \
swift infer \
    --adapters /data/lzx/sft/first/v0-20250222-125301/checkpoint-93 \
    --stream true \
    --infer_backend vllm \
    --max_model_len 8192 \
    --temperature 0 \
    --max_new_tokens 2048 \
    --enforce_eager True 