#!/bin/bash

# 设置只使用2张3090显卡
export CUDA_VISIBLE_DEVICES=2,3

# 使用LoRA微调Qwen2.5-VL-7B进行干扰项生成任务，适合2张3090显卡
swift sft \
    --model /data/lzx/model/qwen2.5-vl-7b-instruct \
    --dataset /home/lzx/lib/pro/dg_finetune/train.jsonl \
    --val_dataset /home/lzx/lib/pro/dg_finetune/validation.jsonl \
    --resume_from_checkpoint /data/lzx/model/dg-sft/v11-20250515-114018/checkpoint-1500 \
    --output_dir /data/lzx/model/dg-sft \
    --train_type lora \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 40 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --max_length 2048 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing true \
    --system "You are an expert in the field of education, and you are good at generating high-quality distractors that seem reasonable but are wrong based on the questions."
