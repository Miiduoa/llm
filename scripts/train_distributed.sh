#!/bin/bash
# 分散式訓練腳本

# 設定環境變數
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=${WANDB_API_KEY}
export HF_HUB_DISABLE_TELEMETRY=1

# 選擇配置
CONFIG=${1:-"configs/models/llama3_8b_lora.yaml"}
TRACKING_CONFIG=${2:-"configs/experiment_tracking.yaml"}

echo "Starting distributed training with config: $CONFIG"

# 使用 Accelerate 啟動分散式訓練
accelerate launch \
    --config_file configs/accelerate_multi_gpu.yaml \
    src/train_with_tracking.py \
    --config $CONFIG \
    --tracking-config $TRACKING_CONFIG

echo "Training completed!"
