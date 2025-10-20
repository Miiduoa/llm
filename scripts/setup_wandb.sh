#!/bin/bash
# Wandb 設定腳本

echo "Setting up Wandb for experiment tracking..."

# 檢查是否已安裝 wandb
if ! command -v wandb &> /dev/null; then
    echo "Installing wandb..."
    pip install wandb
fi

# 登入 Wandb（需要 API key）
echo "Please login to Wandb:"
echo "1. Get your API key from https://wandb.ai/settings"
echo "2. Run: wandb login"
echo "3. Or set environment variable: export WANDB_API_KEY=your_key_here"

# 初始化專案
echo "Initializing Wandb project..."
wandb init --project llm-training --dir outputs/wandb
