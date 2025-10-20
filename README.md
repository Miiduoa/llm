# LLM Training Starter (macOS-friendly)

## 快速開始

- 安裝依賴：
  ```bash
  pip install -r requirements.txt
  ```
- 以 LoRA 在 macOS MPS/CPU 上微調 TinyLlama：
  ```bash
  python src/train_lora.py --config configs/tinyllama_lora.yaml
  ```

## 結構

```
./
├── configs/
│   └── tinyllama_lora.yaml
├── data/
│   └── README.md (描述資料格式)
├── src/
│   └── train_lora.py
├── requirements.txt
└── README.md
```

## 注意
- 先用小模型在本機驗證流程（MPS），再擴展到分散式/大模型。
- 請確保 `accelerate` 已正確設定（`accelerate config`）。
- 若記憶體不足，降低 `per_device_train_batch_size` 或 `max_seq_length`。


## 本機啟動推理 API

```bash
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
# 之後：
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "用 Python 寫個快排。"}'
```

## Render 部署

1. 將此專案推到 Git（如 GitHub）。
2. Render 儀表板 > New > Web Service > 連接你的 repo。
3. Render 會讀取 `render.yaml`，自動設定 build/start 指令。
4. 首次部署完成後，測試：
   ```bash
   curl -X POST https://<your-service-on-render>/v1/generate \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "請用繁體中文解釋二分搜尋。"}'
   ```

> 若要使用 LoRA adapter，先將 `outputs/tinyllama-lora` 夾推到雲端存儲或打包隨部署，並於環境變數/檔案掛載調整 `INFER_CONFIG` 指向對應路徑。


## 使用自定義資料訓練

```bash
# 準備資料
python src/data/prepare.py

# 使用自定義 JSONL 訓練
python src/train_lora_jsonl.py --config configs/tinyllama_custom.yaml
```

## 前端介面

- 本機啟動（含前端）：
  ```bash
  uvicorn src.server_with_frontend:app --reload --host 0.0.0.0 --port 8000
  # 開啟 http://127.0.0.1:8000
  ```

- Render 部署（含前端）：
  - 使用 `render_with_frontend.yaml` 配置
  - 部署後直接訪問你的 Render URL 即可使用聊天介面

## 升級到 8B 模型

```bash
# 在雲端環境（多 GPU）
accelerate launch src/train_lora_jsonl.py --config configs/models/llama3_8b_lora.yaml
```


## 實驗追蹤與分散式訓練

### 實驗追蹤（Wandb + TensorBoard）

```bash
# 設定 Wandb
./scripts/setup_wandb.sh

# 帶追蹤的訓練
python src/train_with_tracking.py --config configs/tinyllama_custom.yaml --tracking-config configs/experiment_tracking.yaml
```

### 分散式訓練

#### 多 GPU 訓練（Accelerate）
```bash
# 設定 Accelerate
accelerate config

# 啟動分散式訓練
./scripts/train_distributed.sh configs/models/llama3_8b_lora.yaml
```

#### 大規模訓練（Ray Train）
```bash
# 安裝 Ray Train
pip install "ray[train]"

# 啟動 Ray 訓練
python -c "
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

# 這裡需要實作 Ray Train 的訓練函數
# 參考 configs/ray_train.yaml 配置
"
```

### 監控與除錯

- **Wandb**: 查看訓練指標、預測範例、模型權重
- **TensorBoard**: `tensorboard --logdir outputs/tensorboard`
- **檢查點**: 自動儲存在 `outputs/` 目錄

