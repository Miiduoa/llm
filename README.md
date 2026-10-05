# LLM Fine-tuning Pipeline Lab

一個把 LoRA 微調流程拆成「設定、資料、訓練、推理、評估、追蹤」的實驗型 repo。

我不把它包裝成「已訓練出一個更好的模型」。目前 repo 真正能證明的是：訓練入口、JSONL data contract、MPS/CPU 路徑、LoRA 設定、推理／批次評估介面，以及設定一致性檢查都已經寫成可檢查的程式。

## Implemented

### LoRA training

兩個訓練入口：

```bash
python src/train_lora.py --config configs/tinyllama_lora.yaml
python src/train_lora_jsonl.py --config configs/tinyllama_custom.yaml
```

目前支援：

- Hugging Face dataset
- local JSONL
- LoRA target modules
- config-driven `fp16 / bf16`
- optional DeepSpeed config path
- MPS / CPU fallback
- gradient accumulation
- max steps / scheduler / checkpoint settings

### Inference and batch evaluation

```bash
python -m uvicorn src.server:app --host 127.0.0.1 --port 8000
python src/eval.py --config configs/eval.yaml
```

`configs/infer.yaml` 可指定 base model 與 local LoRA adapter。

### Experiment tracking

`src/train_with_tracking.py` 支援 W&B / TensorBoard 設定。Prediction callback 直接使用 Trainer 傳入的 model，並把 input tensor 移到 model device，避免 tracking 路徑跟實際訓練 device 脫節。

## Lightweight validation

CI 不下載模型，也不跑 GPU training。它先驗證那些不需要昂貴運算、但很容易出錯的部分：

```bash
pip install -r requirements-ci.txt
python -m unittest discover -s tests -v
python -m src.config_check
python src/data/prepare.py
python -m src.config_check --jsonl data/combined/train.jsonl
```

檢查內容包含：

- training YAML 必要欄位
- dataset / JSONL source 是否互斥
- positive batch / step / sequence settings
- `fp16` 與 `bf16` 不可同時開啟
- LoRA target modules
- infer / eval config
- DeepSpeed JSON syntax
- eval JSONL data contract
- generated training JSONL data contract

## Data contract

JSONL 可以使用：

```json
{"text":"complete training text"}
```

或：

```json
{"prompt":"instruction","response":"expected response"}
```

第二種格式會在 loader 中組成 instruction-following text，再交給 tokenizer。

## Config status

| Config | Status | 用途 |
|---|---|---|
| `tinyllama_lora.yaml` | executable path | Hugging Face dataset + LoRA |
| `tinyllama_custom.yaml` | executable path | local JSONL + LoRA |
| `models/llama3_8b_lora.yaml` | reference config | 8B 級模型參數範例，需要相符的 GPU 環境 |
| `deepspeed_zero2.json` | supported config file | 可由 training config 指定 |
| `ray_train.yaml` | **reference only** | 目前沒有 Ray trainer entrypoint |

## Full training environment

完整模型訓練需要重量依賴：

```bash
pip install -r requirements.txt
```

主要套件：

`transformers` · `torch` · `datasets` · `accelerate` · `peft` · `trl`

## Repository layout

```text
configs/
  models/
  tinyllama_lora.yaml
  tinyllama_custom.yaml
  infer.yaml
  eval.yaml
src/
  train_lora.py
  train_lora_jsonl.py
  train_with_tracking.py
  infer.py
  eval.py
  config_check.py
  data/prepare.py
tests/
data/
.github/workflows/validate.yml
```

## What this repo does not claim

- 沒有提交模型 checkpoint
- 沒有提交 benchmark improvement
- 沒有宣稱 Llama 3.1 8B config 已在多 GPU 成功訓練
- Ray config 目前只是 reference，沒有假裝已實作
- CI 綠燈代表 config / data pipeline 可驗證，不代表模型品質

如果要把它升級成真正的模型實驗報告，下一步應該是固定資料切分、記錄 base vs LoRA benchmark、保存 training metadata，並讓每一個結論都能追溯到 run artifact。
