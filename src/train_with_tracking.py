import os
import argparse
import yaml
import torch
import json
import wandb
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter


class PredictionLoggingCallback(TrainerCallback):
    def __init__(self, tokenizer, log_every=50, max_examples=5):
        self.tokenizer = tokenizer
        self.log_every = log_every
        self.max_examples = max_examples
        self.step_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.step_count += 1
        if self.step_count % self.log_every == 0:
            self.log_predictions(state.global_step)

    def log_predictions(self, step):
        # 生成一些預測範例
        sample_prompts = [
            "請用 Python 寫個快速排序",
            "解釋二分搜尋演算法",
            "什麼是機器學習？"
        ]
        
        predictions = []
        for prompt in sample_prompts[:self.max_examples]:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.tokenizer.decode(
                    self.trainer.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )[0],
                    skip_special_tokens=True
                )
            predictions.append({
                "prompt": prompt,
                "generated": outputs[len(prompt):].strip()
            })
        
        if wandb.run:
            wandb.log({"predictions": wandb.Table(
                columns=["prompt", "generated"],
                data=[[p["prompt"], p["generated"]] for p in predictions]
            )}, step=step)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_jsonl_data(file_path: str, text_field: str = "text"):
    """Load JSONL data and format for training"""
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if "prompt" in data and "response" in data:
                text = f"### 指令:\n{data['prompt']}\n\n### 回應:\n{data['response']}"
            else:
                text = data.get(text_field, "")
            rows.append({"text": text})
    return Dataset.from_list(rows)


def format_examples(example, tokenizer, max_length: int):
    text = example["text"]
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return tokenized


def maybe_enable_mps(use_mps: bool):
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_experiment_tracking(config):
    """Setup Wandb and TensorBoard tracking"""
    tracking_config = config.get("tracking", {})
    
    # Setup Wandb
    if tracking_config.get("use_wandb", False):
        wandb.init(
            project=tracking_config.get("project_name", "llm-training"),
            name=tracking_config.get("experiment_name", "experiment").replace("{timestamp}", datetime.now().strftime("%Y%m%d_%H%M%S")),
            tags=tracking_config.get("tags", []),
            config=config,
            reinit=True
        )
    
    # Setup TensorBoard
    writer = None
    if tracking_config.get("use_tensorboard", False):
        log_dir = tracking_config.get("tensorboard_log_dir", "outputs/tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    
    return writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tracking-config", type=str, default="configs/experiment_tracking.yaml")
    args = parser.parse_args()

    # Load main config
    cfg = load_config(args.config)
    
    # Load tracking config
    tracking_cfg = load_config(args.tracking_config)
    cfg["tracking"] = tracking_cfg

    model_name = cfg["model_name_or_path"]
    output_dir = cfg.get("output_dir", "outputs")
    seed = cfg.get("seed", 42)

    # Setup experiment tracking
    writer = setup_experiment_tracking(cfg)

    # Support both dataset_name and jsonl_path
    dataset_name = cfg.get("dataset_name")
    jsonl_path = cfg.get("jsonl_path")
    split = cfg.get("split", "train")
    text_field = cfg.get("text_field", "text")

    max_steps = cfg.get("max_steps", 200)
    learning_rate = cfg.get("learning_rate", 2e-4)
    weight_decay = cfg.get("weight_decay", 0.0)
    warmup_ratio = cfg.get("warmup_ratio", 0.03)
    scheduler = cfg.get("lr_scheduler_type", "cosine")
    per_device_train_batch_size = cfg.get("per_device_train_batch_size", 1)
    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 8)
    max_seq_length = cfg.get("max_seq_length", 512)
    save_steps = cfg.get("save_steps", 100)
    logging_steps = cfg.get("logging_steps", 10)

    lora_r = cfg.get("lora_r", 8)
    lora_alpha = cfg.get("lora_alpha", 16)
    lora_dropout = cfg.get("lora_dropout", 0.05)
    lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])

    use_mps = cfg.get("use_mps", True)

    torch.manual_seed(seed)

    device = maybe_enable_mps(use_mps)
    print(f"Using device: {device}")

    tokenizer = prepare_tokenizer(model_name)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    model.to(device)

    print("Loading dataset...")
    if jsonl_path and os.path.exists(jsonl_path):
        dataset = load_jsonl_data(jsonl_path, text_field)
    elif dataset_name:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, split=split)
    else:
        raise ValueError("Must provide either dataset_name or jsonl_path")
    
    tokenized_ds = dataset.map(
        lambda ex: format_examples(ex, tokenizer, max_seq_length),
        remove_columns=[col for col in dataset.column_names if col != text_field],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Setup callbacks
    callbacks = []
    if tracking_cfg.get("use_wandb", False):
        callbacks.append(PredictionLoggingCallback(
            tokenizer,
            log_every=tracking_cfg.get("log_predictions_every", 50),
            max_examples=tracking_cfg.get("max_predictions_to_log", 5)
        ))

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        num_train_epochs=1,
        max_steps=max_steps,
        lr_scheduler_type=scheduler,
        fp16=False,
        bf16=False,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=["wandb"] if tracking_cfg.get("use_wandb", False) else ["none"],
        save_strategy=tracking_cfg.get("checkpoint_strategy", {}).get("save_strategy", "steps"),
        save_steps=tracking_cfg.get("checkpoint_strategy", {}).get("save_steps", save_steps),
        save_total_limit=tracking_cfg.get("checkpoint_strategy", {}).get("save_total_limit", 3),
        load_best_model_at_end=tracking_cfg.get("checkpoint_strategy", {}).get("load_best_model_at_end", False),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("Start training...")
    trainer.train()

    print("Saving adapter...")
    trainer.save_model(output_dir)
    
    if writer:
        writer.close()
    
    if wandb.run:
        wandb.finish()
    
    print("Done.")


if __name__ == "__main__":
    main()
