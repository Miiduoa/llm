import os
import argparse
import yaml
import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_jsonl_data(file_path: str, text_field: str = "text"):
    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            data = json.loads(line)
            if "prompt" in data and "response" in data:
                text = (
                    f"### 指令:\n{data['prompt']}\n\n"
                    f"### 回應:\n{data['response']}"
                )
            else:
                text = data.get(text_field, "")

            if not str(text).strip():
                raise ValueError(
                    f"empty training text at line {line_number}"
                )

            rows.append({"text": str(text)})

    if not rows:
        raise ValueError("training JSONL is empty")

    return Dataset.from_list(rows)


def format_examples(example, tokenizer, max_length: int):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def maybe_enable_mps(use_mps: bool):
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_name = cfg["model_name_or_path"]
    output_dir = cfg.get("output_dir", "outputs")
    seed = cfg.get("seed", 42)

    dataset_name = cfg.get("dataset_name")
    jsonl_path = cfg.get("jsonl_path")
    split = cfg.get("split", "train")
    text_field = cfg.get("text_field", "text")

    max_steps = cfg.get("max_steps", 200)
    learning_rate = cfg.get("learning_rate", 2e-4)
    weight_decay = cfg.get("weight_decay", 0.0)
    warmup_ratio = cfg.get("warmup_ratio", 0.03)
    scheduler = cfg.get("lr_scheduler_type", "cosine")
    per_device_train_batch_size = cfg.get(
        "per_device_train_batch_size",
        1,
    )
    gradient_accumulation_steps = cfg.get(
        "gradient_accumulation_steps",
        8,
    )
    max_seq_length = cfg.get("max_seq_length", 512)
    save_steps = cfg.get("save_steps", 100)
    logging_steps = cfg.get("logging_steps", 10)

    lora_r = cfg.get("lora_r", 8)
    lora_alpha = cfg.get("lora_alpha", 16)
    lora_dropout = cfg.get("lora_dropout", 0.05)
    lora_target_modules = cfg.get(
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    use_mps = cfg.get("use_mps", True)
    fp16 = bool(cfg.get("fp16", False))
    bf16 = bool(cfg.get("bf16", False))
    deepspeed = cfg.get("deepspeed")

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
        remove_columns=[
            col for col in dataset.column_names
            if col != "text"
        ],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        num_train_epochs=1,
        max_steps=max_steps,
        lr_scheduler_type=scheduler,
        fp16=fp16,
        bf16=bf16,
        deepspeed=deepspeed,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Start training...")
    trainer.train()

    print("Saving adapter...")
    trainer.save_model(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
