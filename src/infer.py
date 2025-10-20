import os
import torch
import yaml
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def select_device(use_mps: bool) -> torch.device:
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextGenerator:
    def __init__(self, config_path: str) -> None:
        self.config = load_yaml(config_path)
        model_name = self.config.get("model_name_or_path")
        self.device = select_device(self.config.get("use_mps", True))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # better for generation

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
        )

        adapter_path: Optional[str] = self.config.get("adapter_path")
        if adapter_path and os.path.isdir(adapter_path):
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:
        max_seq_length = int(self.config.get("max_seq_length", 512))
        if max_new_tokens is None:
            max_new_tokens = int(self.config.get("default_max_new_tokens", 256))
        if temperature is None:
            temperature = float(self.config.get("default_temperature", 0.7))
        if top_p is None:
            top_p = float(self.config.get("default_top_p", 0.95))

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # return only the completion after the prompt
        return text[len(prompt):].lstrip()
