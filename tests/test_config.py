import json
import tempfile
import unittest
from pathlib import Path

from src.config_check import (
    check_repository,
    validate_jsonl,
    validate_training_config,
)
from src.data.prepare import normalize_item, write_jsonl


class ConfigTests(unittest.TestCase):
    def test_repository_configs_are_self_consistent(self):
        self.assertEqual(check_repository(), [])

    def test_training_config_requires_one_data_source(self):
        base = {
            "model_name_or_path": "model",
            "output_dir": "outputs/test",
            "max_steps": 1,
            "learning_rate": 0.001,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 128,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_target_modules": ["q_proj"],
        }

        errors = validate_training_config(base)
        self.assertTrue(
            any("exactly one" in error for error in errors)
        )

    def test_precision_modes_are_mutually_exclusive(self):
        config = {
            "model_name_or_path": "model",
            "output_dir": "outputs/test",
            "dataset_name": "dataset",
            "max_steps": 1,
            "learning_rate": 0.001,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 128,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_target_modules": ["q_proj"],
            "fp16": True,
            "bf16": True,
        }

        self.assertIn(
            "fp16 and bf16 cannot both be enabled",
            validate_training_config(config),
        )

    def test_jsonl_accepts_prompt_response_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            write_jsonl(
                str(path),
                [normalize_item(" question ", " answer ")],
            )

            self.assertEqual(validate_jsonl(path), [])

            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["prompt"], "question")
            self.assertEqual(row["response"], "answer")


if __name__ == "__main__":
    unittest.main()
