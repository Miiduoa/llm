import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

TRAINING_CONFIGS = (
    ROOT / "configs" / "tinyllama_lora.yaml",
    ROOT / "configs" / "tinyllama_custom.yaml",
    ROOT / "configs" / "models" / "llama3_8b_lora.yaml",
)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return payload


def validate_training_config(config: dict) -> list[str]:
    errors = []

    for key in (
        "model_name_or_path",
        "output_dir",
        "max_steps",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_seq_length",
        "lora_r",
        "lora_alpha",
        "lora_target_modules",
    ):
        if key not in config:
            errors.append(f"missing {key}")

    has_dataset = bool(config.get("dataset_name"))
    has_jsonl = bool(config.get("jsonl_path"))
    if has_dataset == has_jsonl:
        errors.append("provide exactly one of dataset_name or jsonl_path")

    for key in (
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_seq_length",
        "lora_r",
        "lora_alpha",
    ):
        value = config.get(key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            errors.append(f"{key} must be a positive integer")

    learning_rate = config.get("learning_rate")
    if learning_rate is not None and (
        not isinstance(learning_rate, (int, float))
        or learning_rate <= 0
    ):
        errors.append("learning_rate must be positive")

    modules = config.get("lora_target_modules")
    if modules is not None and (
        not isinstance(modules, list)
        or not modules
        or not all(isinstance(item, str) and item for item in modules)
    ):
        errors.append("lora_target_modules must be a non-empty string list")

    if config.get("fp16") and config.get("bf16"):
        errors.append("fp16 and bf16 cannot both be enabled")

    deepspeed = config.get("deepspeed")
    if deepspeed:
        path = ROOT / deepspeed
        if not path.is_file():
            errors.append(f"deepspeed config does not exist: {deepspeed}")

    return errors


def validate_infer_config(config: dict) -> list[str]:
    errors = []

    if not config.get("model_name_or_path"):
        errors.append("missing model_name_or_path")

    for key in ("max_seq_length", "default_max_new_tokens"):
        value = config.get(key)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"{key} must be a positive integer")

    temperature = config.get("default_temperature")
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        errors.append("default_temperature must be positive")

    top_p = config.get("default_top_p")
    if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        errors.append("default_top_p must be within (0, 1]")

    return errors


def validate_eval_config(config: dict) -> list[str]:
    errors = []

    for key in ("config_path", "input_jsonl", "output_path"):
        if not config.get(key):
            errors.append(f"missing {key}")

    max_samples = config.get("max_samples")
    if max_samples is not None and (
        not isinstance(max_samples, int)
        or max_samples <= 0
    ):
        errors.append("max_samples must be a positive integer")

    config_path = config.get("config_path")
    if config_path and not (ROOT / config_path).is_file():
        errors.append(f"inference config does not exist: {config_path}")

    input_jsonl = config.get("input_jsonl")
    if input_jsonl and not (ROOT / input_jsonl).is_file():
        errors.append(f"eval JSONL does not exist: {input_jsonl}")

    return errors


def _load_jsonl_rows(path: str | Path):
    source = Path(path)
    if not source.is_file():
        return source, [], [f"missing JSONL: {source}"]

    rows = []
    errors = []

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"line {line_number}: invalid JSON ({exc.msg})"
                )
                continue

            if not isinstance(payload, dict):
                errors.append(f"line {line_number}: expected object")
                continue

            rows.append((line_number, payload))

    if not rows and not errors:
        errors.append("JSONL contains no records")

    return source, rows, errors


def validate_jsonl(path: str | Path) -> list[str]:
    _, rows, errors = _load_jsonl_rows(path)

    for line_number, payload in rows:
        text = payload.get("text")
        prompt = payload.get("prompt")
        response = payload.get("response")

        has_text = isinstance(text, str) and bool(text.strip())
        has_pair = (
            isinstance(prompt, str)
            and bool(prompt.strip())
            and isinstance(response, str)
            and bool(response.strip())
        )

        if not (has_text or has_pair):
            errors.append(
                f"line {line_number}: need non-empty text "
                "or prompt + response"
            )

    return errors


def validate_eval_jsonl(path: str | Path) -> list[str]:
    _, rows, errors = _load_jsonl_rows(path)

    for line_number, payload in rows:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(
                f"line {line_number}: eval record needs non-empty prompt"
            )

    return errors


def check_repository() -> list[str]:
    errors = []

    for path in TRAINING_CONFIGS:
        for issue in validate_training_config(load_yaml(path)):
            errors.append(f"{path.relative_to(ROOT)}: {issue}")

    infer_path = ROOT / "configs" / "infer.yaml"
    for issue in validate_infer_config(load_yaml(infer_path)):
        errors.append(f"{infer_path.relative_to(ROOT)}: {issue}")

    eval_path = ROOT / "configs" / "eval.yaml"
    eval_config = load_yaml(eval_path)
    for issue in validate_eval_config(eval_config):
        errors.append(f"{eval_path.relative_to(ROOT)}: {issue}")

    eval_jsonl = ROOT / eval_config["input_jsonl"]
    for issue in validate_eval_jsonl(eval_jsonl):
        errors.append(f"{eval_jsonl.relative_to(ROOT)}: {issue}")

    deepspeed_path = ROOT / "configs" / "deepspeed_zero2.json"
    try:
        json.loads(deepspeed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{deepspeed_path.relative_to(ROOT)}: {exc}")

    ray_path = ROOT / "configs" / "ray_train.yaml"
    ray = load_yaml(ray_path)
    if ray.get("status") != "reference_only":
        errors.append(
            "configs/ray_train.yaml: must be labelled reference_only"
        )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl")
    args = parser.parse_args()

    errors = (
        validate_jsonl(args.jsonl)
        if args.jsonl
        else check_repository()
    )

    if errors:
        for error in errors:
            print(f"FAIL {error}")
        raise SystemExit(1)

    print("PASS configuration and data-contract checks")


if __name__ == "__main__":
    main()
