import json
import jsonlines
from typing import Optional
from src.infer import TextGenerator


def run_eval(config_path: str, input_jsonl: str, output_path: str, max_samples: Optional[int] = None, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None, top_p: Optional[float] = None) -> None:
    gen = TextGenerator(config_path)
    count = 0
    with jsonlines.open(input_jsonl, mode="r") as reader, jsonlines.open(output_path, mode="w") as writer:
        for item in reader:
            prompt = item["prompt"]
            output = gen.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
            writer.write({
                "prompt": prompt,
                "output": output,
            })
            count += 1
            if max_samples and count >= max_samples:
                break


if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_eval(
        config_path=cfg["config_path"],
        input_jsonl=cfg["input_jsonl"],
        output_path=cfg["output_path"],
        max_samples=cfg.get("max_samples"),
        max_new_tokens=cfg.get("max_new_tokens"),
        temperature=cfg.get("temperature"),
        top_p=cfg.get("top_p"),
    )
