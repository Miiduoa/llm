import os
import json
from typing import List

# 這個腳本提供資料配方的骨架，示範如何將多來源資料
# 合併成統一的 JSONL（每行一個 {"prompt": ..., "response": ...}）。

SOURCES = [
    {"name": "codealpaca", "hint": "https://github.com/sahil280114/codealpaca"},
    {"name": "openorca", "hint": "https://huggingface.co/datasets/Open-Orca/OpenOrca"},
    {"name": "cot_mix", "hint": "多數據來源的 Chain-of-Thought 資料集集合"},
]


def normalize_item(prompt: str, response: str) -> dict:
    return {"prompt": prompt.strip(), "response": response.strip()}


def write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    # TODO: 實作各資料來源的讀取與清理；這裡用幾筆樣例站位
    rows = [
        normalize_item("請用 Python 寫個快速排序。", "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left  = [x for x in arr if x < pivot]\n    mid   = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)"),
        normalize_item("解釋二分搜尋。", "二分搜尋每次將搜尋範圍減半，直到找到目標或區間為空。"),
    ]
    write_jsonl("data/combined/train.jsonl", rows)
    print("Wrote data/combined/train.jsonl")


if __name__ == "__main__":
    main()
