# Data contract

Training JSONL accepts either of these record shapes:

```json
{"text":"complete training text"}
```

or:

```json
{"prompt":"instruction","response":"expected response"}
```

`src/train_lora_jsonl.py` converts the prompt/response form into one instruction-following text sample before tokenization.

`src/data/prepare.py` only generates a tiny deterministic example dataset for pipeline checks. It is not a benchmark or a production training corpus.
