import os
from fastapi import FastAPI
from pydantic import BaseModel
from src.infer import TextGenerator

CONFIG_PATH = os.environ.get("INFER_CONFIG", "configs/infer.yaml")

gen = TextGenerator(CONFIG_PATH)
app = FastAPI(title="LLM Inference API")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/generate")
async def generate(req: GenerateRequest):
    output = gen.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return {"output": output}

