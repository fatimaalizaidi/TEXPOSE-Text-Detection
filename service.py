from fastapi import APIRouter, Form, HTTPException
from pydantic import BaseModel
from classifier_model import load_models, classify_text

print("level.api.level")

tokenizer, model_ai_hum, model_llm = load_models()

class Request(BaseModel):
    prompt: str

router = APIRouter()

@router.get("/")
async def health():
    print("texpose.api.texpose.health")
    return {"message": "svc_texpose service 0.1 alive"}

@router.post("/classify")
async def classify(request: Request):
    result = classify_text(request.prompt, model_ai_hum, model_llm, tokenizer)
    return result