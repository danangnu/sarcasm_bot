from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
import uvicorn

from core import SarcasmEngineError, analyze_text, get_engine_status, respond


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


class AnalyzeResponse(BaseModel):
    score: float
    is_sarcastic: bool
    label: str
    cleaned_text: str


class ChatResponse(BaseModel):
    reply: str
    user_sarcasm_score: float
    reply_sarcasm_score: float
    user_is_sarcastic: bool
    reply_is_sarcastic: bool
    used_gemini: bool


app = FastAPI(
    title="Sarcasm Detection Chatbot API",
    version="1.0.0-milestone-1",
    description="Working demo API for sarcasm scoring and chatbot response generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", **get_engine_status()}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = analyze_text(message)
        return AnalyzeResponse(
            score=round(result.score, 4),
            is_sarcastic=result.is_sarcastic,
            label=result.label,
            cleaned_text=result.cleaned_text,
        )
    except SarcasmEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analyze failed: {exc}") from exc


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = respond(message)
        return ChatResponse(
            reply=result.reply,
            user_sarcasm_score=round(result.user_sarcasm_score, 4),
            reply_sarcasm_score=round(result.reply_sarcasm_score, 4),
            user_is_sarcastic=result.user_is_sarcastic,
            reply_is_sarcastic=result.reply_is_sarcastic,
            used_gemini=result.used_gemini,
        )
    except SarcasmEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc


BASE_DIR = Path(__file__).resolve().parent
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
