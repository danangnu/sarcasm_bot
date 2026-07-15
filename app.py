from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import SarcasmEngineError, analyze_text, get_engine_status, respond

MAX_INPUT_CHARS = 8000


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)


class AnalyzeResponse(BaseModel):
    score: float
    is_sarcastic: bool
    label: str
    confidence: str
    explanation: str
    cleaned_text: str
    character_count: int


class ChatResponse(BaseModel):
    reply: str
    user_sarcasm_score: float
    reply_sarcasm_score: float
    user_is_sarcastic: bool
    reply_is_sarcastic: bool
    user_confidence: str
    reply_confidence: str
    user_explanation: str
    response_source: str
    fallback_reason: str | None
    generation_attempts: int


app = FastAPI(
    title="Sarcasm Detection Chatbot API",
    version="2.0.0-milestone-2",
    description="Milestone 2 API with improved Gemini reliability, diagnostics, and presentation metadata.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "max_input_chars": MAX_INPUT_CHARS, **get_engine_status()}


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
            confidence=result.confidence,
            explanation=result.explanation,
            cleaned_text=result.cleaned_text,
            character_count=len(message),
        )
    except SarcasmEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The text could not be analyzed. Check the server log.") from exc


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
            user_confidence=result.user_confidence,
            reply_confidence=result.reply_confidence,
            user_explanation=result.user_explanation,
            response_source=result.response_source,
            fallback_reason=result.fallback_reason,
            generation_attempts=result.generation_attempts,
        )
    except SarcasmEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The chatbot request failed. Check the server log.") from exc


BASE_DIR = Path(__file__).resolve().parent
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
