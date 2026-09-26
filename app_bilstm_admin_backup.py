from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import APP_DIR, BUNDLE_DIR, SarcasmEngineError, analyze_text, get_engine_status, respond

MAX_INPUT_CHARS = 8000
FEEDBACK_DIR = APP_DIR / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "corrections.csv"
RETRAINING_DIR = APP_DIR / "retraining"
RETRAINING_STATUS_FILE = RETRAINING_DIR / "status.json"
feedback_lock = threading.Lock()
retrain_lock = threading.Lock()

CANDIDATE_DIR = APP_DIR / "models" / "headline_candidate_v3"
CANDIDATE_REPORT_DIR = APP_DIR / "reports" / "headline_candidate_v3"
CANDIDATE_METRICS_FILE = CANDIDATE_REPORT_DIR / "evaluation_metrics.json"
ACTIVE_MODEL_DIR = APP_DIR / "models"
ACTIVE_MODEL_FILE = ACTIVE_MODEL_DIR / "bilstm_model.keras"
ACTIVE_TOKENIZER_FILE = ACTIVE_MODEL_DIR / "tokenizer.pkl"
ACTIVE_METADATA_FILE = ACTIVE_MODEL_DIR / "model_metadata.json"
COMPAT_MODEL_FILE = ACTIVE_MODEL_DIR / "bilstm_model.compat.keras"
MODEL_ARCHIVE_DIR = ACTIVE_MODEL_DIR / "archive"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)


class AnalyzeResponse(BaseModel):
    score: float
    raw_score: float
    is_sarcastic: Optional[bool]
    label: str
    confidence: str
    explanation: str
    decision_state: str
    sufficient_context: bool
    cleaned_text: str
    character_count: int


class ChatResponse(BaseModel):
    reply: str
    user_sarcasm_score: float
    reply_sarcasm_score: float
    user_is_sarcastic: Optional[bool]
    reply_is_sarcastic: Optional[bool]
    user_label: str
    reply_label: str
    user_confidence: str
    reply_confidence: str
    user_explanation: str
    response_source: str
    fallback_code: Optional[str]
    fallback_message: Optional[str]
    fallback_reason: Optional[str]
    generation_attempts: int
    request_duration_ms: int


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    predicted_label: str
    predicted_score: float = Field(..., ge=0.0, le=1.0)
    correct_label: Literal["not_sarcastic", "sarcastic", "insufficient_context"]
    source: Literal["analyze", "chat"] = "analyze"


class FeedbackUpdateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    correct_label: Literal["not_sarcastic", "sarcastic", "insufficient_context"]


class PromotionRequest(BaseModel):
    confirmation: Literal["PROMOTE"]


app = FastAPI(
    title="Sarcasm Detection and Response API",
    version="2.5.0-admin-feedback",
    description="Admin-enabled build with correction review, controlled retraining, validation reporting, and manual model promotion.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _feedback_count() -> int:
    if not FEEDBACK_FILE.exists():
        return 0
    with FEEDBACK_FILE.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _read_feedback_rows() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    with FEEDBACK_FILE.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_feedback_rows(rows: list[dict]) -> None:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "text", "predicted_label", "predicted_score", "correct_label", "source"]
    temporary = FEEDBACK_FILE.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    temporary.replace(FEEDBACK_FILE)


def _read_candidate_metrics() -> dict:
    if not CANDIDATE_METRICS_FILE.exists():
        return {}
    try:
        return json.loads(CANDIDATE_METRICS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _candidate_summary() -> dict:
    metrics = _read_candidate_metrics()
    promotion = metrics.get("promotion_gate", {}) if isinstance(metrics, dict) else {}
    model = CANDIDATE_DIR / "bilstm_model.keras"
    tokenizer = CANDIDATE_DIR / "tokenizer.pkl"
    metadata = CANDIDATE_DIR / "model_metadata.json"
    return {
        "exists": model.exists() and tokenizer.exists() and metadata.exists(),
        "model_exists": model.exists(),
        "tokenizer_exists": tokenizer.exists(),
        "metadata_exists": metadata.exists(),
        "metrics_available": bool(metrics),
        "promotion_gate_passed": bool(promotion.get("passed")),
        "metrics": metrics,
    }


def _read_status() -> dict:
    if not RETRAINING_STATUS_FILE.exists():
        return {"state": "idle", "message": "No retraining job has been started."}
    try:
        return json.loads(RETRAINING_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"state": "unknown", "message": "Retraining status could not be read."}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "max_input_chars": MAX_INPUT_CHARS,
        "feedback_count": _feedback_count(),
        "retraining_available": not getattr(sys, "frozen", False),
        "retraining_status": _read_status(),
        "candidate": _candidate_summary(),
        **get_engine_status(),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        result = analyze_text(message)
        return AnalyzeResponse(
            score=round(result.score, 4),
            raw_score=round(result.raw_score, 4),
            is_sarcastic=result.is_sarcastic,
            label=result.label,
            confidence=result.confidence,
            explanation=result.explanation,
            decision_state=result.decision_state,
            sufficient_context=result.sufficient_context,
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
        return ChatResponse(**{
            **result.__dict__,
            "user_sarcasm_score": round(result.user_sarcasm_score, 4),
            "reply_sarcasm_score": round(result.reply_sarcasm_score, 4),
        })
    except SarcasmEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The chatbot request failed. Check the server log.") from exc


@app.post("/feedback")
def save_feedback(request: FeedbackRequest):
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": request.text.strip(),
        "predicted_label": request.predicted_label,
        "predicted_score": f"{request.predicted_score:.6f}",
        "correct_label": request.correct_label,
        "source": request.source,
    }
    with feedback_lock:
        exists = FEEDBACK_FILE.exists()
        with FEEDBACK_FILE.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    return {"saved": True, "feedback_count": _feedback_count(), "message": "Correction saved for the next retraining cycle."}


@app.get("/feedback/status")
def feedback_status():
    return {"feedback_count": _feedback_count(), "retraining_available": not getattr(sys, "frozen", False), "retraining_status": _read_status()}


@app.get("/admin/feedback")
def list_feedback():
    with feedback_lock:
        rows = _read_feedback_rows()
    items = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        item["id"] = index
        try:
            item["predicted_score"] = float(item.get("predicted_score", 0))
        except (TypeError, ValueError):
            item["predicted_score"] = 0.0
        items.append(item)
    return {"count": len(items), "items": items}


@app.put("/admin/feedback/{row_id}")
def update_feedback(row_id: int, request: FeedbackUpdateRequest):
    with feedback_lock:
        rows = _read_feedback_rows()
        if row_id < 1 or row_id > len(rows):
            raise HTTPException(status_code=404, detail="Correction was not found.")
        rows[row_id - 1]["text"] = request.text.strip()
        rows[row_id - 1]["correct_label"] = request.correct_label
        _write_feedback_rows(rows)
    return {"updated": True, "feedback_count": len(rows), "message": "Correction updated."}


@app.delete("/admin/feedback/{row_id}")
def delete_feedback(row_id: int):
    with feedback_lock:
        rows = _read_feedback_rows()
        if row_id < 1 or row_id > len(rows):
            raise HTTPException(status_code=404, detail="Correction was not found.")
        deleted = rows.pop(row_id - 1)
        _write_feedback_rows(rows)
    return {"deleted": True, "feedback_count": len(rows), "item": deleted}


@app.get("/admin/feedback/export")
def export_feedback():
    if not FEEDBACK_FILE.exists():
        raise HTTPException(status_code=404, detail="No corrections have been saved.")
    return FileResponse(FEEDBACK_FILE, media_type="text/csv", filename="sarcasm_corrections.csv")


@app.get("/admin/status")
def admin_status():
    return {
        "feedback_count": _feedback_count(),
        "retraining_available": not getattr(sys, "frozen", False),
        "retraining_status": _read_status(),
        "candidate": _candidate_summary(),
        "active_model": get_engine_status(),
    }


@app.get("/admin/candidate/metrics")
def candidate_metrics():
    metrics = _read_candidate_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="Candidate metrics are not available yet.")
    return metrics


@app.post("/admin/promote")
def promote_candidate(request: PromotionRequest):
    if request.confirmation != "PROMOTE":
        raise HTTPException(status_code=400, detail="Promotion confirmation is invalid.")
    summary = _candidate_summary()
    if not summary["exists"]:
        raise HTTPException(status_code=404, detail="A complete candidate model was not found.")
    if not summary["promotion_gate_passed"]:
        raise HTTPException(status_code=409, detail="The candidate did not pass the validation promotion gate.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = MODEL_ARCHIVE_DIR / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for active in (ACTIVE_MODEL_FILE, ACTIVE_TOKENIZER_FILE, ACTIVE_METADATA_FILE):
        if active.exists():
            shutil.copy2(active, backup_dir / active.name)

    for filename in ("bilstm_model.keras", "tokenizer.pkl", "model_metadata.json"):
        source = CANDIDATE_DIR / filename
        destination = ACTIVE_MODEL_DIR / filename
        temporary = destination.with_suffix(destination.suffix + ".new")
        shutil.copy2(source, temporary)
        temporary.replace(destination)

    COMPAT_MODEL_FILE.unlink(missing_ok=True)
    RETRAINING_DIR.mkdir(parents=True, exist_ok=True)
    RETRAINING_STATUS_FILE.write_text(json.dumps({
        "state": "promoted",
        "message": "Candidate promoted. Restart the application to load the new model.",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "backup_dir": str(backup_dir),
        "restart_required": True,
    }, indent=2), encoding="utf-8")
    return {
        "promoted": True,
        "restart_required": True,
        "backup_dir": str(backup_dir),
        "message": "Candidate promoted successfully. Restart Uvicorn to load the new model.",
    }


@app.post("/admin/retrain")
def start_retraining():
    if getattr(sys, "frozen", False):
        raise HTTPException(status_code=409, detail="Retraining is disabled in the client executable. Export corrections and retrain from the development project.")
    minimum = 5
    count = _feedback_count()
    if count < minimum:
        raise HTTPException(status_code=409, detail=f"At least {minimum} saved corrections are required. Current count: {count}.")
    if not retrain_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A retraining request is already being started.")
    try:
        script = APP_DIR / "training" / "train_from_feedback.py"
        if not script.exists():
            raise HTTPException(status_code=404, detail="Feedback retraining script is missing.")
        RETRAINING_DIR.mkdir(parents=True, exist_ok=True)
        RETRAINING_STATUS_FILE.write_text(json.dumps({"state": "queued", "message": "Retraining is starting.", "started_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        process = subprocess.Popen([sys.executable, str(script)], cwd=APP_DIR, creationflags=flags)
        return {"started": True, "process_id": process.pid, "message": "Candidate-model retraining started. The active model will not be replaced automatically."}
    finally:
        retrain_lock.release()




app.mount("/", StaticFiles(directory=BUNDLE_DIR / "static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
