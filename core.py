"""Core sarcasm scoring and response-generation logic for Milestone 2."""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import shutil
import tempfile
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("absl").setLevel(logging.ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "bilstm_model.keras"
COMPAT_MODEL_PATH = MODELS_DIR / "bilstm_model.compat.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"

load_dotenv(BASE_DIR / ".env")

MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "30"))
SARCASM_THRESHOLD = float(os.getenv("SARCASM_THRESHOLD", "0.5"))
REPLY_MIN_SCORE = float(os.getenv("REPLY_MIN_SCORE", "0.65"))
MAX_RETRIES = max(1, int(os.getenv("MAX_RETRIES", "3")))
GEMINI_TIMEOUT_SECONDS = max(2.0, float(os.getenv("GEMINI_TIMEOUT_SECONDS", "20")))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SarcasmResult:
    text: str
    cleaned_text: str
    score: float
    is_sarcastic: bool
    label: str
    confidence: str
    explanation: str


@dataclass(frozen=True)
class GeminiResult:
    text: Optional[str]
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class ChatResult:
    reply: str
    user_sarcasm_score: float
    reply_sarcasm_score: float
    user_is_sarcastic: bool
    reply_is_sarcastic: bool
    user_confidence: str
    reply_confidence: str
    user_explanation: str
    response_source: str
    fallback_reason: Optional[str]
    generation_attempts: int


class SarcasmEngineError(RuntimeError):
    """Raised when the sarcasm model/tokenizer cannot be loaded or used."""


def clean_text(text: str) -> str:
    """Normalize text while preserving contractions and common social-text markers."""
    text = (text or "").lower()
    text = re.sub(r"http\S+|www\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-z\s'#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def confidence_label(score_value: float, threshold: float = SARCASM_THRESHOLD) -> str:
    """Describe confidence using distance from the classification threshold."""
    distance = abs(float(score_value) - threshold)
    if distance >= 0.35:
        return "High"
    if distance >= 0.18:
        return "Moderate"
    return "Low"


def explain_score(score_value: float, is_sarcastic: bool) -> str:
    confidence = confidence_label(score_value)
    if is_sarcastic:
        return (
            f"{confidence} confidence that the wording may be sarcastic. "
            "This is a model estimate, not proof of the writer's intent."
        )
    return (
        f"{confidence} confidence that the wording appears more literal or sincere. "
        "Context can still change the intended meaning."
    )


@lru_cache(maxsize=1)
def _load_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise SarcasmEngineError(f"Tokenizer file not found: {TOKENIZER_PATH}")
    with TOKENIZER_PATH.open("rb") as f:
        return pickle.load(f)


def _remove_unsupported_keras_config(value):
    if isinstance(value, dict):
        value.pop("quantization_config", None)
        dtype = value.get("dtype")
        if (
            isinstance(dtype, dict)
            and dtype.get("class_name") == "DTypePolicy"
            and isinstance(dtype.get("config"), dict)
            and dtype["config"].get("name")
        ):
            value["dtype"] = dtype["config"]["name"]
        for child in list(value.values()):
            _remove_unsupported_keras_config(child)
    elif isinstance(value, list):
        for child in value:
            _remove_unsupported_keras_config(child)
    return value


def _build_compat_model_file(source_path: Path) -> Path:
    if not zipfile.is_zipfile(source_path):
        return source_path
    if COMPAT_MODEL_PATH.exists() and COMPAT_MODEL_PATH.stat().st_mtime >= source_path.stat().st_mtime:
        return COMPAT_MODEL_PATH

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(source_path, "r") as original_zip:
            original_zip.extractall(temp_path)
        config_path = temp_path / "config.json"
        if not config_path.exists():
            return source_path
        config = _remove_unsupported_keras_config(json.loads(config_path.read_text(encoding="utf-8")))
        config_path.write_text(json.dumps(config), encoding="utf-8")
        temp_compat = temp_path / COMPAT_MODEL_PATH.name
        with zipfile.ZipFile(temp_compat, "w", compression=zipfile.ZIP_DEFLATED) as patched_zip:
            for file_path in temp_path.iterdir():
                if file_path != temp_compat:
                    patched_zip.write(file_path, arcname=file_path.name)
        shutil.copyfile(temp_compat, COMPAT_MODEL_PATH)
    return COMPAT_MODEL_PATH


def _load_with_available_keras_loader(model_path: Path):
    load_errors = []
    try:
        from keras.models import load_model as keras_load_model
        try:
            return keras_load_model(model_path, compile=False, safe_mode=False)
        except TypeError:
            return keras_load_model(model_path, compile=False)
        except Exception as exc:
            load_errors.append(exc)
    except Exception as exc:
        load_errors.append(exc)

    try:
        from tensorflow.keras.models import load_model as tf_load_model
        return tf_load_model(model_path, compile=False)
    except Exception as exc:
        load_errors.append(exc)

    raise load_errors[-1] if load_errors else RuntimeError("No Keras model loader is available.")


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise SarcasmEngineError(f"Model file not found: {MODEL_PATH}")
    try:
        return _load_with_available_keras_loader(MODEL_PATH)
    except Exception as first_exc:
        try:
            compat_path = _build_compat_model_file(MODEL_PATH)
            if compat_path != MODEL_PATH:
                return _load_with_available_keras_loader(compat_path)
        except Exception as second_exc:
            raise SarcasmEngineError(
                "Failed to load the sarcasm model after applying the Keras compatibility patch: "
                f"{second_exc}"
            ) from second_exc
        raise SarcasmEngineError(f"Failed to load sarcasm model: {first_exc}") from first_exc


def get_engine_status() -> dict:
    status = {
        "version": "2.0.0-milestone-2",
        "model_file_exists": MODEL_PATH.exists(),
        "compat_model_file_exists": COMPAT_MODEL_PATH.exists(),
        "tokenizer_file_exists": TOKENIZER_PATH.exists(),
        "model_ready": False,
        "tokenizer_ready": False,
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
        "gemini_timeout_seconds": GEMINI_TIMEOUT_SECONDS,
        "fallback_enabled": True,
    }
    try:
        _load_tokenizer()
        status["tokenizer_ready"] = True
    except Exception as exc:
        status["tokenizer_error"] = str(exc)
    try:
        _load_model()
        status["model_ready"] = True
    except Exception as exc:
        status["model_error"] = str(exc)
    return status


def score(text: str) -> float:
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0
    tokenizer = _load_tokenizer()
    model = _load_model()
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except Exception as exc:
        raise SarcasmEngineError(
            "TensorFlow preprocessing utilities are unavailable. Run: pip install -r requirements.txt"
        ) from exc

    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    prediction = model.predict(padded, verbose=0)
    return float(np.asarray(prediction)[0][0])


def analyze_text(text: str) -> SarcasmResult:
    probability = score(text)
    is_sarcastic = probability >= SARCASM_THRESHOLD
    return SarcasmResult(
        text=text,
        cleaned_text=clean_text(text),
        score=probability,
        is_sarcastic=is_sarcastic,
        label="Sarcastic" if is_sarcastic else "Not sarcastic",
        confidence=confidence_label(probability),
        explanation=explain_score(probability, is_sarcastic),
    )


def _get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception as exc:
        logger.warning("Gemini client initialization failed: %s", exc)
        return None


def build_prompt(user_text: str, user_score: float) -> str:
    if user_score >= SARCASM_THRESHOLD:
        tone = f"The user is likely sarcastic ({user_score:.0%} model score). Match the energy lightly."
    else:
        tone = f"The user appears mostly sincere ({1-user_score:.0%} inverse score). Use dry wit without hostility."
    return (
        f"Tone context: {tone}\n\nUser text: {user_text}\n\n"
        "Reply in one or two sentences. Keep it concise, witty, non-abusive, and clearly playful. "
        "Do not claim the model knows the user's true intent. Do not use hashtags."
    )


def _fallback_reply(user_text: str, user_score: float) -> str:
    trimmed = " ".join((user_text or "").split())
    preview = trimmed[:80] + ("..." if len(trimmed) > 80 else "")
    if user_score >= SARCASM_THRESHOLD:
        return (
            f"Yes, '{preview}' has all the subtlety of a fire alarm. "
            "The detector is politely pretending not to notice."
        )
    return (
        f"'{preview}' sounds fairly sincere. Naturally, the bot has decided this calls for unnecessary drama."
    )


def _call_gemini(client, prompt: str) -> Optional[str]:
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = getattr(response, "text", None)
    return text.strip() if text else None


def _generate_with_gemini(prompt: str) -> GeminiResult:
    if not GEMINI_API_KEY:
        return GeminiResult(None, "not_configured", "Gemini API key is not configured.")
    client = _get_gemini_client()
    if client is None:
        return GeminiResult(None, "client_error", "Gemini client could not be initialized.")

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call_gemini, client, prompt)
    try:
        text = future.result(timeout=GEMINI_TIMEOUT_SECONDS)
        if not text:
            return GeminiResult(None, "empty_response", "Gemini returned an empty response.")
        return GeminiResult(text)
    except FutureTimeoutError:
        future.cancel()
        return GeminiResult(None, "timeout", f"Gemini did not respond within {GEMINI_TIMEOUT_SECONDS:g} seconds.")
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "api key" in lowered or "permission" in lowered or "401" in lowered or "403" in lowered:
            code = "authentication_error"
        elif "quota" in lowered or "429" in lowered:
            code = "quota_error"
        else:
            code = "request_error"
        logger.warning("Gemini generation failed [%s]: %s", code, message)
        return GeminiResult(None, code, message[:240])
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def respond(user_text: str) -> ChatResult:
    user_analysis = analyze_text(user_text)
    prompt = build_prompt(user_text, user_analysis.score)

    best_reply: Optional[str] = None
    best_score = -1.0
    last_error: Optional[GeminiResult] = None
    attempts = 0

    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        generated = _generate_with_gemini(prompt)
        if not generated.text:
            last_error = generated
            if generated.error_code in {"not_configured", "authentication_error", "client_error"}:
                break
            if attempt < MAX_RETRIES:
                time.sleep(min(0.5 * attempt, 1.5))
            continue

        candidate_score = score(generated.text)
        if candidate_score > best_score:
            best_reply = generated.text
            best_score = candidate_score
        if candidate_score >= REPLY_MIN_SCORE:
            break

    if best_reply is not None:
        reply_analysis = analyze_text(best_reply)
        return ChatResult(
            reply=best_reply,
            user_sarcasm_score=user_analysis.score,
            reply_sarcasm_score=reply_analysis.score,
            user_is_sarcastic=user_analysis.is_sarcastic,
            reply_is_sarcastic=reply_analysis.is_sarcastic,
            user_confidence=user_analysis.confidence,
            reply_confidence=reply_analysis.confidence,
            user_explanation=user_analysis.explanation,
            response_source="Gemini",
            fallback_reason=None,
            generation_attempts=attempts,
        )

    reply = _fallback_reply(user_text, user_analysis.score)
    reply_analysis = analyze_text(reply)
    reason = last_error.error_message if last_error else "Gemini was unavailable."
    return ChatResult(
        reply=reply,
        user_sarcasm_score=user_analysis.score,
        reply_sarcasm_score=reply_analysis.score,
        user_is_sarcastic=user_analysis.is_sarcastic,
        reply_is_sarcastic=reply_analysis.is_sarcastic,
        user_confidence=user_analysis.confidence,
        reply_confidence=reply_analysis.confidence,
        user_explanation=user_analysis.explanation,
        response_source="Local fallback",
        fallback_reason=reason,
        generation_attempts=attempts,
    )
