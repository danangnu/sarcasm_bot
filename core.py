"""Sarcasm scoring and response generation for the delivery build."""
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import pickle
import re
import shutil
import sys
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

if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    APP_DIR = Path(sys.executable).resolve().parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent
    APP_DIR = BUNDLE_DIR

MODELS_DIR = BUNDLE_DIR / "models"
MODEL_PATH = MODELS_DIR / "bilstm_model.keras"
COMPAT_MODEL_PATH = MODELS_DIR / "bilstm_model.compat.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
ENV_PATH = APP_DIR / ".env"

load_dotenv(ENV_PATH, override=True)


def _load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


MODEL_METADATA = _load_metadata()
CALIBRATION = MODEL_METADATA.get("calibration", {}) or {}
MAX_SEQUENCE_LENGTH = int(MODEL_METADATA.get("sequence_length", os.getenv("MAX_SEQUENCE_LENGTH", "40")))
SARCASM_THRESHOLD = float(MODEL_METADATA.get("threshold", os.getenv("SARCASM_THRESHOLD", "0.44")))
AMBIGUOUS_LOW = float(os.getenv("AMBIGUOUS_LOW", "0.40"))
AMBIGUOUS_HIGH = float(os.getenv("AMBIGUOUS_HIGH", "0.65"))
MIN_CONTEXT_WORDS = int(os.getenv("MIN_CONTEXT_WORDS", "3"))
REPLY_MIN_SCORE = float(os.getenv("REPLY_MIN_SCORE", "0.65"))
MAX_RETRIES = max(1, int(os.getenv("MAX_RETRIES", "3")))
GEMINI_TIMEOUT_SECONDS = max(2.0, float(os.getenv("GEMINI_TIMEOUT_SECONDS", "20")))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

logger = logging.getLogger("sarcasm_bot")
logger.setLevel(logging.INFO)
LOGS_DIR = APP_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
if not any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
    handler = RotatingFileHandler(LOGS_DIR / "sarcasm_bot.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.info(
    "Startup env=%s env_exists=%s key_loaded=%s model=%s threshold=%.2f sequence_length=%s",
    ENV_PATH, ENV_PATH.exists(), bool(GEMINI_API_KEY), GEMINI_MODEL, SARCASM_THRESHOLD, MAX_SEQUENCE_LENGTH,
)


@dataclass(frozen=True)
class SarcasmResult:
    text: str
    cleaned_text: str
    score: float
    raw_score: float
    is_sarcastic: Optional[bool]
    label: str
    confidence: str
    explanation: str
    decision_state: str
    sufficient_context: bool


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


class SarcasmEngineError(RuntimeError):
    pass


def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"http\S+|www\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-z\s'#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def calibrate_probability(raw_score: float) -> float:
    slope = CALIBRATION.get("slope")
    intercept = CALIBRATION.get("intercept")
    if slope is None or intercept is None:
        return min(1.0, max(0.0, float(raw_score)))
    p = min(1.0 - 1e-6, max(1e-6, float(raw_score)))
    logit = math.log(p / (1.0 - p))
    return _sigmoid(float(slope) * logit + float(intercept))


def predicted_class_probability(score_value: float, threshold: float = SARCASM_THRESHOLD) -> float:
    score_value = min(1.0, max(0.0, float(score_value)))
    return score_value if score_value >= threshold else 1.0 - score_value


def confidence_label(score_value: float, label: Optional[str] = None) -> str:
    if label in {"Ambiguous", "Insufficient context", "Outside validated scope"}:
        return "Low"
    probability = predicted_class_probability(score_value)
    if probability >= 0.80:
        return "Very high"
    if probability >= 0.65:
        return "High"
    return "Moderate"


def _decision(score_value: float, cleaned: str) -> tuple[str, Optional[bool], str, bool]:
    words = cleaned.split()
    if len(words) < MIN_CONTEXT_WORDS:
        return "Insufficient context", None, "insufficient_context", False
    if score_value < AMBIGUOUS_LOW:
        return "Not sarcastic", False, "likely_not_sarcastic", True
    if score_value < AMBIGUOUS_HIGH:
        return "Ambiguous", None, "ambiguous", True
    return "Sarcastic", True, "likely_sarcastic", True


def explain_score(score_value: float, label: str) -> str:
    if label == "Insufficient context":
        return "The input is too short to estimate sarcasm reliably. Add more context before treating the result as a classification."
    if label == "Ambiguous":
        return "The model found mixed evidence. The wording is not strong enough to classify confidently as sarcastic or non-sarcastic."
    confidence = confidence_label(score_value, label)
    if label == "Sarcastic":
        return f"{confidence} classification confidence that the wording matches patterns associated with sarcasm. This does not prove the writer's intent."
    return f"{confidence} classification confidence that the wording is more literal or factual. Context can still change the intended meaning."


@lru_cache(maxsize=1)
def _load_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise SarcasmEngineError(f"Tokenizer file not found: {TOKENIZER_PATH}")
    with TOKENIZER_PATH.open("rb") as handle:
        return pickle.load(handle)


def _remove_unsupported_keras_config(value):
    if isinstance(value, dict):
        value.pop("quantization_config", None)
        dtype = value.get("dtype")
        if isinstance(dtype, dict) and dtype.get("class_name") == "DTypePolicy" and isinstance(dtype.get("config"), dict):
            name = dtype["config"].get("name")
            if name:
                value["dtype"] = name
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
        with zipfile.ZipFile(source_path, "r") as archive:
            archive.extractall(temp_path)
        config_path = temp_path / "config.json"
        if not config_path.exists():
            return source_path
        config = _remove_unsupported_keras_config(json.loads(config_path.read_text(encoding="utf-8")))
        config_path.write_text(json.dumps(config), encoding="utf-8")
        temporary_compat = temp_path / COMPAT_MODEL_PATH.name
        with zipfile.ZipFile(temporary_compat, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in temp_path.iterdir():
                if file_path != temporary_compat:
                    archive.write(file_path, arcname=file_path.name)
        shutil.copyfile(temporary_compat, COMPAT_MODEL_PATH)
    return COMPAT_MODEL_PATH


def _load_with_available_keras_loader(model_path: Path):
    errors = []
    try:
        from keras.models import load_model as keras_load_model
        try:
            return keras_load_model(model_path, compile=False, safe_mode=False)
        except TypeError:
            return keras_load_model(model_path, compile=False)
        except Exception as exc:
            errors.append(exc)
    except Exception as exc:
        errors.append(exc)
    try:
        from tensorflow.keras.models import load_model as tf_load_model
        return tf_load_model(model_path, compile=False)
    except Exception as exc:
        errors.append(exc)
    raise errors[-1] if errors else RuntimeError("No Keras loader is available.")


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise SarcasmEngineError(f"Model file not found: {MODEL_PATH}")
    try:
        return _load_with_available_keras_loader(MODEL_PATH)
    except Exception as first_error:
        try:
            compat_path = _build_compat_model_file(MODEL_PATH)
            if compat_path != MODEL_PATH:
                return _load_with_available_keras_loader(compat_path)
        except Exception as second_error:
            raise SarcasmEngineError(f"Failed to load model after compatibility patch: {second_error}") from second_error
        raise SarcasmEngineError(f"Failed to load model: {first_error}") from first_error


def get_engine_status() -> dict:
    status = {
        "version": "2.4.0-delivery-feedback",
        "model_version": MODEL_METADATA.get("model_version", MODEL_METADATA.get("version", "unknown")),
        "model_file_exists": MODEL_PATH.exists(),
        "tokenizer_file_exists": TOKENIZER_PATH.exists(),
        "metadata_file_exists": METADATA_PATH.exists(),
        "model_ready": False,
        "tokenizer_ready": False,
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
        "gemini_timeout_seconds": GEMINI_TIMEOUT_SECONDS,
        "threshold": SARCASM_THRESHOLD,
        "ambiguous_low": AMBIGUOUS_LOW,
        "ambiguous_high": AMBIGUOUS_HIGH,
        "sequence_length": MAX_SEQUENCE_LENGTH,
        "fallback_enabled": True,
    }
    try:
        _load_tokenizer(); status["tokenizer_ready"] = True
    except Exception as exc:
        status["tokenizer_error"] = str(exc)
    try:
        _load_model(); status["model_ready"] = True
    except Exception as exc:
        status["model_error"] = str(exc)
    return status


def raw_score(text: str) -> float:
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0
    tokenizer = _load_tokenizer()
    model = _load_model()
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except Exception as exc:
        raise SarcasmEngineError("TensorFlow preprocessing is unavailable.") from exc
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    prediction = model.predict(padded, verbose=0)
    return float(np.asarray(prediction)[0][0])


def score(text: str) -> float:
    return calibrate_probability(raw_score(text))


def analyze_text(text: str) -> SarcasmResult:
    cleaned = clean_text(text)
    raw = raw_score(text) if cleaned else 0.0
    probability = calibrate_probability(raw)
    label, is_sarcastic, state, sufficient = _decision(probability, cleaned)
    return SarcasmResult(
        text=text,
        cleaned_text=cleaned,
        score=probability,
        raw_score=raw,
        is_sarcastic=is_sarcastic,
        label=label,
        confidence=confidence_label(probability, label),
        explanation=explain_score(probability, label),
        decision_state=state,
        sufficient_context=sufficient,
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


def build_prompt(user_text: str, analysis: SarcasmResult) -> str:
    if analysis.label == "Sarcastic":
        instruction = "The detector found likely sarcasm. Respond with light, controlled wit without hostility."
    elif analysis.label == "Ambiguous":
        instruction = "The detector is uncertain. Respond neutrally and avoid assuming sarcasm."
    elif analysis.label == "Insufficient context":
        instruction = "The input is very short. Ask for context or respond plainly without assuming sarcasm."
    else:
        instruction = "The detector found mostly literal wording. Respond naturally with mild dry wit only if appropriate."
    return (
        f"Classification context: {analysis.label} ({analysis.score:.0%} calibrated score). {instruction}\n\n"
        f"User text: {user_text}\n\nReply in one or two concise sentences. Do not claim certainty about intent. "
        "Keep the response non-abusive and do not use hashtags."
    )


def _fallback_reply(user_text: str, analysis: SarcasmResult) -> str:
    preview = " ".join((user_text or "").split())[:100]
    if analysis.label == "Sarcastic":
        return f"That sounds like praise wearing a very convincing disguise: '{preview}'."
    if analysis.label == "Ambiguous":
        return "That could be sincere or sarcastic depending on the context. A little more background would help."
    if analysis.label == "Insufficient context":
        return "That is too brief to judge reliably. Add a sentence of context and try again."
    return f"'{preview}' reads as mostly literal. The drama department can stand down for now."


def _call_gemini(client, prompt: str) -> Optional[str]:
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    value = getattr(response, "text", None)
    return value.strip() if value else None


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
        return GeminiResult(text) if text else GeminiResult(None, "empty_response", "Gemini returned no text.")
    except FutureTimeoutError:
        future.cancel()
        return GeminiResult(None, "timeout", "Gemini response timed out.")
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "api key" in lowered or "permission" in lowered or "401" in lowered or "403" in lowered:
            code = "authentication_error"
        elif "quota" in lowered or "429" in lowered or "resource_exhausted" in lowered:
            code = "quota_error"
        elif "404" in lowered or "not_found" in lowered or "no longer available" in lowered:
            code = "model_unavailable"
        elif "503" in lowered or "unavailable" in lowered:
            code = "service_unavailable"
        else:
            code = "request_error"
        logger.warning("Gemini generation failed [%s]: %s", code, message)
        return GeminiResult(None, code, message[:300])
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def friendly_fallback_message(error_code: Optional[str]) -> str:
    messages = {
        "not_configured": "Gemini is not configured. A local fallback response was used.",
        "client_error": "Gemini could not be started. A local fallback response was used.",
        "authentication_error": "Gemini authentication failed. Check the API key and project restrictions.",
        "quota_error": "Gemini quota or prepaid credit is unavailable. A local fallback response was used.",
        "model_unavailable": "The configured Gemini model is unavailable. Update GEMINI_MODEL and restart the app.",
        "service_unavailable": "Gemini is temporarily unavailable. A local fallback response was used.",
        "timeout": "Gemini took too long to respond. A local fallback response was used.",
        "empty_response": "Gemini returned no text. A local fallback response was used.",
        "request_error": "Gemini request failed. A local fallback response was used.",
    }
    return messages.get(error_code or "request_error", "Gemini is unavailable. A local fallback response was used.")


def respond(user_text: str) -> ChatResult:
    started = time.perf_counter()
    user_analysis = analyze_text(user_text)
    prompt = build_prompt(user_text, user_analysis)
    best_reply: Optional[str] = None
    last_error: Optional[GeminiResult] = None
    attempts = 0
    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        generated = _generate_with_gemini(prompt)
        if generated.text:
            best_reply = generated.text
            break
        last_error = generated
        if generated.error_code in {"not_configured", "authentication_error", "client_error", "quota_error", "model_unavailable"}:
            break
        if attempt < MAX_RETRIES:
            time.sleep(min(0.5 * attempt, 1.5))
    duration = round((time.perf_counter() - started) * 1000)
    source = "Gemini" if best_reply is not None else "Local fallback"
    reply = best_reply or _fallback_reply(user_text, user_analysis)
    reply_analysis = analyze_text(reply)
    fallback_code = None if best_reply is not None else (last_error.error_code if last_error else "request_error")
    fallback_message = None if best_reply is not None else friendly_fallback_message(fallback_code)
    logger.info("Chat completed source=%s code=%s attempts=%s duration_ms=%s", source, fallback_code, attempts, duration)
    return ChatResult(
        reply=reply,
        user_sarcasm_score=user_analysis.score,
        reply_sarcasm_score=reply_analysis.score,
        user_is_sarcastic=user_analysis.is_sarcastic,
        reply_is_sarcastic=reply_analysis.is_sarcastic,
        user_label=user_analysis.label,
        reply_label=reply_analysis.label,
        user_confidence=user_analysis.confidence,
        reply_confidence=reply_analysis.confidence,
        user_explanation=user_analysis.explanation,
        response_source=source,
        fallback_code=fallback_code,
        fallback_message=fallback_message,
        fallback_reason=fallback_message,
        generation_attempts=attempts,
        request_duration_ms=duration,
    )
