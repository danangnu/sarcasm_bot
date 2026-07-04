"""Core sarcasm scoring and response-generation logic.

Milestone 1 goals covered here:
- Load the saved BiLSTM model and tokenizer from stable project-relative paths.
- Score any pasted text, tweet, headline, or article excerpt.
- Use Gemini when a GEMINI_API_KEY is configured.
- Fall back to a deterministic demo response when Gemini is not configured or fails,
  so the UI can still be tested end-to-end.
"""

from __future__ import annotations

import logging
import os
import json
import pickle
import re
import shutil
import tempfile
import zipfile
import warnings
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
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
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


@dataclass(frozen=True)
class ChatResult:
    reply: str
    user_sarcasm_score: float
    reply_sarcasm_score: float
    user_is_sarcastic: bool
    reply_is_sarcastic: bool
    used_gemini: bool


class SarcasmEngineError(RuntimeError):
    """Raised when the sarcasm model/tokenizer cannot be loaded or used."""


def clean_text(text: str) -> str:
    """Normalize text to match the preprocessing used by the training notebook."""
    text = (text or "").lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=1)
def _load_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise SarcasmEngineError(f"Tokenizer file not found: {TOKENIZER_PATH}")

    with TOKENIZER_PATH.open("rb") as f:
        return pickle.load(f)


def _remove_unsupported_keras_config(value):
    """Remove newer Keras serialization fields that older runtimes may reject.

    The supplied model was saved with a Keras version that can include fields such
    as `quantization_config`. Some TensorFlow/Keras installations fail to load
    that archive with: `Unrecognized keyword arguments passed to Embedding`.
    This helper patches only the JSON config; the trained weights are untouched.
    """
    if isinstance(value, dict):
        value.pop("quantization_config", None)

        # Some older loaders expect dtype to be the policy name, not a nested
        # DTypePolicy object. Keep the actual dtype value only.
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
    """Create a patched .keras archive for runtime compatibility if needed."""
    if not zipfile.is_zipfile(source_path):
        return source_path

    # Rebuild when missing or when the source model was modified more recently.
    if (
        COMPAT_MODEL_PATH.exists()
        and COMPAT_MODEL_PATH.stat().st_mtime >= source_path.stat().st_mtime
    ):
        return COMPAT_MODEL_PATH

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(source_path, "r") as original_zip:
            original_zip.extractall(temp_path)

        config_path = temp_path / "config.json"
        if not config_path.exists():
            return source_path

        config = json.loads(config_path.read_text(encoding="utf-8"))
        config = _remove_unsupported_keras_config(config)
        config_path.write_text(json.dumps(config), encoding="utf-8")

        temp_compat = temp_path / COMPAT_MODEL_PATH.name
        with zipfile.ZipFile(temp_compat, "w", compression=zipfile.ZIP_DEFLATED) as patched_zip:
            for file_path in temp_path.iterdir():
                if file_path == temp_compat:
                    continue
                patched_zip.write(file_path, arcname=file_path.name)

        shutil.copyfile(temp_compat, COMPAT_MODEL_PATH)

    return COMPAT_MODEL_PATH


def _load_with_available_keras_loader(model_path: Path):
    """Load model using the installed Keras/TensorFlow loader."""
    load_errors = []

    # Prefer standalone Keras first because .keras archives are native to Keras 3.
    try:
        from keras.models import load_model as keras_load_model

        try:
            return keras_load_model(model_path, compile=False, safe_mode=False)
        except TypeError:
            return keras_load_model(model_path, compile=False)
        except Exception as exc:
            load_errors.append(exc)
    except Exception as exc:  # pragma: no cover - environment-dependent
        load_errors.append(exc)

    try:
        from tensorflow.keras.models import load_model as tf_load_model

        try:
            return tf_load_model(model_path, compile=False)
        except Exception as exc:
            load_errors.append(exc)
    except Exception as exc:  # pragma: no cover - environment-dependent
        load_errors.append(exc)

    if load_errors:
        raise load_errors[-1]

    raise RuntimeError("No Keras model loader is available.")


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise SarcasmEngineError(f"Model file not found: {MODEL_PATH}")

    try:
        return _load_with_available_keras_loader(MODEL_PATH)
    except Exception as first_exc:  # pragma: no cover - model/runtime-dependent
        try:
            compat_path = _build_compat_model_file(MODEL_PATH)
            if compat_path != MODEL_PATH:
                return _load_with_available_keras_loader(compat_path)
        except Exception as second_exc:
            raise SarcasmEngineError(
                "Failed to load sarcasm model. The saved .keras file appears to "
                "come from a different Keras/TensorFlow version. I tried the "
                f"compatibility loader as well, but it still failed: {second_exc}"
            ) from second_exc

        raise SarcasmEngineError(f"Failed to load sarcasm model: {first_exc}") from first_exc


def get_engine_status() -> dict:
    """Return readiness details for the API health endpoint."""
    status = {
        "model_file_exists": MODEL_PATH.exists(),
        "compat_model_file_exists": COMPAT_MODEL_PATH.exists(),
        "tokenizer_file_exists": TOKENIZER_PATH.exists(),
        "model_ready": False,
        "tokenizer_ready": False,
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
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
    """Return sarcasm probability between 0 and 1."""
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0

    tokenizer = _load_tokenizer()
    model = _load_model()

    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SarcasmEngineError(
            "TensorFlow preprocessing utilities are not available. "
            "Run: pip install -r requirements.txt"
        ) from exc

    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

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
    )


def _get_gemini_client():
    if not GEMINI_API_KEY:
        return None

    try:
        from google import genai
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("google-genai is not available: %s", exc)
        return None

    try:
        return genai.Client(api_key=GEMINI_API_KEY)
    except TypeError:
        # Older google-genai versions can read the key from the environment.
        return genai.Client()
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("Gemini client could not be initialized: %s", exc)
        return None


def build_prompt(user_text: str, user_score: float) -> str:
    if user_score >= SARCASM_THRESHOLD:
        tone = (
            f"The user is being sarcastic with {user_score:.0%} confidence. "
            "Match their energy and respond sarcastically."
        )
    else:
        tone = (
            f"The user appears sincere with {1 - user_score:.0%} confidence. "
            "Respond with a dry, witty sarcastic tone."
        )

    return (
        f"User tone: {tone}\n\n"
        f"User text: {user_text}\n\n"
        "Reply in one or two sentences. Be dry, witty, and clearly sarcastic. "
        "Do not use hashtags. Do not use emojis."
    )


def _fallback_reply(user_text: str, user_score: float) -> str:
    """Deterministic fallback used when Gemini is not configured or unavailable."""
    trimmed = " ".join((user_text or "").split())
    preview = trimmed[:80] + ("..." if len(trimmed) > 80 else "")

    if user_score >= SARCASM_THRESHOLD:
        return (
            f"Yes, '{preview}' definitely has the subtle emotional restraint of a fireworks show. "
            "The sarcasm detector is trying very hard not to look impressed."
        )

    return (
        f"'{preview}' sounds sincere, which is adorable. "
        "Naturally, I will treat that as an invitation to make it unnecessarily dramatic."
    )


def _generate_with_gemini(prompt: str) -> Optional[str]:
    client = _get_gemini_client()
    if client is None:
        return None

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = getattr(response, "text", None)
        if text:
            return text.strip()
    except Exception as exc:  # pragma: no cover - depends on external API
        logger.warning("Gemini generation failed: %s", exc)

    return None


def respond(user_text: str) -> ChatResult:
    user_analysis = analyze_text(user_text)
    prompt = build_prompt(user_text, user_analysis.score)

    reply = None
    used_gemini = False

    for _attempt in range(1, MAX_RETRIES + 1):
        candidate = _generate_with_gemini(prompt)
        if not candidate:
            break

        candidate_score = score(candidate)
        reply = candidate
        used_gemini = True

        if candidate_score >= REPLY_MIN_SCORE:
            return ChatResult(
                reply=candidate,
                user_sarcasm_score=user_analysis.score,
                reply_sarcasm_score=candidate_score,
                user_is_sarcastic=user_analysis.is_sarcastic,
                reply_is_sarcastic=candidate_score >= SARCASM_THRESHOLD,
                used_gemini=True,
            )

    if reply is None:
        reply = _fallback_reply(user_text, user_analysis.score)

    reply_analysis = analyze_text(reply)
    return ChatResult(
        reply=reply,
        user_sarcasm_score=user_analysis.score,
        reply_sarcasm_score=reply_analysis.score,
        user_is_sarcastic=user_analysis.is_sarcastic,
        reply_is_sarcastic=reply_analysis.is_sarcastic,
        used_gemini=used_gemini,
    )
