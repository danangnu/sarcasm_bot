"""Run practical checks against the calibrated pass-2 candidate."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = BASE_DIR / "models" / "headline_candidate_v2"


def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"http\S+|www\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-z\s'#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def calibrate(raw_probability: float, slope: float, intercept: float) -> float:
    raw_probability = min(1.0 - 1e-6, max(1e-6, raw_probability))
    logit = np.log(raw_probability / (1.0 - raw_probability))
    value = slope * logit + intercept
    return float(1.0 / (1.0 + np.exp(-value)))


def main() -> None:
    metadata = json.loads((CANDIDATE_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    with (CANDIDATE_DIR / "tokenizer.pkl").open("rb") as handle:
        tokenizer = pickle.load(handle)
    model = load_model(CANDIDATE_DIR / "bilstm_model.keras", compile=False)

    samples = [
        ("Heavy rainfall causes flooding across northern Queensland.", "NOT SARCASTIC"),
        ("NASA launches new weather observation satellite.", "NOT SARCASTIC"),
        ("Central bank keeps interest rates unchanged.", "NOT SARCASTIC"),
        ("Parliament approves funding for a new public hospital.", "NOT SARCASTIC"),
        ("Scientists publish new climate monitoring data.", "NOT SARCASTIC"),
        ("Local commuters thrilled to spend another three hours sitting in traffic.", "SARCASTIC"),
        ("Nation celebrates exciting new opportunity to pay higher taxes.", "SARCASTIC"),
        ("Government unveils revolutionary plan to hold another meeting.", "SARCASTIC"),
        ("The Martian", "INSUFFICIENT CONTEXT"),
        ("Ironic", "INSUFFICIENT CONTEXT"),
    ]

    threshold = float(metadata["threshold"])
    minimum_tokens = int(metadata.get("minimum_context_tokens", 4))
    max_length = int(metadata["sequence_length"])
    calibration = metadata.get("probability_calibration", {})
    slope = float(calibration.get("slope", 1.0))
    intercept = float(calibration.get("intercept", 0.0))

    print(f"Threshold: {threshold:.2f}; sequence length: {max_length}")
    print(f"Calibration: slope={slope:.4f}, intercept={intercept:.4f}")
    passed = 0

    for text, expected in samples:
        cleaned = clean_text(text)
        token_count = len(cleaned.split())
        if token_count < minimum_tokens:
            actual = "INSUFFICIENT CONTEXT"
            print(f"{'PASS' if actual == expected else 'FAIL'} | {actual:20} | {text}")
            passed += int(actual == expected)
            continue

        padded = pad_sequences(
            tokenizer.texts_to_sequences([cleaned]),
            maxlen=max_length,
            padding="post",
            truncating="post",
        )
        raw = float(np.asarray(model.predict(padded, verbose=0))[0][0])
        probability = calibrate(raw, slope, intercept)
        actual = "SARCASTIC" if probability >= threshold else "NOT SARCASTIC"
        ok = actual == expected
        passed += int(ok)
        print(
            f"{'PASS' if ok else 'FAIL'} | {probability:6.2%} calibrated "
            f"({raw:6.2%} raw) | {actual:13} | {text}"
        )

    print(f"\nSmoke result: {passed}/{len(samples)} passed")
    if passed != len(samples):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
