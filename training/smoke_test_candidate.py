"""Run sanity checks against the candidate headline model."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = BASE_DIR / "models" / "headline_candidate"


def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"http\S+|www\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-z\s'#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    metadata = json.loads((CANDIDATE_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    with (CANDIDATE_DIR / "tokenizer.pkl").open("rb") as handle:
        tokenizer = pickle.load(handle)
    model = load_model(CANDIDATE_DIR / "bilstm_model.keras", compile=False)

    samples = [
        "Heavy rainfall causes flooding across northern Queensland.",
        "NASA launches new weather observation satellite.",
        "Central bank keeps interest rates unchanged.",
        "Local commuters thrilled to spend another three hours sitting in traffic.",
        "Nation celebrates exciting new opportunity to pay higher taxes.",
        "The Martian",
        "Ironic",
    ]

    threshold = float(metadata["threshold"])
    minimum_tokens = int(metadata.get("minimum_context_tokens", 4))
    max_length = int(metadata["sequence_length"])

    print(f"Threshold: {threshold:.2f}; sequence length: {max_length}")
    for text in samples:
        cleaned = clean_text(text)
        token_count = len(cleaned.split())
        if token_count < minimum_tokens:
            print(f"INSUFFICIENT CONTEXT | {text}")
            continue
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
        probability = float(np.asarray(model.predict(padded, verbose=0))[0][0])
        label = "SARCASTIC" if probability >= threshold else "NOT SARCASTIC"
        print(f"{probability:6.2%} | {label:13} | {text}")


if __name__ == "__main__":
    main()
