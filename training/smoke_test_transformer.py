from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "transformer_candidate"

CASES = [
    ("Heavy rainfall causes flooding across northern Queensland.", 0),
    ("NASA launches new weather observation satellite.", 0),
    ("Today it will rain in Alabama.", 0),
    ("The central bank kept interest rates unchanged.", 0),
    ("Local commuters thrilled to spend another three hours sitting in traffic.", 1),
    ("Wonderful, another delay. Exactly what everyone needed.", 1),
]


def main() -> None:
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    threshold = float(metadata["threshold"])
    max_length = int(metadata.get("max_length", 96))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    failures = 0
    for text, expected in CASES:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            probability = torch.softmax(model(**inputs).logits, dim=-1)[0, 1].item()
        predicted = int(probability >= threshold)
        ok = predicted == expected
        failures += int(not ok)
        print(f"{probability:7.2%} | {'PASS' if ok else 'FAIL'} | {text}")
    if failures:
        raise SystemExit(f"Smoke test failed: {failures} case(s).")
    print("All transformer smoke tests passed.")


if __name__ == "__main__":
    main()
