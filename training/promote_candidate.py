"""Promote the evaluated candidate model into the application's active models directory."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
CANDIDATE_DIR = MODELS_DIR / "headline_candidate"
REQUIRED = ["bilstm_model.keras", "tokenizer.pkl", "model_metadata.json"]


def main() -> None:
    missing = [name for name in REQUIRED if not (CANDIDATE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Candidate files are missing: {missing}")

    metadata = json.loads((CANDIDATE_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    metrics = metadata.get("test_metrics", {})
    print("Candidate test metrics:")
    for name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"):
        value = metrics.get(name)
        if value is not None:
            print(f"  {name}: {value:.4f}")

    confirmation = input("Type PROMOTE to replace the active model: ").strip()
    if confirmation != "PROMOTE":
        print("Promotion cancelled.")
        return

    archive_dir = MODELS_DIR / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        active = MODELS_DIR / name
        if active.exists():
            shutil.copy2(active, archive_dir / name)

    for name in REQUIRED:
        shutil.copy2(CANDIDATE_DIR / name, MODELS_DIR / name)

    compat = MODELS_DIR / "bilstm_model.compat.keras"
    if compat.exists():
        compat.unlink()

    print("Candidate promoted successfully.")
    print("Previous active files archived at:", archive_dir)


if __name__ == "__main__":
    main()
