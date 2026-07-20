"""Promote pass-2 candidate only when its automated gate has passed.

Important: the candidate uses probability calibration stored in model_metadata.json.
The active application runtime must read and apply those calibration parameters.
Do not promote into an older runtime that ignores model metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
CANDIDATE_DIR = MODELS_DIR / "headline_candidate_v2"
REQUIRED = ["bilstm_model.keras", "tokenizer.pkl", "model_metadata.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow promotion after a failed gate. Not recommended.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [name for name in REQUIRED if not (CANDIDATE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Candidate files are missing: {missing}")

    metadata = json.loads((CANDIDATE_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    test_metrics = metadata.get("test_metrics", {})
    gate_metrics = metadata.get("real_world_gate_metrics", {})
    gate = metadata.get("promotion_gate", {})

    print("Untouched test metrics:")
    for name in ("accuracy", "specificity", "precision", "recall", "f1", "roc_auc"):
        if name in test_metrics:
            print(f"  {name}: {test_metrics[name]:.4f}")

    print("Real-world gate metrics:")
    for name in ("accuracy", "specificity", "precision", "recall", "f1"):
        if name in gate_metrics:
            print(f"  {name}: {gate_metrics[name]:.4f}")

    gate_passed = bool(gate.get("passed"))
    print("Promotion gate:", "PASSED" if gate_passed else "FAILED")
    if not gate_passed and not args.force:
        raise SystemExit(
            "Promotion refused because the automated gate failed. Review the reports and retrain."
        )

    confirmation = input("Type PROMOTE_PASS2 to replace the active model: ").strip()
    if confirmation != "PROMOTE_PASS2":
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

    print("Candidate files promoted.")
    print("Previous active files archived at:", archive_dir)
    print(
        "Reminder: update core.py to load model_metadata.json, use its sequence_length and "
        "threshold, and apply Platt calibration before displaying probabilities."
    )


if __name__ == "__main__":
    main()
