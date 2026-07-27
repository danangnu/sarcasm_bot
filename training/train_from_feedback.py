"""Create a new candidate model using saved human corrections.

This script never replaces the active model. It creates headline_candidate_v3 and
writes a retraining status file. Promotion remains a separate reviewed step.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK = ROOT / "feedback" / "corrections.csv"
BASE_HARD = ROOT / "training" / "hard_examples_train.csv"
GENERATED_HARD = ROOT / "retraining" / "hard_examples_with_feedback.csv"
STATUS = ROOT / "retraining" / "status.json"
TRAIN_SCRIPT = ROOT / "training" / "train_headline_model_pass2.py"
OUTPUT = ROOT / "models" / "headline_candidate_v3"
REPORTS = ROOT / "reports" / "headline_candidate_v3"


def write_status(state: str, message: str, **extra) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "message": message, "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
    STATUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def map_label(value: str) -> int | None:
    return {"not_sarcastic": 0, "sarcastic": 1}.get(value)


def main() -> int:
    try:
        write_status("running", "Preparing feedback examples.")
        if not FEEDBACK.exists():
            raise FileNotFoundError("No feedback file was found.")
        if not BASE_HARD.exists():
            raise FileNotFoundError("Base hard-example file was not found.")

        with BASE_HARD.open("r", encoding="utf-8", newline="") as handle:
            base_rows = list(csv.DictReader(handle))
        with FEEDBACK.open("r", encoding="utf-8", newline="") as handle:
            feedback_rows = list(csv.DictReader(handle))

        rows = list(base_rows)
        seen = {row.get("text", "").strip().lower() for row in rows}
        added = 0
        for item in feedback_rows:
            label = map_label(item.get("correct_label", ""))
            text = item.get("text", "").strip()
            key = text.lower()
            if label is None or not text or key in seen:
                continue
            rows.append({"text": text, "label": str(label), "source": "human_feedback_hard_negative" if label == 0 else "human_feedback_hard_positive"})
            seen.add(key)
            added += 1

        if added < 1:
            raise ValueError("No usable new binary corrections were found. Insufficient-context feedback is saved but is not used for binary model training.")

        GENERATED_HARD.parent.mkdir(parents=True, exist_ok=True)
        with GENERATED_HARD.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["text", "label", "source"])
            writer.writeheader()
            writer.writerows(rows)

        write_status("running", f"Training candidate model with {added} new corrections.", corrections_added=added)
        command = [
            sys.executable, str(TRAIN_SCRIPT),
            "--hard-train", str(GENERATED_HARD),
            "--output-dir", str(OUTPUT),
            "--reports-dir", str(REPORTS),
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        metrics_path = REPORTS / "evaluation_metrics.json"
        if completed.returncode != 0:
            write_status("failed", "Candidate retraining failed. Review the console and logs.", return_code=completed.returncode)
            return completed.returncode
        passed = False
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            passed = bool(metrics.get("promotion_gate", {}).get("passed"))
        state = "passed" if passed else "completed_not_passed"
        message = "Candidate model passed validation and is ready for manual review." if passed else "Candidate model completed but did not pass every promotion gate."
        write_status(state, message, corrections_added=added, candidate_dir=str(OUTPUT), reports_dir=str(REPORTS))
        return 0
    except Exception as exc:
        write_status("failed", f"Retraining failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
