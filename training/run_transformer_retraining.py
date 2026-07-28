from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# The Transformer training pipeline uses PyTorch, while the main project also
# contains TensorFlow for the retained BiLSTM fallback.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE_DIR = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = BASE_DIR / "training" / "train_transformer_model.py"
STATUS_FILE = BASE_DIR / "retraining" / "status.json"
METRICS_FILE = (
    BASE_DIR
    / "reports"
    / "transformer_candidate"
    / "evaluation_metrics.json"
)


def write_status(state: str, message: str, **extra) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "model_type": "transformer",
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    STATUS_FILE.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_training_module():
    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(
            f"Transformer training script was not found: {TRAINING_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "sarcasm_transformer_training",
        TRAINING_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Transformer training script.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    write_status(
        "running",
        "Transformer candidate training is running.",
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    try:
        module = load_training_module()
        train_main = getattr(module, "main", None)
        if not callable(train_main):
            raise RuntimeError(
                "train_transformer_model.py does not expose main()."
            )

        train_main()

        metrics = {}
        if METRICS_FILE.exists():
            metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))

        passed = bool(
            metrics.get("promotion_passed")
            or metrics.get("promotion_gate", {}).get("passed")
        )

        write_status(
            "passed" if passed else "completed",
            (
                "Transformer candidate passed all promotion checks."
                if passed
                else "Transformer candidate training completed, but the "
                     "promotion checks did not all pass."
            ),
            completed_at=datetime.now(timezone.utc).isoformat(),
            promotion_passed=passed,
            metrics_file=str(METRICS_FILE),
        )

    except Exception as exc:
        write_status(
            "failed",
            f"Transformer retraining failed: {exc}",
            failed_at=datetime.now(timezone.utc).isoformat(),
            error_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
