from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CANDIDATE = BASE_DIR / "models" / "transformer_candidate"
ACTIVE = BASE_DIR / "models" / "transformer_model"
BACKUPS = BASE_DIR / "models" / "archive"


def main() -> None:
    metadata_path = CANDIDATE / "model_metadata.json"
    if not metadata_path.exists():
        raise SystemExit("Transformer candidate metadata was not found. Train the candidate first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("promotion_passed"):
        raise SystemExit("Promotion refused: the transformer candidate did not pass every validation gate.")

    confirmation = input("Type PROMOTE_TRANSFORMER to replace the active transformer model: ").strip()
    if confirmation != "PROMOTE_TRANSFORMER":
        raise SystemExit("Promotion cancelled.")

    BACKUPS.mkdir(parents=True, exist_ok=True)
    if ACTIVE.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(str(ACTIVE), str(BACKUPS / f"transformer_model_{stamp}"))
    shutil.copytree(CANDIDATE, ACTIVE)
    print(f"Transformer model promoted to: {ACTIVE}")
    print("Restart Uvicorn or the executable before testing it.")


if __name__ == "__main__":
    main()
