# Headline Sarcasm Model Retraining Guide

This toolkit creates a new candidate BiLSTM using `data/Sarcasm_Headlines_Dataset_v2.json` without immediately overwriting the current application model.

## 1. Copy the toolkit into the project root

After extraction, the project should contain:

```text
sarcasm_bot/
  data/Sarcasm_Headlines_Dataset_v2.json
  training/train_headline_model.py
  training/smoke_test_candidate.py
  training/promote_candidate.py
  requirements-training.txt
```

## 2. Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```

## 3. Install training packages

```powershell
python -m pip install -r .\requirements-training.txt
```

## 4. Train the candidate

```powershell
python .\training\train_headline_model.py
```

The script creates:

```text
models/headline_candidate/bilstm_model.keras
models/headline_candidate/tokenizer.pkl
models/headline_candidate/model_metadata.json
reports/headline_candidate/evaluation_metrics.json
reports/headline_candidate/confusion_matrix.png
reports/headline_candidate/test_predictions.csv
```

Training time depends on the computer. CPU-only Windows systems may take several minutes.

## 5. Review the metrics

```powershell
Get-Content .\reports\headline_candidate\evaluation_metrics.json
```

The candidate should not be promoted only because training accuracy is high. Review the untouched test metrics, confusion matrix, and sample predictions.

## 6. Run sanity checks

```powershell
python .\training\smoke_test_candidate.py
```

Ordinary headlines should generally be classified as non-sarcastic. Sarcastic news-style headlines should generally receive higher scores. Very short movie and song titles should report `INSUFFICIENT CONTEXT` in this smoke test.

## 7. Promote only after review

```powershell
python .\training\promote_candidate.py
```

Type `PROMOTE` when prompted. The script archives the current active model before replacement.

## 8. Restart FastAPI

```powershell
Ctrl+C
uvicorn app:app --reload
```

## Important

The new model is trained specifically for news headlines. Article paragraphs, social-media posts, movie titles, and song titles are separate domains. They should be validated separately rather than assumed to have the same performance.
