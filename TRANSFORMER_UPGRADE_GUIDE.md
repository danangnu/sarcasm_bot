# Transformer Upgrade Build

This build keeps the existing BiLSTM active until a transformer candidate passes every promotion gate.

## Install training dependencies

```powershell
python -m pip install -r requirements-transformer.txt
```

## Train the candidate

```powershell
python .\training\train_transformer_model.py
```

The first run downloads `distilroberta-base` from Hugging Face. A GPU is recommended; CPU training can take a long time.

## Review results

```powershell
Get-Content .\reports\transformer_candidate\evaluation_metrics.json
python .\training\smoke_test_transformer.py
```

The candidate is promoted only when accuracy and F1 are at least 90%, real-world accuracy is at least 90%, and factual specificity is at least 95%.

## Promote

```powershell
python .\training\promote_transformer_model.py
```

Type `PROMOTE_TRANSFORMER`, restart Uvicorn, and check `/health`. The response should show `active_classifier: transformer`.

## Important

A 90% result cannot be guaranteed before training. This build measures the result honestly and refuses promotion if the target is not met. Saved client corrections are included only in the training split, never in the untouched test split.
