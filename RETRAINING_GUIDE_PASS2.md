# Headline Sarcasm Retraining — Pass 2

This pass addresses the false-positive behavior found in the first candidate:
ordinary factual headlines such as NASA launches and central-bank announcements
were incorrectly classified as sarcastic.

The new training pass adds:

- 560 weighted hard training examples;
- 70 separate hard-validation examples;
- a 50-item real-world promotion gate;
- stronger regularization and earlier stopping;
- Platt probability calibration;
- threshold selection that penalizes factual-headline false positives;
- automatic refusal to promote a failed candidate.

The hard examples are controlled augmentation. They improve robustness but are
not a substitute for a genuinely independent external benchmark.

## 1. Extract into the project root

The files should merge into the current project:

```text
sarcasm_bot/
  data/Sarcasm_Headlines_Dataset_v2.json
  training/train_headline_model_pass2.py
  training/smoke_test_candidate_pass2.py
  training/promote_candidate_pass2.py
  training/hard_examples_train.csv
  training/hard_examples_validation.csv
  training/real_world_gate.csv
  requirements-training.txt
```

## 2. Install or verify dependencies

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r .\requirements-training.txt
```

## 3. Train pass 2

```powershell
python .\training\train_headline_model_pass2.py
```

Outputs:

```text
models/headline_candidate_v2/
reports/headline_candidate_v2/
```

The training script prints both the untouched dataset test metrics and the
separate real-world gate metrics.

## 4. Run the smoke test

```powershell
python .\training\smoke_test_candidate_pass2.py
```

The script exits with an error unless all ten practical checks pass.

## 5. Review the reports

```powershell
Get-Content .\reports\headline_candidate_v2\evaluation_metrics.json
Import-Csv .\reports\headline_candidate_v2\real_world_gate_predictions.csv |
  Where-Object correct -eq 'False' |
  Format-Table text,label,calibrated_probability,predicted_label -Auto
```

Also inspect:

```text
reports/headline_candidate_v2/test_confusion_matrix.png
reports/headline_candidate_v2/real_world_gate_confusion_matrix.png
```

## 6. Do not promote immediately

Send back:

1. the final training summary;
2. the smoke-test output;
3. `evaluation_metrics.json`;
4. any failed gate rows.

The candidate uses probability calibration. The current application runtime must
be updated to load and apply the calibration parameters in `model_metadata.json`
before the candidate is activated.

## Promotion criteria

The automated gate requires:

- untouched test accuracy >= 0.84;
- untouched test F1 >= 0.84;
- untouched test ROC-AUC >= 0.90;
- real-world gate accuracy >= 0.84;
- factual-headline specificity >= 0.85;
- sarcastic-headline recall >= 0.80.

Movie and song titles containing fewer than four tokens remain
`INSUFFICIENT CONTEXT` in the smoke-test policy.
