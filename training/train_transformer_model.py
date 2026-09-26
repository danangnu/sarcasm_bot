from __future__ import annotations

import os

# This training pipeline uses PyTorch. Prevent Transformers from importing
# TensorFlow/Keras from the BiLSTM environment.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "Sarcasm_Headlines_Dataset_v2.json"
FEEDBACK_PATH = BASE_DIR / "feedback" / "corrections.csv"
HARD_TRAIN_PATH = BASE_DIR / "training" / "hard_examples_train.csv"
REAL_WORLD_GATE_PATH = BASE_DIR / "training" / "real_world_gate.csv"
OUTPUT_DIR = BASE_DIR / "models" / "transformer_candidate"
REPORT_DIR = BASE_DIR / "reports" / "transformer_candidate"

MODEL_NAME = "distilroberta-base"
SEED = 42
MAX_LENGTH = 96
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Promotion requirements. Reaching 90% is the goal, not something the script fabricates.
MIN_TEST_ACCURACY = 0.90
MIN_TEST_F1 = 0.90
MIN_REAL_WORLD_ACCURACY = 0.90
MIN_FACTUAL_SPECIFICITY = 0.95


def clean_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[["text", "label"]].copy()
    frame["text"] = frame["text"].astype(str).str.replace(r"\\s+", " ", regex=True).str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["text", "label"])
    frame = frame[frame["text"].str.len().between(3, 8000)]
    frame["label"] = frame["label"].astype(int)
    frame = frame[frame["label"].isin([0, 1])]
    frame["key"] = frame["text"].str.lower()
    contradictory = frame.groupby("key")["label"].nunique()
    bad = set(contradictory[contradictory > 1].index)
    frame = frame[~frame["key"].isin(bad)]
    return frame.drop_duplicates("key")[["text", "label"]].reset_index(drop=True)


def load_base_dataset() -> pd.DataFrame:
    data = pd.read_json(DATASET_PATH, lines=True)
    return clean_rows(data.rename(columns={"headline": "text", "is_sarcastic": "label"}))


def load_extra_training() -> pd.DataFrame:
    frames = []
    if HARD_TRAIN_PATH.exists():
        hard = pd.read_csv(HARD_TRAIN_PATH)
        if "headline" in hard.columns:
            hard = hard.rename(columns={"headline": "text", "is_sarcastic": "label"})
        frames.append(hard[["text", "label"]])
    if FEEDBACK_PATH.exists():
        feedback = pd.read_csv(FEEDBACK_PATH)
        label_map = {
            "not_sarcastic": 0,
            "not sarcastic": 0,
            "sarcastic": 1,
        }
        if "correct_label" in feedback.columns:
            feedback["label"] = feedback["correct_label"].astype(str).str.lower().map(label_map)
            frames.append(feedback[["text", "label"]].dropna())
    if not frames:
        return pd.DataFrame(columns=["text", "label"])
    return clean_rows(pd.concat(frames, ignore_index=True))


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def metric_bundle(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            labels, predictions, target_names=["not_sarcastic", "sarcastic"], output_dict=True, zero_division=0
        ),
    }


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    best = None
    for threshold in np.arange(0.20, 0.811, 0.01):
        metrics = metric_bundle(labels, probabilities, float(threshold))
        # Prefer F1 while penalising factual false positives.
        objective = metrics["f1"] + 0.25 * metrics["specificity"]
        row = {"threshold": float(threshold), "objective": float(objective), **metrics}
        rows.append(row)
        if best is None or objective > best[0]:
            best = (objective, float(threshold))
    return best[1], rows


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    probs = softmax(np.asarray(logits))[:, 1]
    pred = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, pred),
        "f1": f1_score(labels, pred, zero_division=0),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
    }


def predict_probabilities(trainer: Trainer, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    output = trainer.predict(dataset)
    return np.asarray(output.label_ids), softmax(np.asarray(output.predictions))[:, 1]


def load_gate() -> pd.DataFrame:
    if not REAL_WORLD_GATE_PATH.exists():
        return pd.DataFrame(columns=["text", "label", "category"])
    gate = pd.read_csv(REAL_WORLD_GATE_PATH)
    if "headline" in gate.columns:
        gate = gate.rename(columns={"headline": "text", "is_sarcastic": "label"})
    if "category" not in gate.columns:
        gate["category"] = "general"
    return gate[["text", "label", "category"]].dropna()


def main() -> None:
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base_dataset()
    train_pool, test = train_test_split(base, test_size=0.15, random_state=SEED, stratify=base["label"])
    train, validation = train_test_split(
        train_pool, test_size=0.1764705882, random_state=SEED, stratify=train_pool["label"]
    )  # approximately 70/15/15 overall

    extras = load_extra_training()
    if not extras.empty:
        # Corrections and hard examples are training-only. Repeat modestly so they influence learning.
        train = pd.concat([train, extras, extras], ignore_index=True)
        train = clean_rows(train)

    print(f"Train / validation / test: {len(train):,} / {len(validation):,} / {len(test):,}")
    print(f"Additional hard/corrected examples: {len(extras):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def to_dataset(frame: pd.DataFrame) -> Dataset:
        dataset = Dataset.from_pandas(frame[["text", "label"]], preserve_index=False)
        return dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH),
            batched=True,
            remove_columns=["text"],
        )

    train_ds = to_dataset(train)
    validation_ds = to_dataset(validation)
    test_ds = to_dataset(test)

    counts = train["label"].value_counts().to_dict()
    total = len(train)
    class_weights = torch.tensor([
        total / (2 * max(1, counts.get(0, 0))),
        total / (2 * max(1, counts.get(1, 0))),
    ], dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "NOT_SARCASTIC", 1: "SARCASTIC"},
        label2id={"NOT_SARCASTIC": 0, "SARCASTIC": 1},
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )
    trainer.train()

    val_labels, val_probs = predict_probabilities(trainer, validation_ds)
    threshold, threshold_rows = choose_threshold(val_labels, val_probs)
    pd.DataFrame(threshold_rows).to_csv(REPORT_DIR / "threshold_search.csv", index=False)

    test_labels, test_probs = predict_probabilities(trainer, test_ds)
    test_metrics = metric_bundle(test_labels, test_probs, threshold)
    test_predictions = test.copy()
    test_predictions["probability"] = test_probs
    test_predictions["predicted_label"] = (test_probs >= threshold).astype(int)
    test_predictions["correct"] = test_predictions["label"] == test_predictions["predicted_label"]
    test_predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    gate = load_gate()
    gate_metrics = None
    if not gate.empty:
        gate_ds = to_dataset(gate)
        gate_labels, gate_probs = predict_probabilities(trainer, gate_ds)
        gate_metrics = metric_bundle(gate_labels, gate_probs, threshold)
        gate_out = gate.copy()
        gate_out["probability"] = gate_probs
        gate_out["predicted_label"] = (gate_probs >= threshold).astype(int)
        gate_out["correct"] = gate_out["label"] == gate_out["predicted_label"]
        gate_out.to_csv(REPORT_DIR / "real_world_gate_predictions.csv", index=False)

    promotion_checks = {
        "test_accuracy_at_least_90": test_metrics["accuracy"] >= MIN_TEST_ACCURACY,
        "test_f1_at_least_90": test_metrics["f1"] >= MIN_TEST_F1,
        "real_world_accuracy_at_least_90": bool(gate_metrics and gate_metrics["accuracy"] >= MIN_REAL_WORLD_ACCURACY),
        "factual_specificity_at_least_95": bool(gate_metrics and gate_metrics["specificity"] >= MIN_FACTUAL_SPECIFICITY),
    }
    passed = all(promotion_checks.values())

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    shutil.rmtree(OUTPUT_DIR / "checkpoints", ignore_errors=True)

    metadata = {
        "model_version": "distilroberta-sarcasm-1.0",
        "architecture": "DistilRoBERTaForSequenceClassification",
        "base_model": MODEL_NAME,
        "threshold": threshold,
        "max_length": MAX_LENGTH,
        "label_mapping": {"0": "not_sarcastic", "1": "sarcastic"},
        "training_examples": len(train),
        "validation_examples": len(validation),
        "test_examples": len(test),
        "feedback_examples_used": len(extras),
        "test_metrics": test_metrics,
        "real_world_gate_metrics": gate_metrics,
        "promotion_checks": promotion_checks,
        "promotion_passed": passed,
    }
    (OUTPUT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REPORT_DIR / "evaluation_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps({
        "candidate": str(OUTPUT_DIR),
        "threshold": threshold,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "real_world_accuracy": gate_metrics["accuracy"] if gate_metrics else None,
        "promotion_passed": passed,
        "promotion_checks": promotion_checks,
    }, indent=2))


if __name__ == "__main__":
    main()
