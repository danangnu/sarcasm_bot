"""Retrain the headline sarcasm classifier with hard-example augmentation.

This pass addresses the false-positive problem found during the first practical
smoke test. It keeps the original headline dataset as the main corpus, adds a
weighted set of factual hard negatives and sarcastic hard positives, calibrates
probabilities on validation data, and evaluates a separate real-world gate set.

Run from the project root:
    python training/train_headline_model_pass2.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2

DEFAULT_SEED = 42
EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train headline sarcasm model pass 2.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/Sarcasm_Headlines_Dataset_v2.json"),
    )
    parser.add_argument(
        "--hard-train",
        type=Path,
        default=Path("training/hard_examples_train.csv"),
    )
    parser.add_argument(
        "--hard-validation",
        type=Path,
        default=Path("training/hard_examples_validation.csv"),
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=Path("training/real_world_gate.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/headline_candidate_v2"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/headline_candidate_v2"),
    )
    parser.add_argument("--vocab-size", type=int, default=22000)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--lstm-units", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--hard-positive-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"http\S+|www\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-z\s'#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_base_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    frame = pd.read_json(path, lines=True)
    required = {"headline", "is_sarcastic"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    frame = frame[["headline", "is_sarcastic"]].copy()
    frame = frame.rename(columns={"headline": "text", "is_sarcastic": "label"})
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["cleaned_text"] = frame["text"].map(clean_text)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["label"])
    frame["label"] = frame["label"].astype(int)
    frame = frame[frame["label"].isin([0, 1])]
    frame = frame[frame["cleaned_text"].str.len() > 0]

    conflicting = (
        frame.groupby("cleaned_text")["label"].nunique().loc[lambda values: values > 1].index
    )
    if len(conflicting):
        frame = frame[~frame["cleaned_text"].isin(conflicting)]
    frame = frame.drop_duplicates(subset=["cleaned_text"], keep="first")
    frame["source"] = "base"
    return frame.reset_index(drop=True)


def load_hard_examples(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Hard-example file not found: {path.resolve()}")
    frame = pd.read_csv(path)
    required = {"text", "label", "source"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    frame = frame[["text", "label", "source"]].copy()
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["cleaned_text"] = frame["text"].map(clean_text)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["label"])
    frame["label"] = frame["label"].astype(int)
    frame = frame[frame["label"].isin([0, 1])]
    frame = frame[frame["cleaned_text"].str.len() > 0]
    return frame.drop_duplicates(subset=["cleaned_text"], keep="first").reset_index(drop=True)


def load_gate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Gate file not found: {path.resolve()}")
    frame = pd.read_csv(path)
    required = {"text", "label", "category"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Gate file is missing columns: {sorted(missing)}")
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["cleaned_text"] = frame["text"].map(clean_text)
    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


def split_base_dataset(frame: pd.DataFrame, seed: int):
    train, temp = train_test_split(
        frame,
        test_size=0.30,
        stratify=frame["label"],
        random_state=seed,
    )
    validation, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["label"],
        random_state=seed,
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def verify_no_leakage(*frames: tuple[str, pd.DataFrame]) -> None:
    for index, (name_a, frame_a) in enumerate(frames):
        values_a = set(frame_a["cleaned_text"])
        for name_b, frame_b in frames[index + 1 :]:
            overlap = values_a.intersection(frame_b["cleaned_text"])
            if overlap:
                sample = sorted(overlap)[:3]
                raise ValueError(
                    f"Text leakage detected between {name_a} and {name_b}: {sample}"
                )


def make_tokenizer(texts: pd.Series, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts.tolist())
    return tokenizer


def vectorize(tokenizer: Tokenizer, texts: pd.Series, sequence_length: int) -> np.ndarray:
    return pad_sequences(
        tokenizer.texts_to_sequences(texts.tolist()),
        maxlen=sequence_length,
        padding="post",
        truncating="post",
    )


def build_model(vocab_size: int, sequence_length: int, embedding_dim: int, lstm_units: int):
    model = Sequential(
        [
            tf.keras.Input(shape=(sequence_length,), dtype="int32"),
            Embedding(
                vocab_size,
                embedding_dim,
                mask_zero=True,
                embeddings_regularizer=l2(1e-6),
            ),
            SpatialDropout1D(0.30),
            Bidirectional(LSTM(lstm_units, dropout=0.30)),
            Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
            Dropout(0.40),
            Dense(1, activation="sigmoid"),
        ],
        name="headline_sarcasm_bilstm_pass2",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def fit_platt_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[float, float]:
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(logit(probabilities).reshape(-1, 1), labels, sample_weight=sample_weight)
    return float(calibrator.coef_[0][0]), float(calibrator.intercept_[0])


def calibrate(probabilities: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    values = slope * logit(probabilities) + intercept
    return 1.0 / (1.0 + np.exp(-values))


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            target_names=["not_sarcastic", "sarcastic"],
            output_dict=True,
            zero_division=0,
        ),
    }


def choose_threshold(
    base_labels: np.ndarray,
    base_probabilities: np.ndarray,
    hard_labels: np.ndarray,
    hard_probabilities: np.ndarray,
) -> tuple[float, dict, list[dict]]:
    rows: list[dict] = []
    for threshold in np.arange(0.30, 0.851, 0.01):
        base_predictions = (base_probabilities >= threshold).astype(int)
        hard_predictions = (hard_probabilities >= threshold).astype(int)
        hard_matrix = confusion_matrix(hard_labels, hard_predictions, labels=[0, 1])
        tn, fp, fn, tp = hard_matrix.ravel()
        hard_specificity = tn / (tn + fp) if (tn + fp) else 0.0
        hard_recall = tp / (tp + fn) if (tp + fn) else 0.0
        hard_balanced = balanced_accuracy_score(hard_labels, hard_predictions)
        base_f1 = f1_score(base_labels, base_predictions, zero_division=0)
        base_balanced = balanced_accuracy_score(base_labels, base_predictions)
        objective = 0.40 * base_f1 + 0.20 * base_balanced + 0.40 * hard_balanced
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "objective": float(objective),
                "base_f1": float(base_f1),
                "base_balanced_accuracy": float(base_balanced),
                "hard_balanced_accuracy": float(hard_balanced),
                "hard_specificity": float(hard_specificity),
                "hard_recall": float(hard_recall),
            }
        )

    qualified = [
        row
        for row in rows
        if row["base_f1"] >= 0.80
        and row["hard_specificity"] >= 0.85
        and row["hard_recall"] >= 0.80
    ]
    pool = qualified or rows
    best = max(
        pool,
        key=lambda row: (
            row["objective"],
            row["hard_specificity"],
            row["hard_recall"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    info = {
        "selection_metric": "combined_base_and_hard_validation",
        "constraints_satisfied": bool(qualified),
        **best,
    }
    return best["threshold"], info, rows


def save_confusion_matrix(matrix: np.ndarray, destination: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Not sarcastic", "Sarcastic"],
        yticklabels=["Not sarcastic", "Sarcastic"],
        ylabel="Actual",
        xlabel="Predicted",
        title=title,
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def promotion_gate(test_metrics: dict, gate_metrics: dict) -> dict:
    criteria = {
        "test_accuracy_at_least_0_84": test_metrics["accuracy"] >= 0.84,
        "test_f1_at_least_0_84": test_metrics["f1"] >= 0.84,
        "test_roc_auc_at_least_0_90": test_metrics["roc_auc"] >= 0.90,
        "gate_accuracy_at_least_0_84": gate_metrics["accuracy"] >= 0.84,
        "gate_specificity_at_least_0_85": gate_metrics["specificity"] >= 0.85,
        "gate_recall_at_least_0_80": gate_metrics["recall"] >= 0.80,
    }
    return {"passed": all(criteria.values()), "criteria": criteria}


def main() -> None:
    args = parse_args()
    set_determinism(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_dataset(args.data)
    base_train, base_validation, base_test = split_base_dataset(base, args.seed)
    hard_train = load_hard_examples(args.hard_train)
    hard_validation = load_hard_examples(args.hard_validation)
    gate = load_gate(args.gate)

    verify_no_leakage(
        ("base_train", base_train),
        ("base_validation", base_validation),
        ("base_test", base_test),
        ("hard_train", hard_train),
        ("hard_validation", hard_validation),
        ("gate", gate),
    )

    train = pd.concat([base_train, hard_train], ignore_index=True)
    validation = pd.concat([base_validation, hard_validation], ignore_index=True)

    print(f"Base usable rows: {len(base):,}")
    print(
        "Base train / validation / test: "
        f"{len(base_train):,} / {len(base_validation):,} / {len(base_test):,}"
    )
    print(f"Hard train / validation / gate: {len(hard_train):,} / {len(hard_validation):,} / {len(gate):,}")
    print("Combined train label counts:", train["label"].value_counts().sort_index().to_dict())

    tokenizer = make_tokenizer(train["cleaned_text"], args.vocab_size)
    actual_vocab_size = min(args.vocab_size, len(tokenizer.word_index) + 1)

    x_train = vectorize(tokenizer, train["cleaned_text"], args.sequence_length)
    x_validation = vectorize(tokenizer, validation["cleaned_text"], args.sequence_length)
    x_base_validation = vectorize(tokenizer, base_validation["cleaned_text"], args.sequence_length)
    x_hard_validation = vectorize(tokenizer, hard_validation["cleaned_text"], args.sequence_length)
    x_test = vectorize(tokenizer, base_test["cleaned_text"], args.sequence_length)
    x_gate = vectorize(tokenizer, gate["cleaned_text"], args.sequence_length)

    y_train = train["label"].to_numpy(dtype=np.int32)
    y_validation = validation["label"].to_numpy(dtype=np.int32)
    y_base_validation = base_validation["label"].to_numpy(dtype=np.int32)
    y_hard_validation = hard_validation["label"].to_numpy(dtype=np.int32)
    y_test = base_test["label"].to_numpy(dtype=np.int32)
    y_gate = gate["label"].to_numpy(dtype=np.int32)

    classes = np.array([0, 1], dtype=np.int32)
    class_values = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(label): float(weight) for label, weight in zip(classes, class_values)}

    train_sample_weight = np.array([class_weight[int(label)] for label in y_train], dtype=np.float32)
    hard_start = len(base_train)
    for index in range(hard_start, len(train)):
        source = str(train.iloc[index]["source"])
        multiplier = (
            args.hard_negative_weight if "negative" in source else args.hard_positive_weight
        )
        train_sample_weight[index] *= multiplier

    validation_sample_weight = np.ones(len(validation), dtype=np.float32)
    validation_sample_weight[len(base_validation) :] = np.where(
        y_hard_validation == 0,
        args.hard_negative_weight,
        args.hard_positive_weight,
    )

    model = build_model(
        actual_vocab_size,
        args.sequence_length,
        args.embedding_dim,
        args.lstm_units,
    )
    model.summary()

    best_checkpoint = args.output_dir / "best_checkpoint.keras"
    callbacks = [
        ModelCheckpoint(best_checkpoint, monitor="val_loss", mode="min", save_best_only=True),
        EarlyStopping(monitor="val_loss", mode="min", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=1, min_lr=1e-5),
    ]

    history = model.fit(
        x_train,
        y_train,
        sample_weight=train_sample_weight,
        validation_data=(x_validation, y_validation, validation_sample_weight),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Explicitly reload the best validation-loss checkpoint.
    model = load_model(best_checkpoint, compile=False)

    raw_validation = model.predict(x_validation, batch_size=args.batch_size, verbose=0).reshape(-1)
    slope, intercept = fit_platt_calibrator(
        raw_validation,
        y_validation,
        validation_sample_weight,
    )

    base_validation_probabilities = calibrate(
        model.predict(x_base_validation, batch_size=args.batch_size, verbose=0).reshape(-1),
        slope,
        intercept,
    )
    hard_validation_probabilities = calibrate(
        model.predict(x_hard_validation, batch_size=args.batch_size, verbose=0).reshape(-1),
        slope,
        intercept,
    )

    threshold, threshold_info, threshold_table = choose_threshold(
        y_base_validation,
        base_validation_probabilities,
        y_hard_validation,
        hard_validation_probabilities,
    )

    raw_test = model.predict(x_test, batch_size=args.batch_size, verbose=0).reshape(-1)
    raw_gate = model.predict(x_gate, batch_size=args.batch_size, verbose=0).reshape(-1)
    test_probabilities = calibrate(raw_test, slope, intercept)
    gate_probabilities = calibrate(raw_gate, slope, intercept)

    test_metrics = binary_metrics(y_test, test_probabilities, threshold)
    gate_metrics = binary_metrics(y_gate, gate_probabilities, threshold)
    gate_result = promotion_gate(test_metrics, gate_metrics)

    model_path = args.output_dir / "bilstm_model.keras"
    tokenizer_path = args.output_dir / "tokenizer.pkl"
    metadata_path = args.output_dir / "model_metadata.json"
    model.save(model_path)
    with tokenizer_path.open("wb") as handle:
        pickle.dump(tokenizer, handle)

    metadata = {
        "model_version": "headline-bilstm-2.0-hard-negative-calibrated",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "Regularized Bidirectional LSTM",
        "task": "binary sarcasm classification for English news headlines",
        "dataset_file": args.data.name,
        "dataset_rows_after_cleaning": len(base),
        "hard_example_counts": {
            "train": len(hard_train),
            "validation": len(hard_validation),
            "real_world_gate": len(gate),
        },
        "split": {
            "base_train": len(base_train),
            "base_validation": len(base_validation),
            "base_test": len(base_test),
        },
        "label_mapping": {"0": "not_sarcastic", "1": "sarcastic"},
        "sequence_length": args.sequence_length,
        "vocab_size_limit": args.vocab_size,
        "effective_vocab_size": actual_vocab_size,
        "oov_token": "<OOV>",
        "padding": "post",
        "truncating": "post",
        "threshold": threshold,
        "threshold_selection": threshold_info,
        "probability_calibration": {
            "method": "Platt scaling on validation predictions",
            "slope": slope,
            "intercept": intercept,
        },
        "minimum_context_tokens": 4,
        "test_metrics": test_metrics,
        "real_world_gate_metrics": gate_metrics,
        "promotion_gate": gate_result,
        "limitations": [
            "The score estimates learned linguistic patterns and does not prove author intent.",
            "The hard examples are controlled augmentation, not an independent benchmark.",
            "The real-world gate is a small curated check and does not replace external validation.",
            "Movie and song titles may contain too little context for reliable classification.",
            "Article paragraphs and social-media messages require separate validation.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pd.DataFrame(history.history).to_csv(args.reports_dir / "training_history.csv", index=False)
    pd.DataFrame(threshold_table).to_csv(args.reports_dir / "threshold_search.csv", index=False)

    test_predictions = base_test[["text", "cleaned_text", "label"]].copy()
    test_predictions["raw_probability"] = raw_test
    test_predictions["calibrated_probability"] = test_probabilities
    test_predictions["predicted_label"] = (test_probabilities >= threshold).astype(int)
    test_predictions["correct"] = test_predictions["predicted_label"] == test_predictions["label"]
    test_predictions.to_csv(args.reports_dir / "test_predictions.csv", index=False)

    gate_predictions = gate[["text", "cleaned_text", "label", "category"]].copy()
    gate_predictions["raw_probability"] = raw_gate
    gate_predictions["calibrated_probability"] = gate_probabilities
    gate_predictions["predicted_label"] = (gate_probabilities >= threshold).astype(int)
    gate_predictions["correct"] = gate_predictions["predicted_label"] == gate_predictions["label"]
    gate_predictions.to_csv(args.reports_dir / "real_world_gate_predictions.csv", index=False)

    report = {
        "threshold": threshold,
        "calibration": {"slope": slope, "intercept": intercept},
        "test_metrics": test_metrics,
        "real_world_gate_metrics": gate_metrics,
        "promotion_gate": gate_result,
    }
    (args.reports_dir / "evaluation_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    save_confusion_matrix(
        np.asarray(test_metrics["confusion_matrix"]),
        args.reports_dir / "test_confusion_matrix.png",
        "Untouched headline test confusion matrix",
    )
    save_confusion_matrix(
        np.asarray(gate_metrics["confusion_matrix"]),
        args.reports_dir / "real_world_gate_confusion_matrix.png",
        "Curated real-world gate confusion matrix",
    )

    print("\nCandidate pass 2 created:", model_path)
    print("Tokenizer:", tokenizer_path)
    print("Metadata:", metadata_path)
    print("Reports:", args.reports_dir)
    print(f"Selected calibrated threshold: {threshold:.2f}")
    print(
        "Untouched test: "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"specificity={test_metrics['specificity']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"f1={test_metrics['f1']:.4f}, "
        f"roc_auc={test_metrics['roc_auc']:.4f}"
    )
    print(
        "Real-world gate: "
        f"accuracy={gate_metrics['accuracy']:.4f}, "
        f"specificity={gate_metrics['specificity']:.4f}, "
        f"recall={gate_metrics['recall']:.4f}, "
        f"f1={gate_metrics['f1']:.4f}"
    )
    print("PROMOTION GATE:", "PASSED" if gate_result["passed"] else "FAILED")
    for name, passed in gate_result["criteria"].items():
        print(f"  {'PASS' if passed else 'FAIL'} | {name}")


if __name__ == "__main__":
    main()
