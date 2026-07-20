"""Train and evaluate a BiLSTM sarcasm classifier for news headlines.

The script:
1. Loads the JSON-lines headline dataset.
2. Removes invalid and duplicate records.
3. Creates stratified train/validation/test splits.
4. Fits the tokenizer on the training split only.
5. Trains a BiLSTM with early stopping.
6. Selects a decision threshold from the validation split.
7. Evaluates once on the untouched test split.
8. Saves a candidate model, tokenizer, metadata, metrics, and predictions.

Run from the project root:
    python training/train_headline_model.py
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the headline sarcasm BiLSTM model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/Sarcasm_Headlines_Dataset_v2.json"),
        help="JSON-lines dataset path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/headline_candidate"),
        help="Candidate model output directory.",
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/headline_candidate"))
    parser.add_argument("--vocab-size", type=int, default=20000)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
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
    """Use the same normalization during training and inference."""
    text = (text or "").lower()
    text = re.sub(r"http\S+|www\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-z\s'#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    frame = pd.read_json(path, lines=True)
    required = {"headline", "is_sarcastic"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    frame = frame[["headline", "is_sarcastic"]].copy()
    frame["headline"] = frame["headline"].fillna("").astype(str)
    frame["cleaned_text"] = frame["headline"].map(clean_text)
    frame["is_sarcastic"] = pd.to_numeric(frame["is_sarcastic"], errors="coerce")
    frame = frame.dropna(subset=["is_sarcastic"])
    frame["is_sarcastic"] = frame["is_sarcastic"].astype(int)
    frame = frame[frame["is_sarcastic"].isin([0, 1])]
    frame = frame[frame["cleaned_text"].str.len() > 0]

    # Identical cleaned headlines must not leak between splits.
    conflicting = (
        frame.groupby("cleaned_text")["is_sarcastic"].nunique().loc[lambda values: values > 1].index
    )
    if len(conflicting):
        frame = frame[~frame["cleaned_text"].isin(conflicting)]
    frame = frame.drop_duplicates(subset=["cleaned_text"], keep="first").reset_index(drop=True)
    return frame


def split_dataset(frame: pd.DataFrame, seed: int):
    train, temp = train_test_split(
        frame,
        test_size=0.30,
        stratify=frame["is_sarcastic"],
        random_state=seed,
    )
    validation, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["is_sarcastic"],
        random_state=seed,
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def make_tokenizer(train_texts: pd.Series, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts.tolist())
    return tokenizer


def vectorize(tokenizer: Tokenizer, texts: pd.Series, sequence_length: int) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts.tolist())
    return pad_sequences(
        sequences,
        maxlen=sequence_length,
        padding="post",
        truncating="post",
    )


def build_model(vocab_size: int, sequence_length: int, embedding_dim: int, lstm_units: int):
    model = Sequential(
        [
            tf.keras.Input(shape=(sequence_length,), dtype="int32"),
            Embedding(vocab_size, embedding_dim, mask_zero=True),
            SpatialDropout1D(0.20),
            Bidirectional(LSTM(lstm_units, dropout=0.20)),
            Dense(64, activation="relu"),
            Dropout(0.30),
            Dense(1, activation="sigmoid"),
        ],
        name="headline_sarcasm_bilstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
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


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict]:
    """Choose threshold on validation data only, maximizing F1 with balanced-accuracy tie-break."""
    candidates = np.arange(0.20, 0.801, 0.01)
    rows = []
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            }
        )
    best = max(rows, key=lambda row: (row["f1"], row["balanced_accuracy"], -abs(row["threshold"] - 0.5)))
    return best["threshold"], {"selection_metric": "validation_f1", **best}


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
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


def save_confusion_matrix(matrix: np.ndarray, destination: Path) -> None:
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
        title="Headline sarcasm test confusion matrix",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_determinism(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(args.data)
    train, validation, test = split_dataset(frame, args.seed)
    print(f"Usable rows: {len(frame):,}")
    print(f"Train / validation / test: {len(train):,} / {len(validation):,} / {len(test):,}")
    print("Label counts:", frame["is_sarcastic"].value_counts().sort_index().to_dict())

    tokenizer = make_tokenizer(train["cleaned_text"], args.vocab_size)
    actual_vocab_size = min(args.vocab_size, len(tokenizer.word_index) + 1)

    x_train = vectorize(tokenizer, train["cleaned_text"], args.sequence_length)
    x_validation = vectorize(tokenizer, validation["cleaned_text"], args.sequence_length)
    x_test = vectorize(tokenizer, test["cleaned_text"], args.sequence_length)
    y_train = train["is_sarcastic"].to_numpy(dtype=np.int32)
    y_validation = validation["is_sarcastic"].to_numpy(dtype=np.int32)
    y_test = test["is_sarcastic"].to_numpy(dtype=np.int32)

    classes = np.array([0, 1], dtype=np.int32)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(label): float(weight) for label, weight in zip(classes, weights)}

    model = build_model(
        vocab_size=actual_vocab_size,
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
    )
    model.summary()

    best_checkpoint = args.output_dir / "best_checkpoint.keras"
    callbacks = [
        ModelCheckpoint(best_checkpoint, monitor="val_loss", mode="min", save_best_only=True),
        EarlyStopping(monitor="val_loss", mode="min", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=1, min_lr=1e-5),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    validation_probabilities = model.predict(x_validation, batch_size=args.batch_size, verbose=0).reshape(-1)
    threshold, threshold_info = choose_threshold(y_validation, validation_probabilities)
    test_probabilities = model.predict(x_test, batch_size=args.batch_size, verbose=0).reshape(-1)
    metrics = calculate_metrics(y_test, test_probabilities, threshold)

    model_path = args.output_dir / "bilstm_model.keras"
    tokenizer_path = args.output_dir / "tokenizer.pkl"
    metadata_path = args.output_dir / "model_metadata.json"
    model.save(model_path)
    with tokenizer_path.open("wb") as handle:
        pickle.dump(tokenizer, handle)

    metadata = {
        "model_version": "headline-bilstm-1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "Bidirectional LSTM",
        "task": "binary sarcasm classification for English news headlines",
        "dataset_file": args.data.name,
        "dataset_rows_after_cleaning": len(frame),
        "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        "label_mapping": {"0": "not_sarcastic", "1": "sarcastic"},
        "sequence_length": args.sequence_length,
        "vocab_size_limit": args.vocab_size,
        "effective_vocab_size": actual_vocab_size,
        "oov_token": "<OOV>",
        "padding": "post",
        "truncating": "post",
        "threshold": threshold,
        "threshold_selection": threshold_info,
        "minimum_context_tokens": 4,
        "test_metrics": metrics,
        "limitations": [
            "The model estimates linguistic patterns and does not prove author intent.",
            "Movie and song titles may contain too little context for reliable classification.",
            "Performance on social-media messages or article paragraphs requires separate validation.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(args.reports_dir / "training_history.csv", index=False)

    predictions = test[["headline", "cleaned_text", "is_sarcastic"]].copy()
    predictions["sarcasm_probability"] = test_probabilities
    predictions["predicted_label"] = (test_probabilities >= threshold).astype(int)
    predictions["correct"] = predictions["predicted_label"] == predictions["is_sarcastic"]
    predictions.to_csv(args.reports_dir / "test_predictions.csv", index=False)

    (args.reports_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    save_confusion_matrix(np.asarray(metrics["confusion_matrix"]), args.reports_dir / "confusion_matrix.png")

    print("\nCandidate model created:", model_path)
    print("Tokenizer:", tokenizer_path)
    print("Metadata:", metadata_path)
    print("Reports:", args.reports_dir)
    print("Selected threshold:", threshold)
    print(
        "Test metrics: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
