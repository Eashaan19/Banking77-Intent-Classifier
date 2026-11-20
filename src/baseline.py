"""Classical TF-IDF + Logistic Regression baseline and preprocessing ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import BASELINE_MODEL_PATH, SEED
from src.data import DatasetBundle
from src.evaluation import compute_classification_metrics
from src.preprocessing import preprocess_batch


def build_tfidf_lr_pipeline(c_value: float = 5.0, seed: int = SEED) -> Pipeline:
    """Create the TF-IDF (1-2 n-grams) + Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    max_features=15000,
                    token_pattern=r"(?u)\b[a-zA-Z0-9_']+\b",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def run_preprocessing_ablation(
    bundle: DatasetBundle,
    c_value: float = 5.0,
    seed: int = SEED,
) -> pd.DataFrame:
    """Compare raw, aggressive NLTK, and domain-aware preprocessing on val split."""
    modes = [
        ("raw_minimal", "Raw Minimal (Unicode + whitespace only)"),
        ("classical_aggressive", "Aggressive Classical (NLTK stopwords + WordNet lemma)"),
        ("classical_domain_aware", "Domain-Aware Classical (Preserves negation + _amount_/_num_)"),
    ]
    rows: list[dict] = []
    for mode_key, description in modes:
        x_train = preprocess_batch(bundle.train_texts, mode=mode_key)
        x_val = preprocess_batch(bundle.val_texts, mode=mode_key)
        pipe = build_tfidf_lr_pipeline(c_value=c_value, seed=seed)
        pipe.fit(x_train, bundle.train_labels)
        val_preds = pipe.predict(x_val)
        metrics = compute_classification_metrics(bundle.val_labels, val_preds)
        vocab_size = len(pipe.named_steps["tfidf"].vocabulary_)
        rows.append(
            {
                "mode": mode_key,
                "description": description,
                "vocab_size": vocab_size,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "val_weighted_f1": metrics["weighted_f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)


def tune_and_train_baseline(
    bundle: DatasetBundle,
    c_grid: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0),
    seed: int = SEED,
    save_path: Path = BASELINE_MODEL_PATH,
) -> dict:
    """Tune Logistic Regression regularization C on validation split and save artifact."""
    prep_mode = "classical_domain_aware"
    x_train = preprocess_batch(bundle.train_texts, mode=prep_mode)
    x_val = preprocess_batch(bundle.val_texts, mode=prep_mode)
    x_test = preprocess_batch(bundle.test_texts, mode=prep_mode)

    tuning_rows: list[dict] = []
    best_c = float(c_grid[0])
    best_val_f1 = -1.0
    best_pipe: Pipeline | None = None

    for c_val in c_grid:
        pipe = build_tfidf_lr_pipeline(c_value=float(c_val), seed=seed)
        pipe.fit(x_train, bundle.train_labels)
        val_preds = pipe.predict(x_val)
        val_metrics = compute_classification_metrics(bundle.val_labels, val_preds)
        tuning_rows.append(
            {
                "C": float(c_val),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
            }
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_c = float(c_val)
            best_pipe = pipe

    assert best_pipe is not None

    # Reference LinearSVC comparison to show LR matches linear SVM while providing softmax probs.
    svc_pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    max_features=15000,
                    token_pattern=r"(?u)\b[a-zA-Z0-9_']+\b",
                ),
            ),
            ("clf", LinearSVC(C=1.0, random_state=seed)),
        ]
    )
    svc_pipe.fit(x_train, bundle.train_labels)
    svc_val_metrics = compute_classification_metrics(bundle.val_labels, svc_pipe.predict(x_val))
    svc_test_metrics = compute_classification_metrics(bundle.test_labels, svc_pipe.predict(x_test))

    artifact = {
        "pipeline": best_pipe,
        "label_names": list(bundle.label_names),
        "best_c": best_c,
        "preprocessing_mode": prep_mode,
        "val_tuning_results": tuning_rows,
        "linear_svc_ref": {
            "val": svc_val_metrics,
            "test": svc_test_metrics,
        },
        "seed": seed,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, save_path)
    return artifact


def load_baseline(path: Path = BASELINE_MODEL_PATH) -> dict:
    """Load persisted baseline artifact from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Baseline model artifact not found at {path}")
    return joblib.load(path)


def predict_baseline_probs(artifact: dict, texts: Sequence[str]) -> np.ndarray:
    """Predict calibrated class probability matrix (N x 77) using the baseline."""
    mode = artifact.get("preprocessing_mode", "classical_domain_aware")
    cleaned = preprocess_batch(texts, mode=mode)
    return artifact["pipeline"].predict_proba(cleaned)
