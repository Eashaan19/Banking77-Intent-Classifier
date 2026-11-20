"""Evaluation, latency benchmarking, error analysis, and plotting utilities."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

sns.set_theme(style="whitegrid", context="notebook")


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Compute Accuracy plus Macro and Weighted Precision, Recall, and F1."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }


def benchmark_single_query_latency(
    predict_fn: Callable[[str], object],
    sample_texts: Sequence[str],
    n_warmup: int = 5,
    n_runs: int = 50,
) -> dict[str, float]:
    """Measure single-query inference latency (ms) on the active device."""
    import time

    if not sample_texts:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}

    for i in range(n_warmup):
        predict_fn(sample_texts[i % len(sample_texts)])

    timings_ms: list[float] = []
    for i in range(n_runs):
        text = sample_texts[i % len(sample_texts)]
        t0 = time.perf_counter()
        predict_fn(text)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(timings_ms, dtype=float)
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def get_path_size_mb(path: Path) -> float:
    """Return size on disk in MB for a file or directory."""
    path = Path(path)
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024.0 * 1024.0)
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / (1024.0 * 1024.0)


def get_per_class_f1_summary(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
    top_n: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return best and worst performing intent classes ranked by F1."""
    labels = list(range(len(label_names)))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    df = pd.DataFrame(
        {
            "intent": label_names,
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": support,
        }
    )
    df_sorted = df.sort_values(by=["f1", "support"], ascending=[True, False]).reset_index(drop=True)
    worst = df_sorted.head(top_n).copy()
    best = df_sorted.tail(top_n).sort_values(by=["f1", "support"], ascending=[False, False]).reset_index(drop=True)
    return best, worst


def get_top_confused_pairs(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
    top_k: int = 15,
) -> pd.DataFrame:
    """Identify the most frequent (true_intent -> predicted_intent) misclassifications."""
    counts: Counter[tuple[int, int]] = Counter()
    for yt, yp in zip(y_true, y_pred):
        if yt != yp:
            counts[(int(yt), int(yp))] += 1

    rows = []
    for (yt, yp), cnt in counts.most_common(top_k):
        true_name = label_names[yt]
        pred_name = label_names[yp]
        rows.append(
            {
                "true_intent": true_name,
                "predicted_intent": pred_name,
                "pair_label": f"{true_name} -> {pred_name}",
                "count": cnt,
            }
        )
    return pd.DataFrame(rows, columns=["true_intent", "predicted_intent", "pair_label", "count"])


def evaluate_confidence_thresholds(
    y_true: Sequence[int],
    probs: np.ndarray,
    oos_probs: np.ndarray | None = None,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Evaluate automation rate, automated accuracy, and OOS rejection rate across tau."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.30, 0.96, 0.05), 2)

    y_true_arr = np.asarray(y_true)
    preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)
    oos_max_probs = np.max(oos_probs, axis=1) if oos_probs is not None and len(oos_probs) > 0 else None

    rows = []
    for tau in thresholds:
        tau_val = float(tau)
        accepted = max_probs >= tau_val
        coverage = float(np.mean(accepted))
        if np.any(accepted):
            auto_acc = float(np.mean(preds[accepted] == y_true_arr[accepted]))
        else:
            auto_acc = 1.0
        escalation_rate = 1.0 - coverage
        overall_error_rate = float(np.mean((preds != y_true_arr) & accepted))
        row = {
            "threshold": tau_val,
            "automation_coverage": coverage,
            "escalation_rate": escalation_rate,
            "automated_accuracy": auto_acc,
            "accepted_error_rate": overall_error_rate,
        }
        if oos_max_probs is not None:
            row["oos_rejection_rate"] = float(np.mean(oos_max_probs < tau_val))
        rows.append(row)
    return pd.DataFrame(rows)


def select_recommended_threshold(
    curve_df: pd.DataFrame,
    min_coverage: float = 0.80,
) -> float:
    """Pick the highest threshold maintaining at least min_coverage on validation data."""
    eligible = curve_df[curve_df["automation_coverage"] >= min_coverage]
    if eligible.empty:
        return float(curve_df.iloc[0]["threshold"])
    best_row = eligible.sort_values(
        by=["automated_accuracy", "threshold"], ascending=[False, False]
    ).iloc[0]
    return float(best_row["threshold"])


def _infer_error_category(
    text: str,
    true_label: str,
    pred_label: str,
) -> str:
    """Assign a readable heuristic failure tag for error analysis tables."""
    t_lower = text.lower()
    if any(neg in t_lower for neg in ("not ", "n't", "never", "no ", "neither")):
        return "Negation / polarity nuance"
    if len(text.split()) <= 5:
        return "Short / underspecified query"
    true_tokens = set(true_label.split("_"))
    pred_tokens = set(pred_label.split("_"))
    if len(true_tokens & pred_tokens) >= 1:
        return "Fine-grained semantic overlap"
    return "Paraphrase / lexical gap"


def extract_error_cases(
    texts: Sequence[str],
    y_true: Sequence[int],
    base_preds: Sequence[int],
    base_probs: np.ndarray,
    label_names: Sequence[str],
    trans_preds: Sequence[int] | None = None,
    trans_probs: np.ndarray | None = None,
    max_cases: int = 10,
) -> pd.DataFrame:
    """Extract representative misclassifications for report & demo error analysis.

    When Transformer predictions are provided, returns a balanced mix of:
      1) Cases where TF-IDF failed and Transformer succeeded (why Transformer works)
      2) Cases where Transformer also failed (genuine limitations)
    """
    rows: list[dict] = []

    if trans_preds is not None and trans_probs is not None:
        wins: list[dict] = []
        losses: list[dict] = []
        for idx, text in enumerate(texts):
            yt = int(y_true[idx])
            bp = int(base_preds[idx])
            tp = int(trans_preds[idx])
            b_conf = float(base_probs[idx, bp])
            t_conf = float(trans_probs[idx, tp])
            true_name = label_names[yt]
            base_name = label_names[bp]
            trans_name = label_names[tp]

            if bp != yt and tp == yt:
                wins.append(
                    {
                        "case_type": "Transformer Win (Baseline Wrong)",
                        "text": text,
                        "true_intent": true_name,
                        "baseline_pred": base_name,
                        "baseline_conf": round(b_conf, 3),
                        "transformer_pred": trans_name,
                        "transformer_conf": round(t_conf, 3),
                        "likely_cause": _infer_error_category(text, true_name, base_name),
                    }
                )
            elif tp != yt:
                losses.append(
                    {
                        "case_type": "Transformer Error (Limitation)",
                        "text": text,
                        "true_intent": true_name,
                        "baseline_pred": base_name,
                        "baseline_conf": round(b_conf, 3),
                        "transformer_pred": trans_name,
                        "transformer_conf": round(t_conf, 3),
                        "likely_cause": _infer_error_category(text, true_name, trans_name),
                    }
                )

        half = max(1, max_cases // 2)
        wins_sorted = sorted(wins, key=lambda r: r["transformer_conf"], reverse=True)
        losses_sorted = sorted(losses, key=lambda r: r["transformer_conf"], reverse=True)
        rows.extend(wins_sorted[:half])
        rows.extend(losses_sorted[: max_cases - len(rows)])
        return pd.DataFrame(rows)

    for idx, text in enumerate(texts):
        yt = int(y_true[idx])
        bp = int(base_preds[idx])
        if bp != yt:
            true_name = label_names[yt]
            base_name = label_names[bp]
            rows.append(
                {
                    "case_type": "Baseline Error",
                    "text": text,
                    "true_intent": true_name,
                    "baseline_pred": base_name,
                    "baseline_conf": round(float(base_probs[idx, bp]), 3),
                    "transformer_pred": "N/A",
                    "transformer_conf": np.nan,
                    "likely_cause": _infer_error_category(text, true_name, base_name),
                }
            )
    rows.sort(key=lambda r: r["baseline_conf"], reverse=True)
    return pd.DataFrame(rows[:max_cases])


def export_metrics_tables(
    metrics_df: pd.DataFrame,
    csv_path: Path,
    md_path: Path,
) -> None:
    """Save metrics comparison table in CSV and Markdown formats."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(csv_path, index=False)

    display_df = metrics_df.copy()
    for col in display_df.columns:
        if display_df[col].dtype.kind in {"f"}:
            if "ms" in col or "mb" in col:
                display_df[col] = display_df[col].map(lambda v: f"{v:.2f}")
            else:
                display_df[col] = display_df[col].map(lambda v: f"{v:.4f}")
    md_text = display_df.to_markdown(index=False)
    md_path.write_text(md_text + "\n", encoding="utf-8")


def plot_class_distribution(
    train_labels: Sequence[int],
    label_names: Sequence[str],
    save_path: Path,
) -> None:
    """Plot BANKING77 training class distribution and top/bottom sample counts."""
    counts = pd.Series(train_labels).value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(counts.values, bins=18, color="#2563eb", edgecolor="white", alpha=0.9)
    ax.axvline(counts.mean(), color="#dc2626", linestyle="--", linewidth=1.6, label=f"Mean = {counts.mean():.1f}")
    ax.set_title("BANKING77 Training Set — Distribution of Samples per Intent (77 Classes)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Utterances per Intent Class (85% Train Split)")
    ax.set_ylabel("Number of Intent Classes")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_model_comparison(metrics_df: pd.DataFrame, save_path: Path) -> None:
    """Plot grouped bar chart comparing models across core classification metrics."""
    metric_cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    nice_names = {
        "accuracy": "Accuracy",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
    }
    melted = metrics_df.melt(
        id_vars=["model"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="score",
    )
    melted["metric_label"] = melted["metric"].map(nice_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=melted,
        x="metric_label",
        y="score",
        hue="model",
        palette="Set2",
        ax=ax,
    )
    min_score = max(0.0, float(melted["score"].min()) - 0.12)
    ax.set_ylim(min_score, 1.02)
    ax.set_title("Model Progression — Test Set Classification Performance (BANKING77)", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="Model", frameon=True, loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_top_confused_pairs(
    confused_df: pd.DataFrame,
    model_name: str,
    save_path: Path,
) -> None:
    """Plot horizontal bar chart of the top misclassified (True -> Predicted) pairs."""
    if confused_df.empty:
        return
    ordered = confused_df.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.barh(ordered["pair_label"], ordered["count"], color="#e11d48", alpha=0.88)
    ax.set_title(f"Top {len(ordered)} Confused Intent Pairs — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Misclassification Count on Test Set")
    ax.set_ylabel("")
    for bar in ax.patches:
        w = bar.get_width()
        ax.text(w + 0.15, bar.get_y() + bar.get_height() / 2.0, f"{int(w)}", va="center", fontsize=8.5)
    ax.set_xlim(0, float(ordered["count"].max()) * 1.15)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_focused_confusion_cluster(
    y_true: Sequence[int],
    primary_preds: Sequence[int],
    label_names: Sequence[str],
    save_path: Path,
    baseline_preds: Sequence[int] | None = None,
    n_classes: int = 10,
) -> None:
    """Plot a focused NxN confusion heatmap around the highest-error intents."""
    y_true_arr = np.asarray(y_true)
    prim_arr = np.asarray(primary_preds)
    err_counts = Counter()
    for yt, yp in zip(y_true_arr, prim_arr):
        if yt != yp:
            err_counts[int(yt)] += 1
            err_counts[int(yp)] += 1

    focus_ids = [idx for idx, _ in err_counts.most_common(n_classes)]
    if len(focus_ids) < 2:
        focus_ids = list(range(min(n_classes, len(label_names))))

    focus_names = [label_names[i] for i in focus_ids]
    cm_prim = confusion_matrix(y_true_arr, prim_arr, labels=focus_ids)

    if baseline_preds is not None:
        cm_base = confusion_matrix(y_true_arr, np.asarray(baseline_preds), labels=focus_ids)
        vmax = max(int(cm_base.max()), int(cm_prim.max()), 1)
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), sharey=True)
        sns.heatmap(
            cm_base, annot=True, fmt="d", cmap="Blues", vmin=0, vmax=vmax,
            xticklabels=focus_names, yticklabels=focus_names, ax=axes[0], cbar=False
        )
        axes[0].set_title("TF-IDF + Logistic Regression (Baseline)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Predicted Intent")
        axes[0].set_ylabel("True Intent")
        axes[0].tick_params(axis="x", rotation=45, labelsize=8)
        axes[0].tick_params(axis="y", rotation=0, labelsize=8)

        sns.heatmap(
            cm_prim, annot=True, fmt="d", cmap="Greens", vmin=0, vmax=vmax,
            xticklabels=focus_names, yticklabels=focus_names, ax=axes[1], cbar=True
        )
        axes[1].set_title("Fine-Tuned DistilBERT (Primary)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Predicted Intent")
        axes[1].tick_params(axis="x", rotation=45, labelsize=8)
        fig.suptitle(f"Focused Confusion Cluster ({n_classes} Highest-Error Intents)", fontsize=13, fontweight="bold")
    else:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm_prim, annot=True, fmt="d", cmap="Blues",
            xticklabels=focus_names, yticklabels=focus_names, ax=ax
        )
        ax.set_title(f"Focused Confusion Cluster ({n_classes} Highest-Error Intents)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Intent")
        ax.set_ylabel("True Intent")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_confidence_coverage_curve(
    curve_df: pd.DataFrame,
    recommended_tau: float,
    model_name: str,
    save_path: Path,
) -> None:
    """Plot automated accuracy vs. automation coverage & OOS rejection across tau."""
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(
        curve_df["threshold"],
        curve_df["automated_accuracy"],
        marker="o",
        linewidth=2.0,
        color="#16a34a",
        label="Automated Accuracy (In-Scope Accepted)",
    )
    ax.plot(
        curve_df["threshold"],
        curve_df["automation_coverage"],
        marker="s",
        linewidth=2.0,
        color="#2563eb",
        label="Automation Coverage (% Queries Auto-Routed)",
    )
    if "oos_rejection_rate" in curve_df.columns:
        ax.plot(
            curve_df["threshold"],
            curve_df["oos_rejection_rate"],
            marker="^",
            linestyle="--",
            linewidth=2.0,
            color="#d97706",
            label="Out-of-Scope Escalation Rate (Probe Set)",
        )
    ax.axvline(
        recommended_tau,
        color="#dc2626",
        linestyle=":",
        linewidth=1.8,
        label=f"Recommended Threshold tau = {recommended_tau:.2f}",
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Confidence Threshold Tradeoff — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Confidence Threshold (tau)")
    ax.set_ylabel("Rate / Score")
    ax.legend(frameon=True, loc="lower left")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
