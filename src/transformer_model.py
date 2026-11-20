"""Fine-tuned Transformer (DistilBERT) training and batched/single inference."""

from __future__ import annotations

import inspect
import json
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.config import MAX_LENGTH, SEED, TRANSFORMER_BASE_MODEL, TRANSFORMER_MODEL_DIR
from src.data import DatasetBundle
from src.evaluation import compute_classification_metrics
from src.preprocessing import preprocess_batch


def set_global_seed(seed: int = SEED) -> None:
    """Set Python, NumPy, and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Select the best available accelerator (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transformer_weights_exist(model_dir: Path = TRANSFORMER_MODEL_DIR) -> bool:
    """Return True if a saved Hugging Face model checkpoint and tokenizer exist."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return False
    has_config = (model_dir / "config.json").is_file()
    has_weights = (model_dir / "model.safetensors").is_file() or (
        model_dir / "pytorch_model.bin"
    ).is_file()
    return has_config and has_weights


def _make_training_arguments(**kwargs) -> TrainingArguments:
    """Build TrainingArguments safely across transformers 4.x and 5.x."""
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    if "eval_strategy" in params and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif "evaluation_strategy" in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**filtered)


def _make_trainer(**kwargs) -> Trainer:
    """Build Trainer safely across transformers 4.x and 5.x."""
    sig = inspect.signature(Trainer.__init__)
    params = sig.parameters
    if "processing_class" in params and "tokenizer" in kwargs:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    elif "tokenizer" in params and "processing_class" in kwargs:
        kwargs["tokenizer"] = kwargs.pop("processing_class")
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return Trainer(**filtered)


def train_distilbert(
    bundle: DatasetBundle,
    output_dir: Path = TRANSFORMER_MODEL_DIR,
    model_name: str = TRANSFORMER_BASE_MODEL,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 5e-5,
    max_length: int = MAX_LENGTH,
    seed: int = SEED,
) -> dict:
    """Fine-tune DistilBERT on BANKING77 and save best checkpoint by val Macro F1."""
    set_global_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "_checkpoints"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _to_hf_dataset(texts: Sequence[str], labels: Sequence[int]) -> Dataset:
        cleaned = preprocess_batch(texts, mode="transformer_minimal")
        ds = Dataset.from_dict({"text": cleaned, "label": list(labels)})
        return ds.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
            ),
            batched=True,
            remove_columns=["text"],
        )

    train_ds = _to_hf_dataset(bundle.train_texts, bundle.train_labels)
    val_ds = _to_hf_dataset(bundle.val_texts, bundle.val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(bundle.label_names),
        id2label={int(k): v for k, v in bundle.id2label.items()},
        label2id={k: int(v) for k, v in bundle.label2id.items()},
    )

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_classification_metrics(labels, preds)

    total_steps = max(1, math.ceil(len(train_ds) / batch_size) * epochs)
    warmup_steps = max(1, int(0.06 * total_steps))

    training_args = _make_training_arguments(
        output_dir=str(ckpt_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
    )

    trainer = _make_trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    train_result = trainer.train()
    val_eval = trainer.evaluate(val_ds)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    epoch_history = [
        {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in entry.items()}
        for entry in trainer.state.log_history
        if "eval_macro_f1" in entry
    ]
    meta = {
        "base_model": model_name,
        "dataset_source": bundle.dataset_source,
        "num_labels": len(bundle.label_names),
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "device_trained": get_device(),
        "train_runtime_sec": float(train_result.metrics.get("train_runtime", 0.0)),
        "val_metrics": {k: float(v) for k, v in val_eval.items() if isinstance(v, (int, float))},
        "epoch_history": epoch_history,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def load_transformer(model_dir: Path = TRANSFORMER_MODEL_DIR):
    """Load saved fine-tuned Transformer and tokenizer onto the active device."""
    model_dir = Path(model_dir)
    if not transformer_weights_exist(model_dir):
        raise FileNotFoundError(
            f"Fine-tuned Transformer weights not found in '{model_dir}'. "
            "Run notebooks/colab_train_transformer.ipynb on Colab T4 GPU and "
            "extract the saved folder into models/transformer_intent/."
        )
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_transformer_batch(
    texts: Sequence[str],
    model,
    tokenizer,
    device: str,
    label_names: Sequence[str] | None = None,
    batch_size: int = 64,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """Run batched inference and return probability matrix aligned to label_names."""
    cleaned = preprocess_batch(texts, mode="transformer_minimal")
    if not cleaned:
        num_classes = len(label_names) if label_names is not None else model.config.num_labels
        return np.zeros((0, num_classes), dtype=np.float32)

    probs_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(cleaned), batch_size):
            batch_texts = cleaned[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_chunks.append(probs)

    all_probs = np.vstack(probs_chunks)

    # Align probability columns if saved config id2label ordering differs from canonical label_names.
    if label_names is not None and hasattr(model.config, "label2id"):
        cfg_l2i = {str(k): int(v) for k, v in model.config.label2id.items()}
        if all(lbl in cfg_l2i for lbl in label_names):
            col_order = [cfg_l2i[lbl] for lbl in label_names]
            if col_order != list(range(len(label_names))):
                all_probs = all_probs[:, col_order]

    return all_probs
