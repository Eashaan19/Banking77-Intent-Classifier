"""Dataset loading, canonical label mapping, stratified splitting, and OOS probes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    PRIMARY_DATASET_ID,
    PRIMARY_DATASET_REVISION,
    SEED,
    VAL_SIZE,
)


@dataclass
class DatasetBundle:
    """Container holding stratified train/val/test splits and canonical label maps."""

    train_texts: list[str]
    train_labels: list[int]
    val_texts: list[str]
    val_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    label_names: list[str]
    label2id: dict[str, int]
    id2label: dict[int, str]
    dataset_source: str


def _extract_texts_and_label_strings(split_ds) -> tuple[list[str], list[str]]:
    texts = [str(x) for x in split_ds["text"]]
    cols = set(split_ds.column_names)
    if "label_text" in cols:
        label_strs = [str(x) for x in split_ds["label_text"]]
    else:
        feat = split_ds.features["label"]
        if hasattr(feat, "int2str"):
            label_strs = [str(feat.int2str(int(idx))) for idx in split_ds["label"]]
        else:
            label_strs = [str(idx) for idx in split_ds["label"]]
    return texts, label_strs


def load_banking77_splits(
    val_size: float = VAL_SIZE,
    seed: int = SEED,
    use_local_cache: bool = True,
    cache_path: Path = DATA_DIR / "banking77_canonical.json",
) -> DatasetBundle:
    """Load BANKING77 with canonical label mapping and stratified 85/15 train/val split.

    Caches the official train/test texts and string labels to disk after the first
    Hugging Face Hub download so subsequent local notebook runs work offline.
    """
    cache_path = Path(cache_path)
    if use_local_cache and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        raw_train_texts = payload["train_texts"]
        raw_train_str = payload["train_label_texts"]
        test_texts = payload["test_texts"]
        test_str = payload["test_label_texts"]
        source_desc = payload.get("dataset_source", PRIMARY_DATASET_ID)
    else:
        load_kwargs = {}
        if PRIMARY_DATASET_REVISION:
            load_kwargs["revision"] = PRIMARY_DATASET_REVISION
        ds = load_dataset(PRIMARY_DATASET_ID, **load_kwargs)
        raw_train_texts, raw_train_str = _extract_texts_and_label_strings(ds["train"])
        test_texts, test_str = _extract_texts_and_label_strings(ds["test"])
        source_desc = PRIMARY_DATASET_ID
        if use_local_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "dataset_source": source_desc,
                        "train_texts": raw_train_texts,
                        "train_label_texts": raw_train_str,
                        "test_texts": test_texts,
                        "test_label_texts": test_str,
                    }
                ),
                encoding="utf-8",
            )

    label_names = sorted(set(raw_train_str) | set(test_str), key=lambda s: s.lower())
    label2id = {name: idx for idx, name in enumerate(label_names)}
    id2label = {idx: name for idx, name in enumerate(label_names)}

    raw_train_labels = [label2id[s] for s in raw_train_str]
    test_labels = [label2id[s] for s in test_str]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        raw_train_texts,
        raw_train_labels,
        test_size=val_size,
        stratify=raw_train_labels,
        random_state=seed,
    )

    assert len(label_names) == 77, f"Expected 77 classes, found {len(label_names)}"
    assert len(set(train_labels)) == 77, "Missing classes in train split"
    assert len(set(val_labels)) == 77, "Missing classes in val split"
    assert len(set(test_labels)) == 77, "Missing classes in test split"

    return DatasetBundle(
        train_texts=list(train_texts),
        train_labels=list(train_labels),
        val_texts=list(val_texts),
        val_labels=list(val_labels),
        test_texts=list(test_texts),
        test_labels=list(test_labels),
        label_names=label_names,
        label2id=label2id,
        id2label=id2label,
        dataset_source=source_desc,
    )


def get_oos_and_edge_probes() -> list[dict[str, str]]:
    """Return a 20-query probe suite spanning OOS, ambiguous, and multi-intent cases."""
    return [
        # 12 Out-of-scope (non-banking) queries
        {
            "text": "I want to book a flight to Tokyo next week.",
            "category": "out_of_scope",
            "note": "Travel booking (Topic #7 non-banking example)",
        },
        {
            "text": "What is the weather forecast for Sydney tomorrow?",
            "category": "out_of_scope",
            "note": "Weather query",
        },
        {
            "text": "Cancel my Amazon Prime subscription and package order.",
            "category": "out_of_scope",
            "note": "E-commerce order cancellation",
        },
        {
            "text": "Can you recommend a good Italian restaurant nearby?",
            "category": "out_of_scope",
            "note": "Dining recommendation",
        },
        {
            "text": "Reset my Netflix password because I forgot it.",
            "category": "out_of_scope",
            "note": "Third-party streaming password reset",
        },
        {
            "text": "How do I bake sourdough bread at home?",
            "category": "out_of_scope",
            "note": "Cooking instructions",
        },
        {
            "text": "Schedule a dentist appointment for Friday morning.",
            "category": "out_of_scope",
            "note": "Healthcare appointment",
        },
        {
            "text": "Who won the World Cup final in 2022?",
            "category": "out_of_scope",
            "note": "General sports trivia",
        },
        {
            "text": "Write a Python script to sort a dictionary by values.",
            "category": "out_of_scope",
            "note": "Coding assistance",
        },
        {
            "text": "Reserve a two-bedroom hotel room in Melbourne.",
            "category": "out_of_scope",
            "note": "Hotel reservation",
        },
        {
            "text": "My laptop Wi-Fi keeps disconnecting after the Windows update.",
            "category": "out_of_scope",
            "note": "IT hardware troubleshooting",
        },
        {
            "text": "Can you tell me a bedtime joke?",
            "category": "out_of_scope",
            "note": "Chit-chat / entertainment",
        },
        # 4 Ambiguous / ultra-short banking queries
        {
            "text": "card issue",
            "category": "ambiguous_short",
            "note": "Underspecified — could be declined, lost, or broken card",
        },
        {
            "text": "help please",
            "category": "ambiguous_short",
            "note": "No actionable intent signal",
        },
        {
            "text": "payment",
            "category": "ambiguous_short",
            "note": "Single word — pending, declined, fee, or unrecognized?",
        },
        {
            "text": "money problem",
            "category": "ambiguous_short",
            "note": "Vague complaint requiring clarification",
        },
        # 4 Multi-intent banking queries (single-label limitation test)
        {
            "text": "My card was stolen and I also want to close my account permanently.",
            "category": "multi_intent",
            "note": "Combines lost_or_stolen_card + terminate_account",
        },
        {
            "text": "Why was my card declined and how do I change my PIN?",
            "category": "multi_intent",
            "note": "Combines declined_card_payment + change_pin",
        },
        {
            "text": "I forgot my passcode and my top-up also failed twice.",
            "category": "multi_intent",
            "note": "Combines passcode_forgotten + top_up_failed",
        },
        {
            "text": "Track my physical card delivery and check today's euro exchange rate.",
            "category": "multi_intent",
            "note": "Combines card_arrival + exchange_rate",
        },
    ]


def get_demo_presets() -> dict[str, list[str]]:
    """Return 4 categorized preset query groups for the interactive Gradio demo."""
    return {
        "Clear In-Scope (What It Does)": [
            "I lost my debit card and need to freeze it right away.",
            "How do I activate the new physical card that just arrived?",
            "I forgot my app passcode and cannot log in.",
        ],
        "Paraphrased & Negation-Heavy (Why Transformer Beats TF-IDF)": [
            "My contactless tap isn't going through at checkout even though I have funds.",
            "I never received the refund the merchant promised last Tuesday.",
            "My account balance hasn't gone up even though my salary transfer went through.",
        ],
        "Fine-Grained Overlap (Subtle Disambiguation)": [
            "Why is there still a pending charge on my card from yesterday?",
            "I noticed an extra charge on my statement that I didn't authorize.",
            "The ATM gave me less cash than the amount deducted from my account.",
        ],
        "Out-of-Scope & Multi-Intent (Limitations)": [
            "I want to book a flight to Tokyo next week.",
            "My card was stolen and I also want to close my account permanently.",
            "card issue",
        ],
    }
