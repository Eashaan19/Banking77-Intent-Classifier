"""Unit and smoke tests for the banking chatbot intent classification pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import load_baseline, predict_baseline_probs
from src.config import BASELINE_MODEL_PATH, TRANSFORMER_MODEL_DIR
from src.data import get_demo_presets, get_oos_and_edge_probes, load_banking77_splits
from src.inference import IntentClassifierService
from src.preprocessing import (
    clean_classical_aggressive,
    clean_classical_domain_aware,
    clean_transformer_minimal,
)
from src.transformer_model import (
    load_transformer,
    predict_transformer_batch,
    transformer_weights_exist,
)


def test_preprocessing_preserves_negation_and_normalizes_currency() -> None:
    text = "My contactless card isn't working for a $49.99 charge! PIN 1234"
    classical = clean_classical_domain_aware(text)
    assert "isn't" in classical
    assert "_amount_" in classical
    assert "_num_" in classical

    minimal = clean_transformer_minimal("  Hello   world!\u200b ")
    assert minimal == "Hello world!"

    aggressive = clean_classical_aggressive("My card is not working for these cards")
    assert "not" not in aggressive.split()
    assert "card" in aggressive.split()


def test_dataset_splits_and_probes_integrity() -> None:
    bundle = load_banking77_splits(use_local_cache=True)
    assert len(bundle.label_names) == 77
    assert bundle.label_names == sorted(bundle.label_names, key=str.lower)
    assert len(bundle.train_texts) > 0
    assert len(bundle.val_texts) > 0
    assert len(bundle.test_texts) == 3076

    assert set(bundle.train_labels) == set(range(77))
    assert set(bundle.val_labels) == set(range(77))
    assert set(bundle.test_labels) == set(range(77))

    probes = get_oos_and_edge_probes()
    assert len(probes) == 20
    assert all("text" in p and "category" in p for p in probes)

    presets = get_demo_presets()
    assert len(presets) == 4


def test_baseline_model_probabilities_valid() -> None:
    assert BASELINE_MODEL_PATH.exists(), "Baseline model must be trained before running tests."
    artifact = load_baseline(BASELINE_MODEL_PATH)
    samples = [
        "I lost my debit card and need to freeze it immediately.",
        "What is the exchange rate for euros?",
    ]
    probs = predict_baseline_probs(artifact, samples)
    assert probs.shape == (2, 77)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_inference_service_routing_and_escalation() -> None:
    service = IntentClassifierService(
        baseline_path=BASELINE_MODEL_PATH,
        transformer_dir=TRANSFORMER_MODEL_DIR,
    )

    empty_res = service.predict("   ", threshold=0.65)
    assert empty_res["primary"].escalate is True
    assert empty_res["primary"].top_intent == "empty_input"

    clear_query = "I lost my debit card and need to freeze it right away"
    routed = service.predict(clear_query, top_k=3, threshold=0.01, compare_baseline=True)
    prim = routed["primary"]
    assert prim.escalate is False
    assert len(prim.top_k) == 3
    assert prim.top_k[0][1] >= prim.top_k[1][1] >= prim.top_k[2][1]
    assert prim.latency_ms > 0.0

    forced_esc = service.predict(clear_query, top_k=3, threshold=0.9999, compare_baseline=False)
    assert forced_esc["primary"].escalate is True
    assert forced_esc["baseline"] is None


def test_gradio_analyze_query_across_presets() -> None:
    import app

    presets = get_demo_presets()
    for category, queries in presets.items():
        sample = queries[0]
        outputs = app.analyze_query(sample, threshold=0.65, compare_baseline=True)
        status_md, top3_dict, meta_md, *_ = outputs
        assert isinstance(status_md, str) and len(status_md) > 0, f"Empty status for {category}"
        assert isinstance(top3_dict, dict) and len(top3_dict) == 3, f"Expected 3 classes for {category}"
        assert abs(sum(top3_dict.values()) - sum(sorted(top3_dict.values(), reverse=True))) < 1e-6
        assert "ms" in meta_md


@pytest.mark.skipif(
    not transformer_weights_exist(TRANSFORMER_MODEL_DIR),
    reason="Fine-tuned Transformer weights not yet placed in models/transformer_intent/",
)
def test_transformer_model_loads_and_predicts() -> None:
    bundle = load_banking77_splits(use_local_cache=True)
    model, tokenizer, device = load_transformer(TRANSFORMER_MODEL_DIR)
    probs = predict_transformer_batch(
        ["I lost my card and want to block it", "Why is my card payment still pending?"],
        model,
        tokenizer,
        device,
        label_names=bundle.label_names,
    )
    assert probs.shape == (2, 77)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
