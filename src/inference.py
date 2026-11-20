"""Unified inference and confidence-gated human escalation service."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.baseline import load_baseline, predict_baseline_probs
from src.config import (
    BASELINE_MODEL_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MARGIN_THRESHOLD,
    MAX_LENGTH,
    TRANSFORMER_MODEL_DIR,
    get_routing_action,
)
from src.transformer_model import (
    load_transformer,
    predict_transformer_batch,
    transformer_weights_exist,
)


@dataclass
class PredictionResult:
    """Structured intent prediction output for CLI, tests, and Gradio UI."""

    model_name: str
    top_intent: str
    confidence: float
    margin: float
    escalate: bool
    escalation_reason: str
    routing_action: str
    top_k: list[tuple[str, float]]
    latency_ms: float


class IntentClassifierService:
    """Loads saved Baseline and (when available) Fine-tuned DistilBERT models."""

    def __init__(
        self,
        baseline_path=BASELINE_MODEL_PATH,
        transformer_dir=TRANSFORMER_MODEL_DIR,
        margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    ) -> None:
        self.baseline_artifact = load_baseline(baseline_path)
        self.label_names: list[str] = list(self.baseline_artifact["label_names"])
        self.margin_threshold = float(margin_threshold)

        self.has_transformer = transformer_weights_exist(transformer_dir)
        self.transformer_model = None
        self.transformer_tokenizer = None
        self.transformer_device = "cpu"
        if self.has_transformer:
            (
                self.transformer_model,
                self.transformer_tokenizer,
                self.transformer_device,
            ) = load_transformer(transformer_dir)

    def _build_result(
        self,
        model_name: str,
        probs_row: np.ndarray,
        latency_ms: float,
        top_k: int,
        threshold: float,
    ) -> PredictionResult:
        order = np.argsort(probs_row)[::-1]
        top_indices = order[:top_k]
        top_pairs = [(self.label_names[int(i)], float(probs_row[int(i)])) for i in top_indices]
        top_intent, top_conf = top_pairs[0]
        second_conf = top_pairs[1][1] if len(top_pairs) > 1 else 0.0
        margin = float(top_conf - second_conf)

        low_conf = top_conf < threshold
        ambiguous_margin = margin < self.margin_threshold and top_conf < max(threshold + 0.10, 0.80)
        escalate = bool(low_conf)
        if low_conf:
            reason = f"Top confidence ({top_conf:.1%}) is below threshold ({threshold:.0%})."
        elif ambiguous_margin:
            reason = f"Close runner-up margin ({margin:.1%}); monitor for ambiguity."
        else:
            reason = f"Confident match ({top_conf:.1%} >= {threshold:.0%})."

        action = (
            "Escalate to Human Banking Specialist (Uncertain / Possible Out-of-Scope Query)"
            if escalate
            else get_routing_action(top_intent)
        )
        return PredictionResult(
            model_name=model_name,
            top_intent=top_intent,
            confidence=float(top_conf),
            margin=margin,
            escalate=escalate,
            escalation_reason=reason,
            routing_action=action,
            top_k=top_pairs,
            latency_ms=float(latency_ms),
        )

    def predict_baseline(
        self,
        text: str,
        top_k: int = 3,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> PredictionResult:
        if not text or not text.strip():
            return PredictionResult(
                model_name="TF-IDF + Logistic Regression",
                top_intent="empty_input",
                confidence=0.0,
                margin=0.0,
                escalate=True,
                escalation_reason="Empty or whitespace-only query.",
                routing_action="Prompt user for a valid banking question.",
                top_k=[],
                latency_ms=0.0,
            )
        t0 = time.perf_counter()
        probs = predict_baseline_probs(self.baseline_artifact, [text])[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._build_result(
            model_name="TF-IDF + Logistic Regression",
            probs_row=probs,
            latency_ms=latency_ms,
            top_k=top_k,
            threshold=threshold,
        )

    def predict_transformer(
        self,
        text: str,
        top_k: int = 3,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> PredictionResult:
        if not self.has_transformer:
            raise RuntimeError("Transformer model weights are not installed locally yet.")
        if not text or not text.strip():
            return PredictionResult(
                model_name="Fine-Tuned DistilBERT",
                top_intent="empty_input",
                confidence=0.0,
                margin=0.0,
                escalate=True,
                escalation_reason="Empty or whitespace-only query.",
                routing_action="Prompt user for a valid banking question.",
                top_k=[],
                latency_ms=0.0,
            )
        t0 = time.perf_counter()
        probs = predict_transformer_batch(
            [text],
            self.transformer_model,
            self.transformer_tokenizer,
            self.transformer_device,
            label_names=self.label_names,
            batch_size=1,
            max_length=MAX_LENGTH,
        )[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._build_result(
            model_name="Fine-Tuned DistilBERT",
            probs_row=probs,
            latency_ms=latency_ms,
            top_k=top_k,
            threshold=threshold,
        )

    def predict(
        self,
        text: str,
        top_k: int = 3,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        compare_baseline: bool = True,
    ) -> dict[str, PredictionResult | None]:
        """Return primary prediction (DistilBERT if available, else Baseline) and comparison."""
        base_res = self.predict_baseline(text, top_k=top_k, threshold=threshold)
        if self.has_transformer:
            primary_res = self.predict_transformer(text, top_k=top_k, threshold=threshold)
            return {
                "primary": primary_res,
                "baseline": base_res if compare_baseline else None,
            }
        return {
            "primary": base_res,
            "baseline": None,
        }
