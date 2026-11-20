"""Interactive Gradio live demo (Layout B+: Collapsible Sidebar + Wide Workspace)."""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIDENCE_THRESHOLD
from src.inference import IntentClassifierService, PredictionResult

_SERVICE: IntentClassifierService | None = None

SIDEBAR_PRESETS: list[tuple[str, bool, list[tuple[str, str]]]] = [
    (
        "Clear In-Scope",
        True,
        [
            ("Activate physical card", "How do I activate the new physical card that just arrived?"),
            ("Forgot app passcode", "I forgot my app passcode and cannot log in."),
            (
                "Wrong ATM cash amount",
                "The ATM gave me less cash than the amount deducted from my account.",
            ),
        ],
    ),
    (
        "DistilBERT Wins vs TF-IDF",
        True,
        [
            (
                "Pending top-up (Negation)",
                "Topping up my card is not working because I believe the top up is pending",
            ),
            (
                "ATM card failure (Context)",
                "This is very frustrating. I can't get my card to work on any ATM I try and I've tried 3 of them so far. What's going on? Can you please fix this issue immediately?",
            ),
            ("How to exchange FX", "What is the best way to exchange currency?"),
        ],
    ),
    (
        "Fine-Grained Overlap",
        False,
        [
            (
                "Pending card charge",
                "Why is there still a pending charge on my card from yesterday?",
            ),
            (
                "Extra statement charge",
                "I noticed an extra charge on my statement that I didn't authorize.",
            ),
            (
                "Contactless tap failing",
                "My contactless tap isn't going through at checkout even though I have funds.",
            ),
        ],
    ),
    (
        "OOS & Limitations",
        False,
        [
            ("Book Tokyo flight (OOS)", "I want to book a flight to Tokyo next week."),
            ("Vague 'card issue'", "card issue"),
            (
                "Stolen + Close (Multi-intent)",
                "My card was stolen and I also want to close my account permanently.",
            ),
            (
                "Netflix password (Overconf.)",
                "Reset my Netflix password because I forgot it.",
            ),
        ],
    ),
]

CUSTOM_CSS = """
:root {
    color-scheme: light dark;

    --ic-surface-light: #ffffff;
    --ic-surface-dark: #0b0b0e;
    --ic-inset-light: #f4f4f5;
    --ic-inset-dark: #000000;
    --ic-border-light: #71717a;
    --ic-border-dark: #63636e;
    --ic-text-light: #18181b;
    --ic-text-dark: #fafafa;
    --ic-muted-light: #52525b;
    --ic-muted-dark: #d4d4d8;
    --ic-code-bg-light: #e4e4e7;
    --ic-code-bg-dark: #000000;

    --ic-routed-light: #0f766e;
    --ic-routed-dark: #2dd4bf;
    --ic-routed-bg-light: #ccfbf1;
    --ic-routed-bg-dark: #134e4a;

    --ic-escalate-light: #b45309;
    --ic-escalate-dark: #fbbf24;
    --ic-escalate-bg-light: #fef3c7;
    --ic-escalate-bg-dark: #451a03;

    --ic-disagree-light: #be123c;
    --ic-disagree-dark: #fb7185;
    --ic-disagree-bg-light: #ffe4e6;
    --ic-disagree-bg-dark: #4c0519;

    --ic-surface: var(--ic-surface-light);
    --ic-inset: var(--ic-inset-light);
    --ic-border: var(--ic-border-light);
    --ic-text: var(--ic-text-light);
    --ic-muted: var(--ic-muted-light);
    --ic-code-bg: var(--ic-code-bg-light);
    --ic-routed: var(--ic-routed-light);
    --ic-routed-bg: var(--ic-routed-bg-light);
    --ic-escalate: var(--ic-escalate-light);
    --ic-escalate-bg: var(--ic-escalate-bg-light);
    --ic-disagree: var(--ic-disagree-light);
    --ic-disagree-bg: var(--ic-disagree-bg-light);
}

@media (prefers-color-scheme: dark) {
    :root {
        --ic-surface: var(--ic-surface-dark);
        --ic-inset: var(--ic-inset-dark);
        --ic-border: var(--ic-border-dark);
        --ic-text: var(--ic-text-dark);
        --ic-muted: var(--ic-muted-dark);
        --ic-code-bg: var(--ic-code-bg-dark);
        --ic-routed: var(--ic-routed-dark);
        --ic-routed-bg: var(--ic-routed-bg-dark);
        --ic-escalate: var(--ic-escalate-dark);
        --ic-escalate-bg: var(--ic-escalate-bg-dark);
        --ic-disagree: var(--ic-disagree-dark);
        --ic-disagree-bg: var(--ic-disagree-bg-dark);
    }
}

@supports (color: light-dark(white, black)) {
    :root {
        --ic-surface: light-dark(var(--ic-surface-light), var(--ic-surface-dark));
        --ic-inset: light-dark(var(--ic-inset-light), var(--ic-inset-dark));
        --ic-border: light-dark(var(--ic-border-light), var(--ic-border-dark));
        --ic-text: light-dark(var(--ic-text-light), var(--ic-text-dark));
        --ic-muted: light-dark(var(--ic-muted-light), var(--ic-muted-dark));
        --ic-code-bg: light-dark(var(--ic-code-bg-light), var(--ic-code-bg-dark));
        --ic-routed: light-dark(var(--ic-routed-light), var(--ic-routed-dark));
        --ic-routed-bg: light-dark(var(--ic-routed-bg-light), var(--ic-routed-bg-dark));
        --ic-escalate: light-dark(var(--ic-escalate-light), var(--ic-escalate-dark));
        --ic-escalate-bg: light-dark(var(--ic-escalate-bg-light), var(--ic-escalate-bg-dark));
        --ic-disagree: light-dark(var(--ic-disagree-light), var(--ic-disagree-dark));
        --ic-disagree-bg: light-dark(var(--ic-disagree-bg-light), var(--ic-disagree-bg-dark));
    }
}

.dark {
    color-scheme: dark;
    --ic-surface: var(--ic-surface-dark);
    --ic-inset: var(--ic-inset-dark);
    --ic-border: var(--ic-border-dark);
    --ic-text: var(--ic-text-dark);
    --ic-muted: var(--ic-muted-dark);
    --ic-code-bg: var(--ic-code-bg-dark);
    --ic-routed: var(--ic-routed-dark);
    --ic-routed-bg: var(--ic-routed-bg-dark);
    --ic-escalate: var(--ic-escalate-dark);
    --ic-escalate-bg: var(--ic-escalate-bg-dark);
    --ic-disagree: var(--ic-disagree-dark);
    --ic-disagree-bg: var(--ic-disagree-bg-dark);
}

@media (forced-colors: active) {
    :root,
    .dark {
        --ic-surface: Canvas;
        --ic-inset: Canvas;
        --ic-border: CanvasText;
        --ic-text: CanvasText;
        --ic-muted: CanvasText;
        --ic-code-bg: Canvas;
        --ic-routed: Highlight;
        --ic-routed-bg: Canvas;
        --ic-escalate: Highlight;
        --ic-escalate-bg: Canvas;
        --ic-disagree: Highlight;
        --ic-disagree-bg: Canvas;
    }
}

.ic-hero {
    background-color: var(--ic-surface);
    border: 1px solid var(--ic-border);
    border-radius: 12px;
    padding: 14px 18px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px 16px;
    color: var(--ic-text);
}

.ic-hero-title {
    margin: 0;
    font-size: 1.2rem;
    font-weight: 700;
    line-height: 1.3;
    color: var(--ic-text);
}

.ic-hero-sub {
    margin: 4px 0 0;
    font-size: 0.875rem;
    color: var(--ic-muted);
}

.ic-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.ic-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background-color: var(--ic-inset);
    color: var(--ic-text);
    border: 1px solid var(--ic-border);
}

.ic-sidebar-summary {
    background-color: var(--ic-surface);
    border: 1px solid var(--ic-border);
    border-radius: 10px;
    padding: 12px;
    color: var(--ic-text);
    font-size: 0.8125rem;
    line-height: 1.45;
}

.ic-sidebar-summary-title {
    font-size: 0.875rem;
    font-weight: 700;
    margin-bottom: 6px;
    color: var(--ic-text);
}

.ic-sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding: 3px 0;
    color: var(--ic-muted);
    font-size: 0.75rem;
}

.ic-sidebar-stat strong {
    color: var(--ic-text);
    white-space: nowrap;
}

.ic-card {
    background-color: var(--ic-surface);
    border: 1px solid var(--ic-border);
    border-left-width: 5px;
    border-radius: 12px;
    padding: 16px;
    color: var(--ic-text);
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-height: 215px;
}

.ic-card--routed {
    border-left-color: var(--ic-routed);
}

.ic-card--escalate {
    border-left-color: var(--ic-escalate);
}

.ic-card--idle {
    border-left-color: var(--ic-border);
}

.ic-card-top {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}

.ic-model-tag {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ic-muted);
}

.ic-badge-group {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.ic-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    border: 1px solid var(--ic-border);
}

.ic-badge--routed {
    background-color: var(--ic-routed-bg);
    color: var(--ic-routed);
    border-color: var(--ic-routed);
}

.ic-badge--escalate {
    background-color: var(--ic-escalate-bg);
    color: var(--ic-escalate);
    border-color: var(--ic-escalate);
}

.ic-badge--agree {
    background-color: var(--ic-routed-bg);
    color: var(--ic-routed);
    border-color: var(--ic-routed);
}

.ic-badge--disagree {
    background-color: var(--ic-disagree-bg);
    color: var(--ic-disagree);
    border-color: var(--ic-disagree);
}

.ic-intent-title {
    margin: 0;
    font-size: 1.125rem;
    font-weight: 700;
    line-height: 1.3;
    color: var(--ic-text);
}

.ic-kpi-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 12px;
    font-size: 0.75rem;
    color: var(--ic-muted);
}

.ic-kpi-row strong {
    color: var(--ic-text);
}

.ic-code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.75rem;
    background-color: var(--ic-code-bg);
    color: var(--ic-text);
    border: 1px solid var(--ic-border);
    border-radius: 6px;
    padding: 2px 7px;
    overflow-wrap: anywhere;
}

.ic-workflow {
    background-color: var(--ic-inset);
    border: 1px solid var(--ic-border);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 0.8125rem;
    line-height: 1.45;
    color: var(--ic-text);
}

.ic-workflow-note {
    margin-top: 4px;
    font-size: 0.75rem;
    color: var(--ic-muted);
}

.ic-meta {
    background-color: var(--ic-inset);
    border: 1px solid var(--ic-border);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.75rem;
    color: var(--ic-muted);
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 6px 12px;
}

.ic-meta strong {
    color: var(--ic-text);
}

.preset-btn button {
    justify-content: flex-start !important;
    text-align: left !important;
    font-size: 0.8125rem !important;
    padding: 7px 10px !important;
}

/* Desktop: push main content cleanly to the right when 380px sidebar opens */
.sidebar-parent:has(.sidebar.open:not(.right)) {
    padding-left: 380px !important;
}

.sidebar.open:not(.right) {
    border-right: 1px solid var(--ic-border);
    box-shadow: 6px 0 24px rgba(0, 0, 0, 0.22) !important;
}

/* Narrow viewports: overlay and blur the main workspace instead of squeezing cards */
@media (max-width: 900px) {
    .sidebar-parent:has(.sidebar.open:not(.right)) {
        padding-left: 0 !important;
    }

    .sidebar-parent > :not(.sidebar) {
        transition: filter 0.28s ease, opacity 0.28s ease;
    }

    .sidebar-parent:has(.sidebar.open) > :not(.sidebar) {
        filter: blur(4px) brightness(0.78);
        opacity: 0.85;
        pointer-events: none;
    }
}

@media (prefers-reduced-motion: reduce) {
    .sidebar-parent,
    .sidebar-parent > :not(.sidebar),
    .sidebar {
        transition: none !important;
    }
}
"""


def get_service() -> IntentClassifierService:
    """Lazily load and cache the IntentClassifierService singleton."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = IntentClassifierService()
    return _SERVICE


def _pretty_intent(intent_name: str) -> str:
    return intent_name.rstrip("?").replace("_", " ").title()


def _format_idle_card(model_label: str, subtitle: str) -> str:
    return (
        '<div class="ic-card ic-card--idle">'
        '<div class="ic-card-top">'
        f'<span class="ic-model-tag">{html.escape(model_label)}</span>'
        '<span class="ic-pill">Ready</span>'
        "</div>"
        '<h3 class="ic-intent-title">Waiting for Customer Query</h3>'
        f'<div class="ic-workflow">{html.escape(subtitle)}</div>'
        "</div>"
    )


def _format_status_card_html(
    res: PredictionResult,
    model_role: str,
    comparison_primary: PredictionResult | None = None,
) -> str:
    if res.top_intent == "empty_input":
        return _format_idle_card(
            f"{res.model_name} ({model_role})",
            "Enter a customer banking message or click any preset in the left sidebar.",
        )

    card_cls = "ic-card--escalate" if res.escalate else "ic-card--routed"
    status_cls = "ic-badge--escalate" if res.escalate else "ic-badge--routed"
    status_label = "⚠ ESCALATE TO HUMAN AGENT" if res.escalate else "✓ AUTO-ROUTED"
    pretty_title = _pretty_intent(res.top_intent)

    compare_badge = ""
    if comparison_primary is not None and comparison_primary.top_intent != "empty_input":
        if res.top_intent == comparison_primary.top_intent:
            compare_badge = '<span class="ic-badge ic-badge--agree">✓ Agrees with Primary</span>'
        else:
            compare_badge = '<span class="ic-badge ic-badge--disagree">✕ Differs from Primary</span>'

    return (
        f'<div class="ic-card {card_cls}">'
        '<div class="ic-card-top">'
        f'<span class="ic-model-tag">{html.escape(res.model_name)} · {html.escape(model_role)}</span>'
        '<div class="ic-badge-group">'
        f"{compare_badge}"
        f'<span class="ic-badge {status_cls}">{status_label}</span>'
        "</div>"
        "</div>"
        f'<h3 class="ic-intent-title">{html.escape(pretty_title)}</h3>'
        '<div class="ic-kpi-row">'
        f'<code class="ic-code">{html.escape(res.top_intent)}</code>'
        f"<span>Confidence: <strong>{res.confidence:.1%}</strong></span>"
        f"<span>Margin vs #2: <strong>{res.margin:.1%}</strong></span>"
        "</div>"
        '<div class="ic-workflow">'
        f"<div><strong>Suggested Bot Workflow:</strong> {html.escape(res.routing_action)}</div>"
        f'<div class="ic-workflow-note">{html.escape(res.escalation_reason)}</div>'
        "</div>"
        "</div>"
    )


def _format_meta_html(res: PredictionResult | None, threshold: float) -> str:
    if res is None:
        return (
            '<div class="ic-meta">'
            "<span><strong>Model:</strong> TF-IDF + Logistic Regression (Baseline)</span>"
            f"<span><strong>Threshold (tau):</strong> {threshold:.2f}</span>"
            "</div>"
        )
    return (
        '<div class="ic-meta">'
        f"<span><strong>Model:</strong> {html.escape(res.model_name)}</span>"
        f"<span><strong>Latency:</strong> {res.latency_ms:.2f} ms</span>"
        f"<span><strong>Threshold (tau):</strong> {threshold:.2f}</span>"
        "</div>"
    )


def analyze_query(
    text: str,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    compare_baseline: bool = True,
):
    """Predict top-3 banking intents, escalation state, and optional baseline comparison."""
    service = get_service()
    tau = float(threshold)
    show_compare = bool(compare_baseline)

    res = service.predict(
        text=text,
        top_k=3,
        threshold=tau,
        compare_baseline=show_compare,
    )
    prim = res["primary"]
    base = res["baseline"]
    assert prim is not None

    prim_role = "Primary" if service.has_transformer else "Baseline"
    prim_status_html = _format_status_card_html(prim, model_role=prim_role)
    prim_top3 = {intent: round(prob, 4) for intent, prob in prim.top_k}
    prim_meta_html = _format_meta_html(prim, tau)

    if base is not None:
        base_status_html = _format_status_card_html(
            base,
            model_role="Baseline",
            comparison_primary=prim,
        )
        base_top3 = {intent: round(prob, 4) for intent, prob in base.top_k}
        base_meta_html = _format_meta_html(base, tau)
        base_visible = True
    elif not service.has_transformer:
        base_status_html = (
            '<div class="ic-card ic-card--idle">'
            '<div class="ic-card-top">'
            '<span class="ic-model-tag">Transformer Weights Missing</span>'
            '<span class="ic-pill">Fallback Mode</span>'
            "</div>"
            '<h3 class="ic-intent-title">Baseline-Only Preview Active</h3>'
            '<div class="ic-workflow">Place fine-tuned weights inside '
            '<code class="ic-code">models/transformer_intent/</code> to enable live '
            "side-by-side <strong>DistilBERT vs. TF-IDF</strong> comparisons.</div>"
            "</div>"
        )
        base_top3 = {}
        base_meta_html = _format_meta_html(None, tau)
        base_visible = show_compare
    else:
        base_status_html = ""
        base_top3 = {}
        base_meta_html = ""
        base_visible = False

    return (
        prim_status_html,
        prim_top3,
        prim_meta_html,
        gr.update(visible=base_visible),
        base_status_html,
        base_top3,
        base_meta_html,
    )


def _on_control_change(text: str, threshold: float, compare_baseline: bool):
    if text and text.strip():
        return analyze_query(text, threshold, compare_baseline)
    return _idle_outputs(threshold, compare_baseline)


def _idle_outputs(threshold: float, compare_baseline: bool):
    service = get_service()
    tau = float(threshold)
    show_compare = bool(compare_baseline)
    prim_title = "Fine-Tuned DistilBERT (Primary)" if service.has_transformer else "TF-IDF + Logistic Regression"
    prim_idle = _format_idle_card(
        prim_title,
        "Type a banking customer query above or pick a preset from the left sidebar.",
    )
    base_idle = _format_idle_card(
        "TF-IDF + Logistic Regression (Baseline)",
        "Baseline predictions will appear side-by-side here for comparison.",
    )
    prim_meta = _format_meta_html(None, tau).replace(
        "TF-IDF + Logistic Regression (Baseline)", prim_title
    )
    base_meta = _format_meta_html(None, tau)
    return (
        prim_idle,
        {},
        prim_meta,
        gr.update(visible=show_compare),
        base_idle,
        {},
        base_meta,
    )


def _reset_ui(threshold: float, compare_baseline: bool):
    return ("", *_idle_outputs(threshold, compare_baseline))


def _make_preset_handler(full_query: str):
    def _handler(threshold: float, compare_baseline: bool):
        return (full_query, *analyze_query(full_query, threshold, compare_baseline))

    return _handler


def build_demo() -> gr.Blocks:
    """Construct the Layout B+ Gradio interactive demo interface."""
    service = get_service()
    active_badge = (
        "DistilBERT + TF-IDF Active" if service.has_transformer else "TF-IDF Baseline Only"
    )
    primary_label_title = (
        "Fine-Tuned DistilBERT — Top-3 Intents"
        if service.has_transformer
        else "TF-IDF + Logistic Regression — Top-3 Intents"
    )

    with gr.Blocks(title="Banking Chatbot Intent Classifier (BANKING77)") as demo:
        with gr.Sidebar(open=True, width=380, position="left"):
            gr.HTML(
                value=(
                    '<div class="ic-sidebar-summary">'
                    '<div class="ic-sidebar-summary-title">Controls &amp; Demo Presets</div>'
                    '<div class="ic-sidebar-stat"><span>Dataset</span><strong>BANKING77 · 77 Intents</strong></div>'
                    '<div class="ic-sidebar-stat"><span>DistilBERT</span><strong>92.43% · ~19 ms</strong></div>'
                    '<div class="ic-sidebar-stat"><span>TF-IDF LR</span><strong>89.47% · ~0.8 ms</strong></div>'
                    "</div>"
                ),
                container=False,
                padding=False,
            )

            threshold_slider = gr.Slider(
                minimum=0.30,
                maximum=0.95,
                value=DEFAULT_CONFIDENCE_THRESHOLD,
                step=0.05,
                label="Threshold (tau)",
                info="Queries below tau escalate to a human agent.",
            )
            compare_toggle = gr.Checkbox(
                value=True,
                label="Show TF-IDF Baseline Panel",
            )

            gr.Markdown("### Guided Demo Presets")
            preset_buttons: list[tuple[gr.Button, str]] = []
            for group_title, is_open, items in SIDEBAR_PRESETS:
                with gr.Accordion(group_title, open=is_open):
                    for short_label, full_query in items:
                        btn = gr.Button(
                            short_label,
                            variant="secondary",
                            size="sm",
                            elem_classes=["preset-btn"],
                        )
                        preset_buttons.append((btn, full_query))

        gr.HTML(
            value=(
                '<div class="ic-hero">'
                "<div>"
                '<h1 class="ic-hero-title">Banking Chatbot Intent Classifier &amp; Escalation Demo</h1>'
                '<p class="ic-hero-sub">Topic #7 (BANKING77) — Contextual DistilBERT vs. TF-IDF Baseline with confidence-gated human escalation.</p>'
                "</div>"
                '<div class="ic-pill-row">'
                '<span class="ic-pill">77 Banking Intents</span>'
                f'<span class="ic-pill">{html.escape(active_badge)}</span>'
                "</div>"
                "</div>"
            ),
            container=False,
            padding=False,
        )

        with gr.Row():
            with gr.Column(scale=5):
                query_input = gr.Textbox(
                    label="Customer Support Message",
                    placeholder="Type a banking query (e.g. 'Topping up my card is not working because it is pending') or pick a preset on the left...",
                    lines=2,
                    max_lines=4,
                )
            with gr.Column(scale=1, min_width=140):
                submit_btn = gr.Button("Classify Intent", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

        init_vals = _idle_outputs(DEFAULT_CONFIDENCE_THRESHOLD, True)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                primary_status_html = gr.HTML(
                    value=init_vals[0],
                    container=False,
                    padding=False,
                )
                primary_label = gr.Label(
                    num_top_classes=3,
                    label=primary_label_title,
                )
                primary_meta_html = gr.HTML(
                    value=init_vals[2],
                    container=False,
                    padding=False,
                )

            with gr.Column(scale=1, visible=True) as baseline_col:
                baseline_status_html = gr.HTML(
                    value=init_vals[4],
                    container=False,
                    padding=False,
                )
                baseline_label = gr.Label(
                    num_top_classes=3,
                    label="TF-IDF + Logistic Regression — Top-3 Intents",
                )
                baseline_meta_html = gr.HTML(
                    value=init_vals[6],
                    container=False,
                    padding=False,
                )

        outputs = [
            primary_status_html,
            primary_label,
            primary_meta_html,
            baseline_col,
            baseline_status_html,
            baseline_label,
            baseline_meta_html,
        ]

        submit_btn.click(
            fn=analyze_query,
            inputs=[query_input, threshold_slider, compare_toggle],
            outputs=outputs,
        )
        query_input.submit(
            fn=analyze_query,
            inputs=[query_input, threshold_slider, compare_toggle],
            outputs=outputs,
        )
        threshold_slider.change(
            fn=_on_control_change,
            inputs=[query_input, threshold_slider, compare_toggle],
            outputs=outputs,
        )
        compare_toggle.change(
            fn=_on_control_change,
            inputs=[query_input, threshold_slider, compare_toggle],
            outputs=outputs,
        )
        clear_btn.click(
            fn=_reset_ui,
            inputs=[threshold_slider, compare_toggle],
            outputs=[query_input, *outputs],
        )

        for btn, full_query in preset_buttons:
            btn.click(
                fn=_make_preset_handler(full_query),
                inputs=[threshold_slider, compare_toggle],
                outputs=[query_input, *outputs],
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="cyan",
            neutral_hue="zinc",
        ).set(
            body_background_fill_dark="#000000",
            background_fill_primary_dark="#000000",
            background_fill_secondary_dark="#050507",
            block_background_fill_dark="#0b0b0e",
            panel_background_fill_dark="#0b0b0e",
            border_color_primary_dark="#3f3f46",
            block_border_color_dark="#3f3f46",
            panel_border_color_dark="#3f3f46",
            code_background_fill_dark="#000000",
        ),
        css=CUSTOM_CSS,
        head='<meta name="color-scheme" content="light dark">',
    )
