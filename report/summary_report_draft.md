# Intent Classification for Banking Chatbots (`BANKING77`)

**Student:** Eashaan Marwah (`BS24DMU001`) | **Course:** NLP, Language Models & Analytics — Individual Project #7  
**Repository:** https://github.com/Eashaan19/Banking77-Intent-Classifier | **Deadline:** November 25, 2026

---

## 1. Problem Definition & Real-World Constraints

Customer-support chatbots in retail banking must map free-form user messages to exact downstream workflows (e.g., freezing a lost card, disputing a duplicate charge, tracking a pending transfer). Unlike coarse sentiment tasks, banking intent routing presents three operational constraints:

1. **Fine-Grained Semantic Overlap:** Similar phrasing often implies distinct actions (e.g., `pending_card_payment` vs. `declined_card_payment` vs. `reverted_card_payment?`, or `why_verify_identity` vs. `verify_my_identity`).
2. **Interactive Latency Budget:** Inference must remain responsive on standard CPU hardware (< 50 ms per query) without requiring expensive dedicated GPU servers at serving time.
3. **Safe Handling of Ambiguous & Out-of-Scope (OOS) Queries:** High-stakes financial bots cannot blindly guess when a query is underspecified (*"card issue"*) or unrelated (*"book a flight to Tokyo"*). Our system introduces a **confidence-gated escalation layer** (`max_prob < tau`) that routes uncertain queries to a human banking specialist.

---

## 2. Dataset & Dual-Track Preprocessing

We evaluate on **`mteb/banking77`** (**13,069 utterances across 77 banking intents**). Preserving the official test set strictly untouched (`3,076` queries), we carve a **15% stratified validation split** (`SEED = 42`) from training (`8,494` train / `1,499` val). Training utterances average `11.93` words (`99th percentile = 42` words, `max = 79`), so a `96`-subword token budget avoids truncation with minimal padding.

Because classical bag-of-words and contextual Transformers tokenize differently, we implement **two distinct preprocessing tracks** and run a quantitative validation ablation (`TF-IDF 1–2 n-grams + Logistic Regression, C=5.0`):

| Preprocessing Track | Vocabulary Size | Val Accuracy | Val Macro F1 |
| :--- | :---: | :---: | :---: |
| **Raw Minimal** (Unicode NFKC + whitespace) | 9,167 | 0.8793 | 0.8753 |
| **Track A — Domain-Aware Classical** (lowercase, `_amount_`/`_num_` tokens, **preserves negation/contractions**) | 9,153 | 0.8786 | 0.8734 |
| **Ablation — Aggressive NLTK** (stopword removal + WordNet lemmatization) | 5,727 | **0.8559** | **0.8507** |

**Key Insight:** Removing standard NLTK English stopwords strips negation and intent-bearing function words (`not`, `isn't`, `no`, `never`, `why`), shrinking the feature space by 37% and causing a **-2.27 to -2.46 point Macro F1 drop**. Track B (Transformer) applies only NFKC Unicode and control-character normalization so DistilBERT's WordPiece tokenizer retains full syntactic structure.

---

## 3. Model Progression & Hyperparameter Findings

1. **Classical Baseline — `TF-IDF (1–2 n-grams)` + `Logistic Regression`:** Validation tuning over `C ∈ {0.5, 1.0, 2.0, 5.0, 10.0}` selected **`C = 10.0`** (`Val Macro F1 = 0.8764`). A reference `LinearSVC(C=1.0)` scored `0.8909` Test Macro F1, confirming Logistic Regression matches linear-margin performance while natively producing calibrated softmax probabilities for Top-3 ranking and threshold gating.
2. **Primary Model — Fine-Tuned `distilbert-base-uncased`:** Fine-tuned on a Tesla T4 GPU (`max_length = 96`, `SEED = 42`, selecting best checkpoint by validation Macro F1). An initial 5-epoch configuration (`batch_size = 32, lr = 3e-5`, 1,330 steps) underfit across 77 classes (`85.31%` Test Accuracy, `eval_loss = 1.7213`). Increasing training budget and step rate (`batch_size = 16, lr = 5e-5, 10 epochs`, `EarlyStoppingCallback(patience=3)`) achieved clean convergence (`eval_loss = 0.7216`) in `331s`.

---

## 4. Quantitative Evaluation (Untouched Test Set, n = 3,076)

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | Mean / p95 Latency (CPU) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression (Baseline)** | 0.8947 | 0.8995 | 0.8946 | 0.8948 | 0.8949 | **0.79 / 1.24 ms** | **5.73** |
| **Fine-Tuned `distilbert-base-uncased`** | **0.9243** | **0.9272** | **0.9242** | **0.9243** | **0.9244** | 18.65 / 22.01 ms | 256.34 |

Fine-tuned DistilBERT outperforms the strong n-gram baseline by **+2.96 percentage points** in Accuracy and **+2.95 points** in Macro F1 — a **28.1% relative reduction in classification error** — while running at **`18.65 ms` mean (`22.01 ms` p95) on local CPU** (Intel Core Ultra 7 258V), comfortably inside a 50 ms interactive budget.

*Supporting visuals saved in `outputs/`: `model_comparison_bar.png`, `top_confused_pairs_bar.png`, `top_confused_cluster_cm.png`, and `confidence_coverage_curve.png`.*

### Confidence-Gated Human Escalation Tradeoff (Validation + 20-Query Probe Suite)

- **Default Interactive Threshold (`tau = 0.85`):** Automates **`93.66%`** of in-scope queries at **`94.94%`** accuracy (`66.67%` non-banking OOS rejection).
- **High-Precision Threshold (`tau = 0.95`):** Automates **`90.79%`** of queries at **`96.11%`** accuracy (halving accepted errors to `3.54%`) and rejects **`83.33%` (10/12)** of non-banking OOS queries plus **4/4** vague queries (*"card issue"*, *"help please"*).

---

## 5. Error Analysis & System Limitations

1. **Where DistilBERT Beats TF-IDF (Contextual Disambiguation):**
   - *"Topping up my card is not working because I believe the top up is pending"* -> TF-IDF predicts `top_up_failed` (`0.647`); DistilBERT resolves clause structure to **`pending_top_up`** (`0.999`).
   - *"This is very frustrating. I can't get my card to work on any ATM I try..."* -> TF-IDF falls back to generic `card_not_working` (`0.284`); DistilBERT captures ATM context for **`declined_cash_withdrawal`** (`0.999`).
   - *"What is the best way to exchange currency?"* -> TF-IDF predicts `exchange_charge` (`0.422`); DistilBERT predicts **`exchange_via_app`** (`0.999`).
2. **Remaining Confusions & Dataset Ambiguity:** Even DistilBERT struggles on near-identical intent pairs (`why_verify_identity -> verify_my_identity`: 7 errors; `card_arrival <-> card_delivery_estimate`: 8 combined; `pin_blocked -> change_pin`: 4) and genuine label noise (e.g., *"When I try and to buy something using my card it keeps getting declined"* is labeled `declined_transfer` in the benchmark while DistilBERT reasonably predicts `declined_card_payment`).
3. **Observed Limitations & Future Improvements:**
   - **Softmax Overconfidence on OOS Lexical Overlap:** Because `BANKING77` contains no explicit `out_of_scope` class, non-banking queries sharing banking keywords still trigger high confidence (e.g., *"Reset my Netflix password because I forgot it"* -> `passcode_forgotten` at `0.997`). *Mitigation:* Temperature scaling, energy-based OOD scoring, or adding an explicit `oos` class.
   - **Single-Label Assumption:** Multi-intent inputs (*"My card was stolen and I also want to close my account permanently"*) predict only one dominant intent (`lost_or_stolen_card`, `0.930`) and drop the second action. *Mitigation:* Multi-label sigmoid heads or clause splitting.
   - **Lexical Dataset Bias:** UK-centric dataset phrasing (where `"debit"` mostly appears in `direct_debit_*`) can mislead queries mentioning `"debit card"`.
