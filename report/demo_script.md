# Live Application Demo Script (~2–3 Minutes, No Slides)

**Launch Command:**
```powershell
cd "c:\Users\easha\OneDrive\Documents\SP JAIN\SEM 5\NLP\INDV\intent_classifier"
C:\Users\easha\venvs\nlp_intent\Scripts\python.exe app.py
```
Open http://127.0.0.1:7860 in your browser before starting. Keep the **left sidebar open (`380px`)**, **Show TF-IDF Baseline Panel** checked, and **Threshold (`tau`)** at its default **`0.85`**. *(You can click the sidebar collapse arrow `<` anytime to give the comparison cards 100% width.)*

---

## Part 1: What It Does (~45 seconds)
1. **Framing:** *"This is a banking customer-support intent classifier trained on `BANKING77` — 77 fine-grained banking intents. In production, a chatbot needs two things: accurate routing at low CPU latency, and a confidence gate so uncertain or off-topic queries escalate to a human agent rather than taking a wrong financial action."*
2. **In the left sidebar -> `Clear In-Scope` -> click `Activate physical card`** *(loads `"How do I activate the new physical card that just arrived?"` and classifies in one click)*:
   - Point out the **DistilBERT Primary card**: **`✓ AUTO-ROUTED`**, **`Activate My Card` (`99.7%`, ~20–30 ms CPU)**, the **Suggested Bot Workflow** (`Open Card Activation Wizard`), and the **Top-3 probability bars**.
   - Contrast with the **TF-IDF Baseline card** on the right (`62.5%` confidence, ~2 ms — lightweight at `5.7 MB`, yet below our `0.85` threshold).

## Part 2: Why It Works (~45 seconds)
1. **In the left sidebar -> `DistilBERT Wins vs TF-IDF` -> click `Pending top-up (Negation)`** *(loads `"Topping up my card is not working because I believe the top up is pending"`)*:
   - Point out the **`✕ Differs from Primary`** badge on the right card:
     - **TF-IDF** gets tripped up by surface words (`"not working"`) and misclassifies as `top_up_failed` (`64.7%`).
     - **DistilBERT** reads clause context across the whole sentence and predicts **`pending_top_up`** at **`99.9%`**.
   - *(Optional second click: **`ATM card failure (Context)`** -> DistilBERT gets `declined_cash_withdrawal` at `99.9%` vs. TF-IDF `card_not_working` at `28.4%`.)*
   - Briefly mention our **preprocessing ablation**: stripping standard NLTK stopwords (`not`, `isn't`, `no`) hurt TF-IDF by `~2.3` Macro F1 points because negation flips intent.
2. **Expand `Fine-Grained Overlap` -> click `Pending card charge`:**
   - Shows DistilBERT cleanly separating `pending_card_payment` (`99.7%`) while TF-IDF spreads probability thinly (`35.1%`).

## Part 3: Limitations & Real-World Safeguards (~45 seconds)
1. **Expand `OOS & Limitations` -> click `Book Tokyo flight (OOS)`** and **`Vague 'card issue'`**:
   - Both trigger the amber **`⚠ ESCALATE TO HUMAN AGENT`** state cleanly (`66.9%` and `31.9%`, both `< tau = 0.85`).
2. **Show honest failure modes + live threshold control:**
   - Click **`Stolen + Close (Multi-intent)`** (`"My card was stolen and I also want to close my account permanently."`): DistilBERT routes `lost_or_stolen_card` (`93.0%`), dropping the secondary `terminate_account` request (**single-label limitation**). Then **drag the `Threshold (tau)` slider live to `0.95`** — the card flips instantaneously to **`⚠ ESCALATE TO HUMAN AGENT`**.
   - Click **`Netflix password (Overconf.)`** (`"Reset my Netflix password because I forgot it."`): DistilBERT still predicts `passcode_forgotten` (`99.7%`) because `BANKING77` has no explicit `out_of_scope` training class (**neural softmax overconfidence on keyword overlap**), pointing directly to next steps like temperature scaling and explicit OOS training data.
