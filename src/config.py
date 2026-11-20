"""Project paths, hyperparameters, dataset settings, and chatbot routing actions."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "report"

SEED = 42
VAL_SIZE = 0.15

PRIMARY_DATASET_ID = "mteb/banking77"
PRIMARY_DATASET_REVISION: str | None = None

TRANSFORMER_BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 96

DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_MARGIN_THRESHOLD = 0.10

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_tfidf_lr.joblib"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "transformer_intent"

_EXACT_ROUTING_ACTIONS: dict[str, str] = {
    "activate_my_card": "Open Card Activation Wizard",
    "age_limit": "Show Account Eligibility & Age Policy",
    "apple_pay_or_google_pay": "Open Mobile Wallet Setup Guide",
    "atm_support": "Show ATM Locator & Partner ATM Guide",
    "automatic_top_up": "Open Auto Top-Up Rules Settings",
    "balance_not_updated_after_bank_transfer": "Check Pending Inbound Transfer Status",
    "balance_not_updated_after_cheque_or_cash_deposit": "Check Cheque/Cash Deposit Clearing Timeline",
    "beneficiary_not_allowed": "Show Supported Beneficiaries & Transfer Rules",
    "cancel_transfer": "Open Outbound Transfer Cancellation Flow",
    "card_about_to_expire": "Start Expiring Card Renewal Flow",
    "card_acceptance": "Show Merchant Card Acceptance FAQ",
    "card_arrival": "Track Physical Card Delivery Status",
    "card_delivery_estimate": "Display Card Shipping & Delivery ETA",
    "card_linking": "Open Link Existing Card Flow",
    "card_not_working": "Run Card Diagnostics & Freeze/Replace Check",
    "card_payment_fee_charged": "Break Down Card Payment Fees",
    "card_payment_not_recognised": "Start Unrecognized Transaction Dispute Flow",
    "card_payment_wrong_exchange_rate": "Show FX Rate Breakdown & Dispute Option",
    "card_swallowed": "Freeze Swallowed Card & Order Replacement",
    "cash_withdrawal_charge": "Explain ATM Cash Withdrawal Fee Schedule",
    "cash_withdrawal_not_recognised": "Open Unauthorized ATM Withdrawal Dispute",
    "change_pin": "Open In-App PIN Change Instructions",
    "compromised_card": "Immediately Freeze Compromised Card & Escalate",
    "contactless_not_working": "Run Contactless / NFC Troubleshooting Steps",
    "country_support": "Show Supported Countries & Residency List",
    "declined_card_payment": "Inspect Declined Card Payment Reasons",
    "declined_cash_withdrawal": "Inspect Declined ATM Withdrawal Limits",
    "declined_transfer": "Inspect Declined Bank Transfer Details",
    "direct_debit_payment_not_recognised": "Open Direct Debit Dispute & Mandate Manager",
    "disposable_card_limits": "Show Disposable Virtual Card Spend Limits",
    "edit_personal_details": "Open Profile & KYC Details Update Flow",
    "exchange_charge": "Show Currency Exchange Fee Schedule",
    "exchange_rate": "Show Live Interbank FX Rate Calculator",
    "exchange_via_app": "Open In-App Currency Conversion Guide",
    "extra_charge_on_statement": "Review Statement Fee & Surcharge Breakdown",
    "failed_transfer": "Troubleshoot Failed Transfer & Retry",
    "fiat_currency_support": "Show Supported Fiat Currencies List",
    "get_disposable_virtual_card": "Issue Single-Use Disposable Virtual Card",
    "get_physical_card": "Start Order New Physical Card Flow",
    "getting_spare_card": "Explain Multi-Card / Spare Card Eligibility",
    "getting_virtual_card": "Issue Standard Virtual Card",
    "lost_or_stolen_card": "Immediately Freeze Lost/Stolen Card & Reissue",
    "lost_or_stolen_phone": "Lock Account Sessions & Deactivate Mobile Wallet",
    "order_physical_card": "Start Order Physical Card Checkout",
    "passcode_forgotten": "Start Passcode Reset Verification Flow",
    "pending_card_payment": "Explain Merchant Authorization Hold Timeline",
    "pending_cash_withdrawal": "Explain Pending ATM Hold Settlement",
    "pending_top_up": "Check Pending Top-Up Processing Status",
    "pending_transfer": "Track Pending Outbound/Inbound Transfer",
    "pin_blocked": "Start PIN Unblock & PUK Recovery Flow",
    "receiving_money": "Show IBAN / Account Details for Receiving Funds",
    "Refund_not_showing_up": "Track Merchant Refund Settlement Window",
    "request_refund": "Open Merchant Refund / Chargeback Request",
    "reverted_card_payment?": "Explain Reverted Card Authorization",
    "supported_cards_and_currencies": "Show Supported Funding Cards & Currencies",
    "terminate_account": "Open Account Closure Checklist & Retention",
    "top_up_by_bank_transfer_charge": "Show Bank Transfer Top-Up Fee Policy",
    "top_up_by_card_charge": "Show Card Top-Up Fee Policy",
    "top_up_by_cash_or_cheque": "Show Cash/Cheque Deposit Locations",
    "top_up_failed": "Troubleshoot Failed Top-Up & Card 3DS",
    "top_up_limits": "Display Account Top-Up Limits",
    "top_up_reverted": "Explain Reverted Top-Up Authorization",
    "topping_up_by_card": "Open Top Up by Debit/Credit Card Guide",
    "transaction_charged_twice": "Open Duplicate Charge Dispute Form",
    "transfer_fee_charged": "Show Cross-Border Transfer Fee Breakdown",
    "transfer_into_account": "Show Inbound Bank Transfer Instructions",
    "transfer_not_received_by_recipient": "Trace Missing Outbound Transfer",
    "transfer_timing": "Show Domestic & International Transfer ETAs",
    "unable_to_verify_identity": "Escalate KYC Identity Verification Failure",
    "verify_my_identity": "Open Document & Selfie KYC Verification",
    "verify_source_of_funds": "Open Source-of-Funds Document Upload",
    "verify_top_up": "Open Top-Up Card Ownership Verification",
    "virtual_card_not_working": "Troubleshoot Virtual Card CVV & Freeze Status",
    "visa_or_mastercard": "Explain Visa vs. Mastercard Card Issuance",
    "why_verify_identity": "Show Regulatory KYC Verification FAQ",
    "wrong_amount_of_cash_received": "Open ATM Cash Discrepancy Claim",
    "wrong_exchange_rate_for_cash_withdrawal": "Show ATM FX Markup & DCC Explanation",
}


def get_routing_action(intent_name: str) -> str:
    """Return the downstream chatbot workflow action for a predicted banking intent."""
    if intent_name in _EXACT_ROUTING_ACTIONS:
        return _EXACT_ROUTING_ACTIONS[intent_name]
    pretty = intent_name.replace("_", " ").rstrip("?").title()
    return f"Route to Banking Workflow: {pretty}"
