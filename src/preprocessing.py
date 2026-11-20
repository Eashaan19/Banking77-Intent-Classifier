"""Dual-track text preprocessing pipelines for intent classification.

Track A provides domain-aware cleaning for classical n-gram / TF-IDF models
(preserving negation cues and contractions while normalizing currency and numbers),
plus an aggressive NLTK stopword+lemmatization ablation variant.
Track B provides minimal Unicode and whitespace normalization for Transformer
subword tokenizers.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Minimal fallback stopword list if NLTK corpora are unavailable offline.
_FALLBACK_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "i", "you", "my", "me", "we",
        "our", "your", "they", "their", "this", "these", "those", "am",
    }
)

_CURRENCY_RE = re.compile(
    r"(?:[$£€]\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*(?:usd|gbp|eur|aud|cad|dollars?|pounds?|euros?)\b)",
    flags=re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_DOMAIN_KEEP_RE = re.compile(r"[^a-z0-9_'\-\s]+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]+")
_CONTROL_CHARS_RE = re.compile(r"[\u200b-\u200d\ufeff\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")

_CACHED_STOPWORDS: frozenset[str] | None = None
_LEMMATIZER: WordNetLemmatizer | None = None


def _get_stopwords() -> frozenset[str]:
    global _CACHED_STOPWORDS
    if _CACHED_STOPWORDS is None:
        try:
            _CACHED_STOPWORDS = frozenset(stopwords.words("english"))
        except LookupError:
            _CACHED_STOPWORDS = _FALLBACK_STOPWORDS
    return _CACHED_STOPWORDS


def _get_lemmatizer() -> WordNetLemmatizer | None:
    global _LEMMATIZER
    if _LEMMATIZER is None:
        try:
            lem = WordNetLemmatizer()
            lem.lemmatize("cards")
            _LEMMATIZER = lem
        except LookupError:
            _LEMMATIZER = None
    return _LEMMATIZER


def normalize_unicode_whitespace(text: str) -> str:
    """Apply Unicode NFKC normalization, strip control chars, and collapse spaces."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _CONTROL_CHARS_RE.sub("", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def clean_transformer_minimal(text: str) -> str:
    """Track B: Minimal cleaning preserving casing, punctuation, and contractions."""
    return normalize_unicode_whitespace(text)


def clean_classical_domain_aware(text: str) -> str:
    """Track A: Domain-aware classical cleaning for TF-IDF + linear models.

    Normalizes currency amounts and numbers to semantic tokens while preserving
    contractions and negation words (not, no, never, can't, won't, didn't).
    """
    cleaned = normalize_unicode_whitespace(text).lower()
    cleaned = _CURRENCY_RE.sub(" _amount_ ", cleaned)
    cleaned = _NUMBER_RE.sub(" _num_ ", cleaned)
    # Normalize curly apostrophes after NFKC just in case
    cleaned = cleaned.replace("’", "'").replace("‘", "'")
    cleaned = _DOMAIN_KEEP_RE.sub(" ", cleaned)
    # Strip leading/trailing apostrophes/hyphens on isolated tokens while keeping internal ones
    tokens = [tok.strip("'-") for tok in cleaned.split()]
    tokens = [tok for tok in tokens if tok]
    return " ".join(tokens)


def clean_classical_aggressive(text: str) -> str:
    """Ablation track: Standard stopword removal + WordNet lemmatization."""
    cleaned = normalize_unicode_whitespace(text).lower()
    cleaned = _NON_ALPHA_RE.sub(" ", cleaned)
    sw = _get_stopwords()
    lem = _get_lemmatizer()
    tokens: list[str] = []
    for tok in cleaned.split():
        if tok in sw:
            continue
        tokens.append(lem.lemmatize(tok) if lem is not None else tok)
    return " ".join(tokens)


PREPROCESSORS = {
    "raw_minimal": clean_transformer_minimal,
    "transformer_minimal": clean_transformer_minimal,
    "classical_domain_aware": clean_classical_domain_aware,
    "classical_aggressive": clean_classical_aggressive,
}


def preprocess_batch(texts: Iterable[str], mode: str = "classical_domain_aware") -> list[str]:
    """Apply a named preprocessing pipeline to an iterable of utterances."""
    if mode not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocessing mode '{mode}'. Expected one of {list(PREPROCESSORS)}")
    fn = PREPROCESSORS[mode]
    return [fn(t) for t in texts]
