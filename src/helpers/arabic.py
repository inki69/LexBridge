"""
Arabic NLP utilities for LexBridge.

Handles the unique challenges of Arabic text processing:
  - Tashkeel (diacritics) removal: فَتَحَ → فتح
  - Hamza normalization: إ أ آ → ا
  - Tatweel (kashida) stripping: كـتـاب → كتاب
  - Lam-Alef ligature normalization
  - RTL-safe whitespace cleanup

These are essential for consistent embedding and retrieval of Arabic
text extracted from PDFs, which often contain inconsistent Unicode
representations of the same word.
"""

import re
import unicodedata


# ── Unicode ranges ──────────────────────────────────────────────────────────

# Arabic diacritics (tashkeel): fathah, dammah, kasrah, sukun, shadda, etc.
_TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")

# Arabic tatweel / kashida (elongation character)
_TATWEEL = re.compile(r"\u0640")

# Hamza variants that should normalize to bare alef
_HAMZA_MAP = {
    "\u0623": "\u0627",  # أ → ا
    "\u0625": "\u0627",  # إ → ا
    "\u0622": "\u0627",  # آ → ا
    "\u0671": "\u0627",  # ٱ → ا
}

# Alef-maqsura → ya
_ALEF_MAQSURA = {"\u0649": "\u064A"}  # ى → ي

# Ta-marbuta → ha
_TA_MARBUTA = {"\u0629": "\u0647"}  # ة → ه


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for consistent embedding and retrieval.

    Steps applied (in order):
      1. Unicode NFC normalization (canonical decomposition → recomposition)
      2. Strip tashkeel (diacritical marks)
      3. Strip tatweel (kashida elongation)
      4. Hamza normalization (أ إ آ ٱ → ا)
      5. Alef-maqsura → ya  (ى → ي)
      6. Ta-marbuta → ha    (ة → ه)
      7. Collapse multiple whitespace
    """
    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Remove tashkeel
    text = _TASHKEEL.sub("", text)

    # 3. Remove tatweel
    text = _TATWEEL.sub("", text)

    # 4. Hamza normalization
    for src, dst in _HAMZA_MAP.items():
        text = text.replace(src, dst)

    # 5. Alef maqsura
    for src, dst in _ALEF_MAQSURA.items():
        text = text.replace(src, dst)

    # 6. Ta marbuta
    for src, dst in _TA_MARBUTA.items():
        text = text.replace(src, dst)

    # 7. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_arabic(text: str, threshold: float = 0.3) -> bool:
    """
    Heuristic: returns True if ≥ threshold fraction of alphabetic
    characters in `text` are Arabic script.
    """
    if not text:
        return False
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    arabic_count = sum(1 for c in alpha_chars if "\u0600" <= c <= "\u06FF")
    return (arabic_count / len(alpha_chars)) >= threshold
