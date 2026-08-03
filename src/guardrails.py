"""
Lightweight guardrails for input and output safety.

These are simple, rule-based checks (not a full ML-based guardrails
framework) -- enough to catch the most common issues: prompt-injection
attempts, empty/junk input, and PII (emails, phone numbers, card-like
numbers) leaking into an answer.
"""

import re

# Patterns that suggest someone is trying to override the system prompt
# / make the assistant ignore its grounding instructions
_INJECTION_PATTERNS = [
    r"ignore (all|the|any) (previous|prior|above) instructions",
    r"disregard (all|the|any) (previous|prior|above) instructions",
    r"you are now",
    r"act as (a|an) ",
    r"system prompt",
    r"reveal your (instructions|prompt|system prompt)",
    r"jailbreak",
]

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3}[-.\s]?\d{3,4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def check_input(query):
    """
    Returns (is_safe, reason). Blocks empty input, excessively long input,
    and common prompt-injection phrasing.
    """
    if not query or not query.strip():
        return False, "Question is empty."

    if len(query) > 1000:
        return False, "Question is too long (max 1000 characters)."

    lowered = query.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return False, (
                "This looks like an attempt to override the assistant's "
                "instructions, which isn't allowed."
            )

    return True, ""


def redact_pii(text):
    """Masks emails, phone numbers, and card-like numbers in a string."""
    text = _EMAIL_RE.sub("[REDACTED EMAIL]", text)
    text = _CARD_RE.sub("[REDACTED NUMBER]", text)
    text = _PHONE_RE.sub("[REDACTED PHONE]", text)
    return text


def check_groundedness(answer, context, min_overlap=0.15):
    """
    Lightweight hallucination heuristic: checks what fraction of the
    answer's significant words also appear in the retrieved context. Not a
    substitute for a real groundedness model, but catches answers that
    clearly drifted away from the source material.
    """
    if not answer or not context:
        return True  # nothing to compare against, don't false-flag

    def _words(t):
        return set(w for w in re.findall(r"\w+", t.lower()) if len(w) > 3)

    answer_words = _words(answer)
    context_words = _words(context)

    if not answer_words:
        return True

    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap >= min_overlap