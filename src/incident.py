"""Incident-type tagging, used in two places:
1. ingest.py tags every chunk with the incident types it talks about.
2. rag.py classifies the user's question first (the "scenario-aware" pipeline).

Keyword rules are deliberately simple and transparent so the team can explain
them in the video. Overlap (e.g. ransomware arriving by email) is handled by
allowing more than one label.
"""
import re

KEYWORDS = {
    "phishing": [
        "phishing", "phish", "suspicious link", "suspicious email", "clicked",
        "attachment", "malicious link", "fake email", "smishing", "text message",
        "opened a file", "spoofed",
    ],
    "ransomware": [
        "ransomware", "ransom", "encrypted", "encrypt", "locked files",
        "files locked", "decrypt", "extortion", "pay the attackers", "backup",
        "backups", "can't open my files", "cannot open files", "bitcoin",
        "cryptocurrency", "ransom note",
    ],
    "bec": [
        "business email compromise", "bec", "bank details", "bank account details",
        "change of bank", "invoice", "payment redirection", "supplier", "fake invoice",
        "pay a new account", "ceo fraud", "urgent payment", "wire",
    ],
    "account_compromise": [
        "mfa", "multi-factor", "two-factor", "2fa", "authenticator", "login prompt",
        "sign-in", "sign in", "password", "account compromise", "hacked account",
        "unusual login", "login alert", "locked out", "push notification",
        "compromised account", "passphrase", "logged into", "logged in",
        "unauthorised access", "someone has access",
    ],
}

_patterns = {
    t: [re.compile(r"\b" + re.escape(k) + r"\b", re.I) for k in kws]
    for t, kws in KEYWORDS.items()
}


def score_types(text: str) -> dict:
    """Number of keyword hits per incident type."""
    return {t: sum(1 for p in pats if p.search(text)) for t, pats in _patterns.items()}


def tag_chunk(text: str, min_hits: int = 1) -> list:
    """All incident types a chunk mentions; ['general'] if none."""
    scores = score_types(text)
    tags = [t for t, s in scores.items() if s >= min_hits]
    return tags or ["general"]


def classify_query(question: str) -> list:
    """Return the top incident type(s) for a question.

    Returns the best type plus any type tied with it (overlap case).
    Returns [] if no keyword matched; the pipeline then falls back to
    normal retrieval instead of guessing.
    """
    scores = score_types(question)
    best = max(scores.values())
    if best == 0:
        return []
    return [t for t, s in scores.items() if s == best]
