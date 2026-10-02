"""Deterministic guardrails. These run outside the LLM on purpose: they must not be
talk-able out of their job by the content they inspect."""
from __future__ import annotations

import re

from .state import CriterionScore, Segment

# --- PII + protected attributes -------------------------------------------------
_PII_PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "PHONE": re.compile(r"\+?\d[\d\s().-]{7,}\d"),
    "AGE": re.compile(r"\b\d{2}\s*(?:years?\s*old|yrs?\s*old|y/o)\b", re.I),
    # Only the lead-in is case-insensitive; the name itself must be capitalised, so
    # "I am passionate about" and "I am Arjun from Kochi" don't swallow ordinary words.
    "NAME_INTRO": re.compile(r"\b((?i:I'm|I am|my name is))\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"),
}
_PROTECTED_TERMS = re.compile(
    r"\b(married|pregnan\w*|husband|wife|church|mosque|temple|hindu|muslim|christian|"
    r"disabilit\w*|wheelchair)\b", re.I)


def redact(segments: list[Segment]) -> tuple[list[Segment], list[str]]:
    """Return redacted copies and the list of redaction types applied.
    Scoring agents only ever see the redacted transcript."""
    out, hits = [], []
    for s in segments:
        text = s.text
        for label, pat in _PII_PATTERNS.items():
            if pat.search(text):
                hits.append(label)
                repl = (lambda m: f"{m.group(1)} [NAME]") if label == "NAME_INTRO" else f"[{label}]"
                text = pat.sub(repl, text)
        if _PROTECTED_TERMS.search(text):
            hits.append("PROTECTED_ATTRIBUTE")
            text = _PROTECTED_TERMS.sub("[PROTECTED]", text)
        out.append(s.model_copy(update={"text": text}))
    return out, sorted(set(hits))


# --- Prompt injection carried inside the video (speech or on-screen text) -------
_INJECTION = re.compile(
    r"(ignore (all |any )?(previous|prior|above) instructions|disregard .* instructions|"
    r"you are now|system prompt|rate (me|this candidate) \d|give (me|this candidate) (a )?(5|five|full marks)|"
    r"as an ai)", re.I)


def detect_injection(segments: list[Segment]) -> list[Segment]:
    return [s for s in segments if _INJECTION.search(s.text)]


def strip_injection(segments: list[Segment]) -> list[Segment]:
    return [s.model_copy(update={"text": "[REMOVED: instruction-like content]"})
            if _INJECTION.search(s.text) else s for s in segments]


# --- Output checks on the scorer -------------------------------------------------
def unsupported_evidence(score: CriterionScore, transcript: str) -> list[str]:
    """Every quoted piece of evidence must appear verbatim in the transcript.
    Catches hallucinated evidence before a human ever sees it."""
    return [e.quote for e in score.evidence if e.quote not in transcript]


def mentions_prohibited(rationale: str, prohibited: list[str]) -> list[str]:
    """Whole-word match, so "age" doesn't fire on "engagement"."""
    return [p for p in prohibited
            if re.search(rf"\b{re.escape(p.replace('_', ' '))}\b", rationale, re.I)]
