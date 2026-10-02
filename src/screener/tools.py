"""Tools the agents call, with local stand-ins so everything runs offline.

What each one is in production (the schema stays, only the body changes):
  probe_media      ffprobe: duration, codec, loudness, SNR
  transcribe       faster-whisper large-v3 on a GPU; fallback is a second ASR vendor
  search_kb        hybrid search (BM25 + pgvector) over the approved sports knowledge base
  verify_claim     web search + sports data API + agency CRM, called through MCP servers
  ats_update       the ATS/CRM API (or an MCP server wrapping it)
"""
from __future__ import annotations

import json
import time
from functools import wraps
from pathlib import Path

import yaml

from .state import Segment, Verification

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = json.loads((ROOT / "fixtures/candidates.json").read_text())
_KB = json.loads((ROOT / "fixtures/sports_kb.json").read_text())

# Schema the claims verifier sees when it runs on a real model with function calling.
VERIFY_CLAIM_SCHEMA = {
    "type": "function",
    "function": {
        "name": "verify_claim",
        "description": "Check one career claim (employer, credit, stat) against the CV, "
                       "public sources and the agency CRM. Returns verified, unverified or mismatch.",
        "parameters": {
            "type": "object",
            "properties": {"claim": {"type": "string", "maxLength": 300}},
            "required": ["claim"],
        },
    },
}


class ToolError(Exception):
    pass


def with_retry(attempts: int = 3, base_delay: float = 0.2):
    """Exponential backoff for rate limits and timeouts."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            for i in range(attempts):
                try:
                    return fn(*a, **kw)
                except ToolError:
                    if i == attempts - 1:
                        raise
                    time.sleep(base_delay * 2 ** i)
        return wrapper
    return deco


def probe_media(video_uri: str) -> dict:
    c = CANDIDATES[video_uri]
    return {k: c[k] for k in ("duration_s", "audio_snr_db", "consent", "age", "parental_consent", "jurisdiction")}


@with_retry()
def transcribe(video_uri: str, model: str = "primary") -> list[Segment]:
    c = CANDIDATES[video_uri]
    segs = c.get("fallback_segments", c["segments"]) if model == "fallback" else c["segments"]
    return [Segment(**s) for s in segs]


def search_kb(query: str, k: int = 2) -> list[dict]:
    q = query.lower()
    ranked = sorted(_KB, key=lambda d: -sum(tag in q for tag in d["tags"]))
    return [d for d in ranked[:k] if any(tag in q for tag in d["tags"])]


_CLAIMS = {
    "grew football series from 3k to 41k followers":
        ("verified", "Public Instagram profile linked in the application shows 41.2k followers"),
    "lead presenter for the official IPL Gujarat franchise channel for three seasons":
        ("mismatch", "Franchise channel credits list a different lead presenter for 2024-2026"),
}


@with_retry()
def verify_claim(claim: str) -> Verification:
    status, source = _CLAIMS.get(claim, ("unverified", "No source found"))
    return Verification(claim=claim, status=status, source=source)


def load_rubric(role_id: str) -> dict:
    for p in (ROOT / "rubrics").glob("*.yaml"):
        r = yaml.safe_load(p.read_text())
        if r["role_id"] == role_id:
            return r
    raise ToolError(f"No rubric for {role_id}")


ATS_LOG: list[dict] = []
OVERRIDES: list[dict] = []      # recruiter corrections, later added to the test set


def ats_update(candidate_id: str, status: str, note: str) -> dict:
    rec = {"candidate_id": candidate_id, "status": status, "note": note}
    ATS_LOG.append(rec)
    return rec
