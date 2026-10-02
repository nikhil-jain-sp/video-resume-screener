"""State shared by every node in the graph.

Fields written by parallel agents use a merge-dict reducer keyed by agent name, so
when a recruiter sends a candidate back for a re-run, the new output replaces the old
one instead of piling up next to it.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


def merge(left: dict | None, right: dict | None) -> dict:
    """Merge by key. A value of None deletes the key (used to clear stale flags on a re-run)."""
    return {k: v for k, v in {**(left or {}), **(right or {})}.items() if v is not None}


class Segment(BaseModel):
    t: float                      # start time, seconds
    conf: float                   # speech-to-text confidence, 0..1
    text: str


class Evidence(BaseModel):
    quote: str                    # must appear word for word in the redacted transcript
    t: float                      # so the recruiter can jump straight to the clip


class CriterionScore(BaseModel):
    criterion: str
    score: Optional[int] = Field(None, ge=1, le=5)   # None means "not enough evidence"
    evidence: list[Evidence] = []
    rationale: str = ""


class AgentReport(BaseModel):
    agent: str
    model: str
    scores: list[CriterionScore] = []
    notes: list[str] = []         # shown to the recruiter, never used for ranking


class Verification(BaseModel):
    claim: str
    status: Literal["verified", "unverified", "mismatch"]
    source: str = ""


class Flag(BaseModel):
    code: str                     # e.g. PROMPT_INJECTION, UNDER_18, LOW_ASR_CONFIDENCE
    severity: Literal["info", "review", "block"]
    detail: str
    agent: str


Tier = Literal["fast_track", "standard", "needs_review", "human_only"]


class ScreenState(TypedDict, total=False):
    # inputs
    candidate_id: str
    video_uri: str
    role_id: str
    # 1 intake
    consent: bool
    age: int
    # 2 perception
    segments: list[Segment]
    asr_conf: float
    asr_model: str
    low_speech: bool
    # 3 guard
    redacted: list[Segment]
    claims: list[str]
    # 4-7 specialists (parallel)
    reports: Annotated[dict[str, AgentReport], merge]
    verifications: list[Verification]
    # 8 synthesizer, 9 critic
    scores: list[CriterionScore]
    coverage: float
    weighted_score: Optional[float]
    # 10 router
    tier: Tier
    # recruiter
    decision: dict
    corrections: dict
    reruns: int
    # cross-cutting
    flags: Annotated[dict[str, Flag], merge]
    audit: Annotated[list[str], operator.add]
