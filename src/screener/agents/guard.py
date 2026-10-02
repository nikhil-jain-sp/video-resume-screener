"""3. Guard (no LLM): strip identity details and instruction-like content before any
scoring agent sees the transcript. Plain code on purpose, so it can't be talked out of its job."""
from .. import guardrails as g
from .. import tools
from ._common import flag


def guard(state):
    segs = state["segments"]
    flags = {}
    injected = g.detect_injection(segs)
    if injected:
        flags |= flag("guard", "PROMPT_INJECTION", "review",
                      f"Instruction-like speech at {[s.t for s in injected]}s removed before scoring.")
        segs = g.strip_injection(segs)
    redacted, hits = g.redact(segs)
    # Stand-in for the CV parser: checkable claims from the application.
    claims = tools.CANDIDATES[state["video_uri"]]["claims"]
    return {"redacted": redacted, "claims": claims, "flags": flags,
            "audit": [f"guard: redacted={hits} injected_segments={len(injected)}"]}
