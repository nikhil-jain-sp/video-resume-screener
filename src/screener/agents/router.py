"""10. Router (no LLM): sets review priority. There is deliberately no reject tier."""
from .. import config


def router(state):
    flags = state.get("flags", {}).values()
    if state.get("low_speech"):
        tier = "human_only"
    elif any(f.severity in ("review", "block") for f in flags) or state.get("weighted_score") is None:
        tier = "needs_review"
    elif state["weighted_score"] >= config.FAST_TRACK_AT:
        tier = "fast_track"
    else:
        tier = "standard"
    return {"tier": tier, "audit": [f"router: tier={tier}"]}
