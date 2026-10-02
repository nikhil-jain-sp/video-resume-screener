"""Recruiter review and ATS sync. The graph pauses here (LangGraph interrupt, state
checkpointed) until a person decides. Nothing reaches the ATS without that decision."""
from langgraph.types import interrupt

from .. import config, tools

ACTIONS = {"advance", "hold", "reject", "rerun"}


def recruiter_review(state):
    packet = {
        "candidate_id": state["candidate_id"],
        "tier": state["tier"],
        "weighted_score": state.get("weighted_score"),
        "scores": [s.model_dump() for s in state.get("scores", [])],
        "verifications": [v.model_dump() for v in state.get("verifications", [])],
        "notes": {a: r.notes for a, r in state.get("reports", {}).items() if r.notes},
        "flags": [f.model_dump() for f in state.get("flags", {}).values()],
        "actions": sorted(ACTIONS),
    }
    decision = interrupt(packet)
    status = decision.get("status")
    if status not in ACTIONS:
        raise ValueError(f"Unknown action {status!r}")
    if not decision.get("reviewer") or not decision.get("reason"):
        raise ValueError("A decision needs the reviewer's name and a reason.")
    if status == "rerun" and state.get("reruns", 0) >= config.MAX_RERUNS:
        raise ValueError("Re-run limit reached; please decide.")
    corrections = decision.get("corrections", {})
    if corrections:
        tools.OVERRIDES.append({"candidate_id": state["candidate_id"], "corrections": corrections,
                                "reviewer": decision["reviewer"], "reason": decision["reason"]})
    out = {"decision": decision, "corrections": corrections,
           "audit": [f"recruiter_review: {decision['reviewer']} -> {status}"
                     + (f" corrections={corrections}" if corrections else "")]}
    if status == "rerun":
        out["reruns"] = state.get("reruns", 0) + 1
        # Everything from perception onwards runs again and re-raises whatever still
        # applies, so drop its old flags. Intake doesn't re-run; its flags stay.
        out["flags"] = {k: None for k, f in state.get("flags", {}).items() if f.agent != "intake"}
    return out


def sync_ats(state):
    d = state["decision"]
    tools.ats_update(state["candidate_id"], d["status"], f"{d['reviewer']}: {d['reason']}")
    return {"audit": ["sync_ats: written"]}
