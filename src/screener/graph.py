"""The workflow as a LangGraph StateGraph. Node names match the one-pager diagram.

  intake ─┬─ (no consent / minor without parental consent) ──────────────► END (manual queue)
          └─► perception ─┬─ (highlight reel) ─────────────────────────────► router
                          └─► guard ─┬─► communication ─┐
                                     ├─► domain ────────┤
                                     ├─► claims_verifier┼─► synthesizer ─► critic ─► router
                                     └─► signal_quality ┘
  router ─► recruiter_review (interrupt) ─┬─ rerun ─► perception (stale flags cleared)
                                          └─ advance / hold / reject ─► sync_ats ─► END

Rules enforced in code, not prompts:
  * no reject tier; every analysed candidate stops at recruiter_review
  * scoring agents only see the redacted transcript and have no tools
  * a score without a real quote is dropped by the critic
  * only sync_ats writes to the ATS, and only after a human decision
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from . import agents
from .state import ScreenState

SPECIALISTS = ["communication", "domain", "claims_verifier", "signal_quality"]


def _after_intake(state):
    return "stop" if state.get("tier") == "human_only" else "perception"


def _after_perception(state):
    return "router" if state.get("low_speech") else "guard"


def _after_review(state):
    # A re-run starts at perception, so a re-uploaded or cleaned-up video is transcribed
    # again and a highlight reel still skips scoring.
    return "perception" if state["decision"]["status"] == "rerun" else "sync_ats"


def build_graph(checkpointer=None):
    g = StateGraph(ScreenState)
    for name in ["intake", "perception", "guard", *SPECIALISTS, "synthesizer", "critic",
                 "router", "recruiter_review", "sync_ats"]:
        g.add_node(name, getattr(agents, name))

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", _after_intake, {"stop": END, "perception": "perception"})
    g.add_conditional_edges("perception", _after_perception, {"router": "router", "guard": "guard"})
    for s in SPECIALISTS:                       # fan out
        g.add_edge("guard", s)
    g.add_edge(SPECIALISTS, "synthesizer")      # wait for all four
    g.add_edge("synthesizer", "critic")
    g.add_edge("critic", "router")
    g.add_edge("router", "recruiter_review")
    g.add_conditional_edges("recruiter_review", _after_review, {"perception": "perception", "sync_ats": "sync_ats"})
    g.add_edge("sync_ats", END)

    # In production this is a PostgresSaver, so a paused review survives restarts and deploys.
    serde = JsonPlusSerializer(allowed_msgpack_modules=[
        ("screener.state", n) for n in
        ("Segment", "Evidence", "CriterionScore", "AgentReport", "Verification", "Flag")])
    return g.compile(checkpointer=checkpointer or InMemorySaver(serde=serde))
