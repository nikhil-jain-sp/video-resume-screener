"""9. Critic: a different model family checks the work.
  - every quote must be real (word for word in the transcript), or the score is dropped
  - reasoning that leans on a prohibited signal is dropped
  - it re-scores independently, and again with the transcript order reversed;
    big gaps or order-dependent answers send the candidate to review"""
from .. import config, tools
from .. import guardrails as g
from ..models import get_judge
from ..state import CriterionScore
from ._common import flag, summarise


def critic(state):
    rubric = tools.load_rubric(state["role_id"])
    transcript = " ".join(s.text for s in state["redacted"])
    flags, checked = {}, []
    for s in state["scores"]:
        fake = g.unsupported_evidence(s, transcript)
        banned = g.mentions_prohibited(s.rationale, rubric["prohibited_signals"])
        if s.score is not None and (fake or banned):
            flags |= flag("critic", f"DROPPED_{s.criterion.upper()}", "review",
                          f"Dropped: quotes not in transcript={fake}, prohibited signals={banned}")
            s = CriterionScore(criterion=s.criterion, rationale="Dropped by critic.")
        checked.append(s)

    judge = get_judge(config.CRITIC_MODEL)
    forward = {c.criterion: c.score for c in judge.score(state["redacted"], rubric["criteria"], rubric["prohibited_signals"])}
    reverse = {c.criterion: c.score for c in judge.score(list(reversed(state["redacted"])), rubric["criteria"], rubric["prohibited_signals"])}
    gaps = {s.criterion: abs(s.score - forward[s.criterion]) for s in checked
            if s.score is not None and forward.get(s.criterion) is not None}
    worst = max(gaps.values(), default=0)
    if worst > config.MAX_DISAGREEMENT:
        flags |= flag("critic", "DISAGREEMENT", "review", f"Critic and agents differ by up to {worst} points: {gaps}")
    unstable = [k for k in forward if forward[k] != reverse.get(k)]
    if unstable:
        flags |= flag("critic", "ORDER_SENSITIVE", "review", f"Scores changed when order was swapped: {unstable}")

    coverage, weighted = summarise(checked, rubric)
    return {"scores": checked, "coverage": coverage, "weighted_score": weighted, "flags": flags,
            "audit": [f"critic: model={judge.name} max_gap={worst} weighted={weighted}"]}
