"""8. Synthesizer: one score per rubric criterion, taken from the agent that owns it.
Abstentions stay abstentions. No overall score if too little of the rubric has evidence."""
from .. import tools
from ..state import CriterionScore
from ._common import flag, summarise


def synthesizer(state):
    rubric = tools.load_rubric(state["role_id"])
    by_criterion = {s.criterion: s for r in state["reports"].values() for s in r.scores}
    scores = [by_criterion.get(c["id"], CriterionScore(criterion=c["id"], rationale="Not scored."))
              for c in rubric["criteria"]]
    coverage, weighted = summarise(scores, rubric)
    flags = {} if weighted is not None else flag(
        "synthesizer", "LOW_COVERAGE", "review", f"Only {coverage:.0%} of the rubric has evidence.")
    return {"scores": scores, "coverage": coverage, "weighted_score": weighted, "flags": flags,
            "audit": [f"synthesizer: coverage={coverage:.2f} weighted={weighted}"]}
