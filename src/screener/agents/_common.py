from ..state import CriterionScore, Flag


def flag(agent: str, code: str, severity: str, detail: str) -> dict:
    return {f"{agent}:{code}": Flag(code=code, severity=severity, detail=detail, agent=agent)}


def summarise(scores: list[CriterionScore], rubric: dict) -> tuple[float, float | None]:
    """Coverage = share of rubric weight with evidence. No overall score if coverage is too low."""
    from ..config import MIN_COVERAGE
    weights = {c["id"]: c["weight"] for c in rubric["criteria"]}
    scored = [s for s in scores if s.score is not None]
    coverage = sum(weights[s.criterion] for s in scored)
    if coverage < MIN_COVERAGE:
        return coverage, None
    return coverage, sum(weights[s.criterion] * s.score for s in scored) / coverage
