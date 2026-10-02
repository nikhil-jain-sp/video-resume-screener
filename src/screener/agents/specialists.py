"""4. Communication, 5. Domain, 7. Signal quality. These run in parallel and have no tools,
so injected text has nothing to hijack. Each one only scores the criteria it owns."""
from .. import config, tools
from ..models import get_judge, words_per_minute
from ..state import AgentReport


def _score(state, agent: str, extra_context: list[dict] | None = None) -> dict:
    rubric = tools.load_rubric(state["role_id"])
    criteria = [dict(c, reference=extra_context or []) for c in rubric["criteria"] if c["owner"] == agent]
    judge = get_judge(config.SPECIALIST_MODEL)
    scores = judge.score(state["redacted"], criteria, rubric["prohibited_signals"])
    return AgentReport(agent=agent, model=judge.name, scores=scores)


def communication(state):
    report = _score(state, "communication")
    return {"reports": {"communication": report},
            "audit": [f"communication: {[(s.criterion, s.score) for s in report.scores]}"]}


def domain(state):
    # Ground the domain judgement in the approved knowledge base, not the model's memory.
    transcript = " ".join(s.text for s in state["redacted"])
    refs = tools.search_kb(transcript)
    report = _score(state, "domain", extra_context=refs)
    report.notes = [f"grounded on {[r['id'] for r in refs]}"]
    return {"reports": {"domain": report},
            "audit": [f"domain: {[(s.criterion, s.score) for s in report.scores]} kb={[r['id'] for r in refs]}"]}


def signal_quality(state):
    """Notes for the recruiter only. Nothing here feeds the score."""
    meta = tools.probe_media(state["video_uri"])
    text = " ".join(s.text.lower() for s in state["redacted"])
    notes = [f"audio SNR {meta['audio_snr_db']} dB" + (" (noisy)" if meta["audio_snr_db"] < 12 else ""),
             f"pace about {words_per_minute(state['redacted']):.0f} words a minute"]
    if not any(w in text for w in ("takeaway", "thanks", "so what")):
        notes.append("no clear close")
    report = AgentReport(agent="signal_quality", model="rules", notes=notes)
    return {"reports": {"signal_quality": report}, "audit": [f"signal_quality: {notes}"]}
