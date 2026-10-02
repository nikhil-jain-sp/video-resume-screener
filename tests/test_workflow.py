import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest  # noqa: E402
from langgraph.types import Command  # noqa: E402

from screener import config, guardrails as g, tools  # noqa: E402
from screener.graph import build_graph  # noqa: E402
from screener.state import CriterionScore, Evidence, Segment  # noqa: E402

ROLE = "REQ-2026-014"


def start(cid):
    graph = build_graph()
    cfg = {"configurable": {"thread_id": cid}}
    return graph, cfg, graph.invoke({"candidate_id": cid, "video_uri": cid, "role_id": ROLE}, cfg)


def packet(result):
    return result["__interrupt__"][0].value


def codes(result):
    flags = packet(result)["flags"] if "__interrupt__" in result else [f.model_dump() for f in result["flags"].values()]
    return {f["code"] for f in flags}


OK = {"status": "hold", "reviewer": "asha@agency", "reason": "Watched the clips."}


# --- routing ----------------------------------------------------------------------
@pytest.mark.parametrize("cid", ["cand_001", "cand_002", "cand_003", "cand_005", "cand_006"])
def test_every_analysed_candidate_stops_at_a_person(cid):
    _, _, r = start(cid)
    assert "__interrupt__" in r


def test_router_has_no_reject_tier():
    tiers = set()
    for cid in tools.CANDIDATES:
        _, _, r = start(cid)
        tiers.add(packet(r)["tier"] if "__interrupt__" in r else r["tier"])
    assert "reject" not in tiers
    assert tiers <= {"fast_track", "standard", "needs_review", "human_only"}


def test_strong_candidate_is_fast_tracked():
    _, _, r = start("cand_001")
    assert packet(r)["tier"] == "fast_track"


def test_no_consent_means_no_model_sees_the_video():
    _, _, r = start("cand_004")
    assert r["tier"] == "human_only" and "segments" not in r
    assert "NO_AI_CONSENT" in codes(r)


def test_under_18_always_goes_to_review():
    _, _, r = start("cand_005")
    assert "UNDER_18" in codes(r)
    assert packet(r)["tier"] == "needs_review"


def test_highlight_reel_skips_scoring():
    _, _, r = start("cand_006")
    assert "HIGHLIGHT_REEL" in codes(r)
    assert packet(r)["tier"] == "human_only" and packet(r)["scores"] == []


def test_low_confidence_tries_the_fallback_model_then_flags():
    _, _, r = start("cand_003")
    assert "LOW_ASR_CONFIDENCE" in codes(r)
    assert any("asr=fallback" in line for line in r["audit"])


# --- guardrails ---------------------------------------------------------------------
def test_prompt_injection_is_removed_and_flagged():
    _, _, r = start("cand_002")
    assert "PROMPT_INJECTION" in codes(r)
    assert all("Ignore previous" not in s.text for s in r["redacted"])


def test_claim_mismatch_goes_to_review():
    _, _, r = start("cand_002")
    assert "CLAIM_MISMATCH" in codes(r) and packet(r)["tier"] == "needs_review"


def test_pii_is_redacted_before_scoring():
    red, hits = g.redact([Segment(t=0, conf=1, text="I'm Jordan, 34 years old, call 98450 12345")])
    assert "Jordan" not in red[0].text and "34" not in red[0].text and "98450" not in red[0].text
    assert {"AGE", "PHONE", "NAME_INTRO"} <= set(hits)


def test_name_redaction_only_takes_the_name():
    red, _ = g.redact([Segment(t=0, conf=1, text="I am Arjun from Kochi"),
                       Segment(t=1, conf=1, text="I am passionate about cricket analysis"),
                       Segment(t=2, conf=1, text="i'm Priya Nair, hi")])
    assert [s.text for s in red] == ["I am [NAME] from Kochi",
                                     "I am passionate about cricket analysis",
                                     "i'm [NAME], hi"]


def test_made_up_quotes_are_caught():
    s = CriterionScore(criterion="x", score=5, evidence=[Evidence(quote="I won an Emmy", t=3)])
    assert g.unsupported_evidence(s, "I ran a football series") == ["I won an Emmy"]


def test_prohibited_signal_check_uses_whole_words():
    assert g.mentions_prohibited("Strong accent, hard to follow", ["accent", "age"]) == ["accent"]
    assert g.mentions_prohibited("18 percent engagement", ["age"]) == []


def test_signal_quality_never_scores():
    _, _, r = start("cand_001")
    assert r["reports"]["signal_quality"].scores == []


def test_critic_runs_on_its_own_model():
    _, _, r = start("cand_001")
    assert any(f"model={config.CRITIC_MODEL}" in line for line in r["audit"])
    assert config.CRITIC_MODEL != config.SPECIALIST_MODEL


# --- the human step ------------------------------------------------------------------
def test_nothing_reaches_the_ats_before_a_decision():
    before = len(tools.ATS_LOG)
    start("cand_001")
    assert len(tools.ATS_LOG) == before


def test_decision_needs_reviewer_and_reason():
    graph, cfg, _ = start("cand_001")
    with pytest.raises(ValueError):
        graph.invoke(Command(resume={"status": "advance"}), cfg)


def test_rerun_goes_back_through_scoring_then_waits_again():
    graph, cfg, _ = start("cand_003")
    r = graph.invoke(Command(resume={**OK, "status": "rerun"}), cfg)
    assert "__interrupt__" in r
    assert sum(line.startswith("synthesizer") for line in r["audit"]) == 2


def test_rerun_limit():
    graph, cfg, _ = start("cand_003")
    for _ in range(config.MAX_RERUNS):
        graph.invoke(Command(resume={**OK, "status": "rerun"}), cfg)
    with pytest.raises(ValueError):
        graph.invoke(Command(resume={**OK, "status": "rerun"}), cfg)


def test_recruiter_corrections_are_kept_for_the_test_set():
    graph, cfg, _ = start("cand_001")
    before = len(tools.OVERRIDES)
    graph.invoke(Command(resume={**OK, "status": "advance", "corrections": {"delivery_clarity": 4}}), cfg)
    assert len(tools.OVERRIDES) == before + 1
    assert tools.ATS_LOG[-1]["status"] == "advance"


def test_rerun_transcribes_again_and_clears_stale_flags(monkeypatch):
    graph, cfg, r = start("cand_003")
    assert "LOW_ASR_CONFIDENCE" in codes(r)
    # The audio gets cleaned up: the new transcript is clear.
    clean = copy.deepcopy(tools.CANDIDATES["cand_003"])
    clean["segments"] = [{**s, "conf": 0.95} for s in clean.pop("fallback_segments")]
    monkeypatch.setitem(tools.CANDIDATES, "cand_003", clean)
    r = graph.invoke(Command(resume={**OK, "status": "rerun"}), cfg)
    assert sum(line.startswith("perception") for line in r["audit"]) == 2
    assert "LOW_ASR_CONFIDENCE" not in codes(r)


def test_rerun_keeps_intake_flags():
    graph, cfg, _ = start("cand_005")
    r = graph.invoke(Command(resume={**OK, "status": "rerun"}), cfg)
    assert "UNDER_18" in codes(r) and packet(r)["tier"] == "needs_review"


def test_rerun_of_highlight_reel_still_skips_scoring():
    graph, cfg, _ = start("cand_006")
    r = graph.invoke(Command(resume={**OK, "status": "rerun"}), cfg)
    assert "HIGHLIGHT_REEL" in codes(r)
    assert packet(r)["tier"] == "human_only" and packet(r)["scores"] == []
    assert not any(line.startswith("synthesizer") for line in r["audit"])
