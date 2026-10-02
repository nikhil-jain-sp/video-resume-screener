"""Runs the six sample candidates through the graph and plays the recruiter.

    python run_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langgraph.types import Command  # noqa: E402

from screener.graph import build_graph  # noqa: E402
from screener.tools import ATS_LOG, CANDIDATES, OVERRIDES  # noqa: E402

graph = build_graph()


def show(packet):
    ws = packet["weighted_score"]
    print(f"  tier={packet['tier']}  overall={'-' if ws is None else round(ws, 2)}")
    for s in packet["scores"]:
        print(f"    {s['criterion']:<17} {s['score'] if s['score'] is not None else '-':<2} {s['rationale']}")
    for agent, notes in packet["notes"].items():
        print(f"    note ({agent}): {'; '.join(notes)}")
    for v in packet["verifications"]:
        print(f"    claim: {v['status']}: {v['claim']}")
    for f in packet["flags"]:
        print(f"    [{f['severity']}] {f['code']}: {f['detail']}")


for cid in CANDIDATES:
    cfg = {"configurable": {"thread_id": cid}}
    result = graph.invoke({"candidate_id": cid, "video_uri": cid, "role_id": "REQ-2026-014"}, cfg)
    print(f"\n=== {cid} ===")
    if "__interrupt__" not in result:
        print(f"  tier={result['tier']} (no AI analysis)")
        for f in result["flags"].values():
            print(f"    [{f.severity}] {f.code}: {f.detail}")
        continue
    packet = result["__interrupt__"][0].value
    show(packet)

    # The recruiter's decision normally arrives later from the review UI.
    if cid == "cand_003":
        # Example of "send it back": recruiter asks for a re-run, then decides.
        result = graph.invoke(Command(resume={"status": "rerun", "reviewer": "asha@agency",
                                              "reason": "Re-run after audio was cleaned up."}), cfg)
        print("  ...re-run requested, back at review")
    decision = {"status": "advance" if packet["tier"] == "fast_track" else "hold",
                "reviewer": "asha@agency", "reason": "Watched the cited clips."}
    if cid == "cand_001":
        decision["corrections"] = {"delivery_clarity": 4}
    final = graph.invoke(Command(resume=decision), cfg)
    print("  audit: " + "\n         ".join(final["audit"]))

print("\nATS writes:", *ATS_LOG, sep="\n  ")
print("Recruiter corrections (go into the test set):", OVERRIDES)
