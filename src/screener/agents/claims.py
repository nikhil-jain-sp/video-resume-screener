"""6. Claims verifier: the only agent with tools. Bounded: one call per claim, capped."""
from .. import tools
from ._common import flag

MAX_CLAIMS = 5


def claims_verifier(state):
    results, flags = [], {}
    for claim in state.get("claims", [])[:MAX_CLAIMS]:
        try:
            v = tools.verify_claim(claim)
        except tools.ToolError as e:
            v = tools.Verification(claim=claim, status="unverified", source=f"tool error: {e}")
        results.append(v)
        if v.status == "mismatch":
            flags |= flag("claims_verifier", "CLAIM_MISMATCH", "review", f"'{claim}': {v.source}")
    return {"verifications": results, "flags": flags,
            "audit": [f"claims_verifier: {[v.status for v in results]}"]}
