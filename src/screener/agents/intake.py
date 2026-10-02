"""1. Intake (no LLM): is the file usable, do we have consent, is the candidate a minor?"""
from .. import config, tools
from ._common import flag


def intake(state):
    meta = tools.probe_media(state["video_uri"])
    flags, tier = {}, None
    if not meta["consent"]:
        tier = "human_only"
        flags |= flag("intake", "NO_AI_CONSENT", "block",
                      "No consent to AI analysis. Sent to the manual queue; no model sees this video.")
    elif meta["age"] < config.ADULT_AGE and not meta["parental_consent"]:
        tier = "human_only"
        flags |= flag("intake", "UNDER_18_NO_PARENTAL_CONSENT", "block",
                      "Under 18 without verified parental consent. Manual queue only.")
    elif meta["age"] < config.ADULT_AGE:
        flags |= flag("intake", "UNDER_18", "review", "Under 18: always reviewed by a person.")
    if meta["duration_s"] > config.MAX_ANALYSED_SECONDS:
        flags |= flag("intake", "LONG_VIDEO", "info", "Only the first 5 minutes are analysed.")
    out = {"consent": meta["consent"], "age": meta["age"], "flags": flags, "reruns": 0,
           "audit": [f"intake: consent={meta['consent']} age={meta['age']}"]}
    if tier:
        out["tier"] = tier
    return out
