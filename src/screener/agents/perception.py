"""2. Perception: transcript with timestamps and confidence. Falls back to a second
speech-to-text model when confidence is low, and spots highlight reels with little speech."""
from statistics import mean

from .. import config, tools
from ..models import words_per_minute
from ._common import flag


def perception(state):
    meta = tools.probe_media(state["video_uri"])
    segs, model = tools.transcribe(state["video_uri"]), "primary"
    conf = mean(s.conf for s in segs) if segs else 0.0
    if conf < config.MIN_ASR_CONF:
        retry = tools.transcribe(state["video_uri"], model="fallback")
        retry_conf = mean(s.conf for s in retry) if retry else 0.0
        if retry_conf > conf:
            segs, conf, model = retry, retry_conf, "fallback"
    flags = {}
    if conf < config.MIN_ASR_CONF:
        flags |= flag("perception", "LOW_ASR_CONFIDENCE", "review",
                      f"Transcript confidence {conf:.2f} even after the fallback model. "
                      "Scores on it are unreliable, so a person should watch the video.")
    wpm = words_per_minute(segs, meta["duration_s"])
    low_speech = wpm < config.MIN_WORDS_PER_MINUTE
    if low_speech:
        flags |= flag("perception", "HIGHLIGHT_REEL", "review",
                      f"About {wpm:.0f} words a minute: mostly footage, not talking. "
                      "Skipping scoring and sending straight to a scout.")
    return {"segments": segs, "asr_conf": conf, "asr_model": model, "low_speech": low_speech,
            "flags": flags, "audit": [f"perception: asr={model} conf={conf:.2f} wpm={wpm:.0f}"]}
