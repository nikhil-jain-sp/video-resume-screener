"""Thresholds. These are starting guesses; the plan is to tune them in shadow mode."""
import os

MAX_ANALYSED_SECONDS = 300      # longer videos: only the first 5 minutes are analysed
MIN_ASR_CONF = 0.75             # below this we try the fallback model, then send to a person
MIN_WORDS_PER_MINUTE = 40       # below this it's a highlight reel, not a talking video
ADULT_AGE = 18
MIN_COVERAGE = 0.60             # share of rubric weight that needs evidence for an overall score
MAX_DISAGREEMENT = 1            # critic vs specialist, in rubric points
FAST_TRACK_AT = 3.8
MAX_RERUNS = 2

# Models. "mock" runs offline. Anything else is sent to an OpenAI-compatible endpoint
# (set LLM_BASE_URL for a self-hosted vLLM/Ollama server or a gateway).
SPECIALIST_MODEL = os.getenv("SPECIALIST_MODEL", "mock")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "mock-critic")   # deliberately a different model family
