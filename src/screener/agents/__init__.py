"""One module per agent in the one-pager diagram (numbers match the diagram)."""
from .intake import intake                                   # 1
from .perception import perception                           # 2
from .guard import guard                                     # 3
from .specialists import communication, domain, signal_quality  # 4, 5, 7
from .claims import claims_verifier                          # 6
from .synthesizer import synthesizer                         # 8
from .critic import critic                                   # 9
from .router import router                                   # 10
from .review import recruiter_review, sync_ats

__all__ = ["intake", "perception", "guard", "communication", "domain", "claims_verifier",
           "signal_quality", "synthesizer", "critic", "router", "recruiter_review", "sync_ats"]
