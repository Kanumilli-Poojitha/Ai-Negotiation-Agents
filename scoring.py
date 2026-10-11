import re
from typing import Dict, List

CONCESSION_KEYWORDS = [
    "agree", "accept", "concede", "lower", "compromise", "reduce", "flexible",
    "willing", "phased", "mutual", "balanced", "welcome", "consider", "meet halfway",
]
COMBATIVE_KEYWORDS = [
    "reject", "refuse", "unacceptable", "demand", "never", "non-negotiable",
    "oppose", "threaten", "retaliate", "insist", "will not", "cannot accept",
]


def _count(text: str, keywords: List[str]) -> int:
    low = text.lower()
    return sum(len(re.findall(r"\b" + re.escape(k) + r"\w*", low)) for k in keywords)


def calculate_compromise(history: List[Dict], positions: dict | None = None) -> float:
    """Heuristic compromise score in [0.0, 1.0] based on concession vs combative keywords.

    The second half of the negotiation is weighted more heavily, since late-stage
    movement matters more for reaching agreement.
    """
    if not history:
        return 0.0
    concessions = combative = 0.0
    n = len(history)
    for i, rnd in enumerate(history):
        weight = 1.0 + (i / max(1, n - 1)) if n > 1 else 1.0
        text = " ".join(str(v) for k, v in rnd.items() if k != "round")
        concessions += weight * _count(text, CONCESSION_KEYWORDS)
        combative += weight * _count(text, COMBATIVE_KEYWORDS)
    total = concessions + combative
    if total == 0:
        return 0.5
    return round(max(0.0, min(1.0, concessions / total)), 4)
