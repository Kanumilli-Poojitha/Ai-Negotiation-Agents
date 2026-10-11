import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.negotiator import Negotiator
from scoring import calculate_compromise

BASE_DIR = Path(__file__).resolve().parent
POSITIONS_PATH = BASE_DIR / "data" / "trade_positions.json"
LOG_PATH = BASE_DIR / "negotiation_log.json"
AGREEMENT_THRESHOLD = 0.6

app = FastAPI(title="AI Trade Negotiation Agents")

with open(POSITIONS_PATH, "r", encoding="utf-8") as f:
    POSITIONS = json.load(f)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
usa_agent = Negotiator("usa", POSITIONS["usa"], OLLAMA_URL)
china_agent = Negotiator("china", POSITIONS["china"], OLLAMA_URL)

_log_lock = asyncio.Lock()


class NegotiateRequest(BaseModel):
    issue: str = Field(..., min_length=1)
    rounds: int = Field(3, ge=1, le=10)


def _append_log(entry: dict) -> None:
    entries = []
    if LOG_PATH.exists():
        try:
            entries = json.loads(LOG_PATH.read_text(encoding="utf-8")) or []
        except json.JSONDecodeError:
            entries = []
    entries.append(entry)
    LOG_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/negotiate")
async def negotiate(request: NegotiateRequest):
    history = []
    try:
        for i in range(1, request.rounds + 1):
            usa_text = await usa_agent.make_proposal(request.issue, history)
            partial = history + [{"round": i, "usa_proposal": usa_text}]
            china_text = await china_agent.make_proposal(request.issue, partial)
            history.append(
                {"round": i, "usa_proposal": usa_text, "china_response": china_text}
            )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Negotiation failed: {exc}")

    score = float(calculate_compromise(history, POSITIONS))
    last = history[-1]
    outcome = {
        "agreement_reached": score > AGREEMENT_THRESHOLD,
        "final_terms": f"USA: {last['usa_proposal']} | China: {last['china_response']}",
        "compromise_score": score,
    }
    result = {"rounds": history, "outcome": outcome}

    async with _log_lock:
        _append_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "issue": request.issue,
                **result,
            }
        )
    return result
