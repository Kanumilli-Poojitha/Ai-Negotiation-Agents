import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
from agents.negotiator import Negotiator  # noqa: E402
from scoring import calculate_compromise  # noqa: E402

POSITIONS = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "trade_positions.json").read_text()
)
LIVE = os.getenv("RUN_LIVE_LLM") == "1"


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch, tmp_path):
    """Deterministic stub instead of Ollama, unless RUN_LIVE_LLM=1 (live model)."""
    monkeypatch.setattr(main, "LOG_PATH", tmp_path / "negotiation_log.json")
    if LIVE:
        return

    async def fake(self, prompt: str) -> str:
        if self.country == "usa":
            return "We agree to lower tariffs on technology if IP protection is strengthened."
        return "China will accept expanded market access but needs flexible technology transfer terms."

    monkeypatch.setattr(Negotiator, "generate_response", fake)


@pytest.fixture
def client():
    return TestClient(main.app)


def test_returns_three_rounds(client):
    r = client.post("/negotiate", json={"issue": "Test", "rounds": 3})
    assert r.status_code == 200
    data = r.json()
    assert len(data["rounds"]) == 3


def test_returns_one_round(client):
    data = client.post("/negotiate", json={"issue": "Test", "rounds": 1}).json()
    assert len(data["rounds"]) == 1


def test_default_rounds_is_three(client):
    assert len(client.post("/negotiate", json={"issue": "Test"}).json()["rounds"]) == 3


def test_round_schema_and_outcome(client):
    data = client.post("/negotiate", json={"issue": "Technology Tariffs", "rounds": 2}).json()
    assert set(data) >= {"rounds", "outcome"}
    first = data["rounds"][0]
    assert isinstance(first["usa_proposal"], str) and first["usa_proposal"]
    assert isinstance(first["china_response"], str) and first["china_response"]
    o = data["outcome"]
    assert isinstance(o["agreement_reached"], bool)
    assert isinstance(o["final_terms"], str)
    assert isinstance(o["compromise_score"], (int, float))
    assert 0.0 <= o["compromise_score"] <= 1.0


def test_proposals_reference_initial_priorities(client):
    data = client.post("/negotiate", json={"issue": "Technology Tariffs", "rounds": 3}).json()
    usa_text = " ".join(r["usa_proposal"] for r in data["rounds"]).lower()
    china_text = " ".join(r["china_response"] for r in data["rounds"]).lower()
    assert any(t in usa_text for t in ["tariff", "ip", "intellectual property", "protection"])
    assert any(t in china_text for t in ["market access", "agricultur", "technology transfer"])


def test_negotiation_is_logged(client, tmp_path):
    log = tmp_path / "negotiation_log.json"
    assert not log.exists()
    client.post("/negotiate", json={"issue": "Test", "rounds": 2})
    entries = json.loads(log.read_text())
    assert len(entries) == 1 and len(entries[0]["rounds"]) == 2
    client.post("/negotiate", json={"issue": "Test", "rounds": 1})
    assert len(json.loads(log.read_text())) == 2


def test_invalid_request_rejected(client):
    assert client.post("/negotiate", json={"rounds": 2}).status_code == 422
    assert client.post("/negotiate", json={"issue": "x", "rounds": 0}).status_code == 422


def test_scoring_bounds():
    assert calculate_compromise([], POSITIONS) == 0.0
    friendly = [{"round": 1, "usa_proposal": "We agree and accept", "china_response": "We agree"}]
    hostile = [{"round": 1, "usa_proposal": "We reject this", "china_response": "Unacceptable, we refuse"}]
    assert calculate_compromise(friendly, POSITIONS) == 1.0
    assert calculate_compromise(hostile, POSITIONS) == 0.0


def test_positions_loaded_from_file():
    assert main.POSITIONS == POSITIONS
    assert main.usa_agent.positions["priorities"] == POSITIONS["usa"]["priorities"]


def test_fallback_when_llm_down(monkeypatch):
    async def boom(self, prompt):
        raise RuntimeError("down")

    monkeypatch.setattr(Negotiator, "generate_response", boom)
    agent = Negotiator("usa", POSITIONS["usa"], "http://nowhere")
    text = asyncio.run(agent.make_proposal("Tariffs", []))
    assert "tariff" in text.lower()
