import asyncio
import logging
import os
import re
from typing import Dict, List

import httpx

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the", "and", "for", "on", "of", "to", "a", "an", "in", "greater",
    "gain", "reduce", "limit", "strengthen", "clauses", "products",
}


def extract_keywords(priorities: List[str]) -> List[str]:
    """Derive lowercase keywords from priority strings (used for grounding checks)."""
    words = set()
    for p in priorities:
        for w in re.findall(r"[A-Za-z]+", p):
            wl = w.lower()
            if wl not in STOPWORDS and len(wl) > 1:
                words.add(wl)
    return sorted(words)


class Negotiator:
    """One negotiating agent (country persona) backed by a local Ollama model."""

    def __init__(self, country: str, positions: dict, ollama_url: str):
        self.country = country
        self.positions = positions
        self.ollama_url = ollama_url.rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "llama3")
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.max_retries = 3

    @property
    def label(self) -> str:
        return {"usa": "the United States", "china": "China"}.get(
            self.country.lower(), self.country
        )

    # ---------- LLM access ----------
    async def generate_response(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": 120},
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout, connect=10.0)
                ) as client:
                    resp = await client.post(f"{self.ollama_url}/api/generate", json=payload)
                    resp.raise_for_status()
                    cleaned = self._clean(resp.json().get("response", ""))
                    if cleaned:
                        return cleaned
                    last_err = ValueError("empty LLM response")
            except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
                last_err = exc
                logger.warning("[%s] LLM attempt %d failed: %s", self.country, attempt + 1, exc)
            await asyncio.sleep(min(2 ** attempt, 8) * 0.5)  # exponential backoff
        raise RuntimeError(f"Ollama request failed after retries: {last_err}")

    @staticmethod
    def _clean(text: str) -> str:
        """Strip markdown, quotes, and conversational preambles."""
        text = re.sub(r"```.*?```", "", text.strip(), flags=re.S)
        text = re.sub(r"[*_`#>]+", "", text)
        text = " ".join(l.strip() for l in text.splitlines() if l.strip())
        text = re.sub(
            r"^(here('s| is) (my|the) [^:]{0,40}:|sure[,!.]?|certainly[,!.]?|okay[,!.]?)\s*",
            "", text, flags=re.I,
        )
        return text.strip().strip('"').strip()

    # ---------- Prompting ----------
    def build_prompt(self, issue: str, history: List[Dict]) -> str:
        priorities = "\n".join(f"- {p}" for p in self.positions.get("priorities", []))
        flex = ", ".join(f"{k}: {v}" for k, v in self.positions.get("flexibility", {}).items())
        opponent = "China" if self.country.lower() == "usa" else "USA"
        transcript = ""
        for h in history:
            if h.get("usa_proposal"):
                transcript += f"USA: {h['usa_proposal']}\n"
            if h.get("china_response"):
                transcript += f"China: {h['china_response']}\n"
        transcript = transcript or "(no prior discussion - you are opening the talks)"
        return (
            f"You are a seasoned senior trade diplomat representing {self.label}.\n"
            f"Negotiation topic: {issue}\n\n"
            f"Your priorities:\n{priorities}\n"
            f"Flexibility (0=rigid, 1=very flexible): {flex}\n\n"
            f"Negotiation so far:\n{transcript}\n"
            f"Think step by step privately about your strategy, what you can concede on flexible "
            f"items, and how to respond to {opponent}'s latest position. Do NOT show your reasoning.\n"
            f"Then output ONLY your next negotiating statement: 1-2 sentences, explicitly mentioning "
            f"at least one of your priorities by name, and offering a concrete compromise where "
            f"you are flexible. No preamble, no markdown, no quotes."
        )

    def fallback_proposal(self, issue: str, history: List[Dict]) -> str:
        prios = self.positions.get("priorities", ["mutual trade benefits"])
        p = prios[len(history) % len(prios)]
        return (
            f"On {issue}, {self.label} proposes to {p[0].lower() + p[1:]}, and we agree "
            f"to accept a phased compromise on our other demands."
        )

    def _grounded(self, text: str) -> bool:
        low = text.lower()
        return any(k in low for k in extract_keywords(self.positions.get("priorities", [])))

    async def make_proposal(self, issue: str, history: List[Dict]) -> str:
        try:
            text = await self.generate_response(self.build_prompt(issue, history))
        except RuntimeError as exc:
            logger.error("[%s] using fallback proposal: %s", self.country, exc)
            return self.fallback_proposal(issue, history)
        if not self._grounded(text):
            prios = self.positions.get("priorities", [""])
            text = f"On {prios[len(history) % len(prios)].lower()}: {text}"
        return text
