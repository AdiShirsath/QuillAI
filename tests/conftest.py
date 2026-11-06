"""Shared fixtures. No network, no API key: the LLM is replaced by a scripted fake."""

import json
import os
import types

import pandas as pd
import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ["ENABLE_EPISODIC_MEMORY"] = "false"
os.environ["REDIS_HOST"] = "127.0.0.1"
os.environ["REDIS_PORT"] = "1"  # nothing listens here -> in-memory fallback
os.environ["MAX_RETRIES_PER_STEP"] = "3"


def _msg(content):
    return types.SimpleNamespace(content=content)


PLAN = {
    "reasoning": "simple",
    "estimated_complexity": "low",
    "requires_clarification": False,
    "clarification_questions": [],
    "steps": [
        {
            "step_id": "s1",
            "step_type": "CODE",
            "title": "Describe",
            "description": "Print mean of x",
            "rationale": "r",
            "expected_output": "mean",
            "depends_on": [],
        },
        {
            "step_id": "s2",
            "step_type": "SUMMARIZE",
            "title": "Summary",
            "description": "Summarize",
            "rationale": "r",
            "expected_output": "text",
            "depends_on": ["s1"],
        },
    ],
}


class FakeLLM:
    """Routes on the system prompt. `code_writes` / `code_fixes` are consumed in order."""

    def __init__(self, code_writes=None, code_fixes=None):
        self.code_writes = list(code_writes or ["print(df['x'].mean())"])
        self.code_fixes = list(code_fixes or [])
        self.calls = []

    def invoke(self, messages):
        system = messages[0].content
        self.calls.append(system[:40])
        if "analytical planner" in system:
            return _msg(json.dumps(PLAN))
        if "expert Python data scientist" in system:
            return _msg(self.code_writes.pop(0))
        if "debugging Python code" in system:
            return _msg(self.code_fixes.pop(0) if self.code_fixes else "")
        if "interpreting code output" in system:
            return _msg(
                json.dumps({"interpretation": "mean is 2", "key_findings": ["mean x is 2"], "confidence_score": 0.9})
            )
        return _msg(json.dumps({"output": "Final summary", "key_findings": ["done"], "confidence_score": 0.8}))


@pytest.fixture
def df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [3, 2, 1]})


@pytest.fixture
def make_agent(monkeypatch):
    def _make(**kw):
        from src.agent import agent as agent_mod

        fake = FakeLLM(**kw)
        monkeypatch.setattr("src.agent.executor.build_llm", lambda *a, **k: fake)
        monkeypatch.setattr("src.agent.planner.build_llm", lambda *a, **k: fake)
        a = agent_mod.DataAnalysisAgent()
        a._fake = fake
        return a

    return _make
