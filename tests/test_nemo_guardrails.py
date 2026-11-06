"""NeMo Guardrails layer. Skipped automatically if `nemoguardrails` is not installed."""

import asyncio

import pytest

pytest.importorskip("nemoguardrails")
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from src.tools import guardrails as g  # noqa: E402

CODE = "print(df['x'].mean())"


def fake(answer, n=4):
    return FakeListChatModel(responses=[answer] * n)


def test_nemo_blocks_when_rail_says_yes():
    r = g.nemo_check(CODE, llm=fake("Yes"))
    assert not r.allowed and r.kind == "nemo"


def test_nemo_allows_when_rail_says_no():
    assert g.nemo_check(CODE, llm=fake("No")).allowed


def test_nemo_fails_closed_when_llm_errors():
    class Boom(FakeListChatModel):
        def _call(self, *a, **k):
            raise RuntimeError("rate limited")

    r = g.nemo_check(CODE, llm=Boom(responses=["x"]))
    assert not r.allowed and r.kind == "judge_unavailable"


async def test_nemo_works_inside_running_event_loop():
    assert asyncio.get_running_loop()
    assert g.nemo_check(CODE, llm=fake("No")).allowed
    assert not g.nemo_check(CODE, llm=fake("Yes")).allowed  # repeated calls, fresh loops


def test_static_policy_still_wins_over_nemo(monkeypatch):
    from src.configs.settings import get_settings

    monkeypatch.setenv("ENABLE_NEMO_GUARDRAILS", "true")
    get_settings.cache_clear()
    try:
        r = g.check_code("import os", "s1", llm=fake("No"))
        assert not r.allowed and r.kind != "nemo"  # blocked by static policy, NeMo never consulted
        assert g.check_code(CODE, "s2", llm=fake("No")).allowed
        assert not g.check_code(CODE, "s3", llm=fake("Yes")).allowed
    finally:
        get_settings.cache_clear()


async def _events(agent, df):
    return [e async for e in agent.run_streaming("Compute the mean of x column", dataframe=df)]


@pytest.mark.parametrize("answer,expect_blocked", [("No", False), ("Yes", True)])
async def test_agent_end_to_end_with_nemo_on(make_agent, df, monkeypatch, answer, expect_blocked):
    from src.configs.settings import get_settings

    monkeypatch.setenv("ENABLE_NEMO_GUARDRAILS", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(g, "_nemo_rails", None)
    monkeypatch.setattr(g, "_build_judge_llm", lambda: fake(answer, 50))
    try:
        agent = make_agent(code_writes=["print(df['x'].mean())"], code_fixes=[])
        ev = await _events(agent, df)
        s1 = next(e for e in ev if e.event_type == "step_complete" and e.data.get("step_id") == "s1")
        assert (s1.data.get("error_type") == "SecurityError") is expect_blocked
    finally:
        get_settings.cache_clear()
        g._nemo_rails = None
