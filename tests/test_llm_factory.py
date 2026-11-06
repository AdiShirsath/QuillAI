import pytest

from src.configs.settings import get_settings
from src.tools.llm_factory import build_llm


@pytest.fixture
def env(monkeypatch):
    # Start from a known state even if your real .env sets role overrides.
    for role_var in ("PLANNER_MODEL", "EXECUTOR_MODEL", "EVALUATOR_MODEL"):
        monkeypatch.setenv(role_var, "")

    def _set(**kv):
        for k, v in kv.items():
            monkeypatch.setenv(k, v)
        get_settings.cache_clear()

    yield _set
    get_settings.cache_clear()


def test_groq_is_default_and_gets_reasoning_only_for_gpt_oss(env):
    env(LLM_PROVIDER="groq", GROQ_API_KEY="k", GROQ_MODEL="openai/gpt-oss-20b")
    llm = build_llm("executor", 0.1)
    assert type(llm).__name__ == "ChatGroq" and llm.reasoning_effort == "low" and llm.max_retries == 6
    env(GROQ_MODEL="llama-3.1-8b-instant")
    assert build_llm("executor").reasoning_effort is None


def test_mistral_selected(env):
    env(LLM_PROVIDER="mistral", MISTRAL_API_KEY="k", MISTRAL_MODEL="mistral-small-latest")
    llm = build_llm("executor", 0.0)
    assert type(llm).__name__ == "ChatMistralAI" and llm.model == "mistral-small-latest" and llm.max_retries == 6


def test_gemini_selected_and_thinking_disabled_on_flash(env):
    env(LLM_PROVIDER="gemini", GEMINI_API_KEY="k", GEMINI_MODEL="gemini-2.5-flash")
    llm = build_llm("executor", 0.1)
    assert type(llm).__name__ == "ChatGoogleGenerativeAI" and llm.max_retries == 6 and llm.thinking_budget == 0
    env(GEMINI_MODEL="gemini-2.5-pro")  # Pro cannot turn thinking off
    assert build_llm("executor").thinking_budget is None


def test_missing_key_and_unknown_provider_fail_clearly(env):
    env(LLM_PROVIDER="mistral", MISTRAL_API_KEY="")
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        build_llm("executor")
    env(LLM_PROVIDER="gemini", GEMINI_API_KEY="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_llm("executor")
    env(LLM_PROVIDER="nope")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_llm("executor")


def test_active_model_name(env):
    from src.tools.llm_factory import active_model_name

    env(LLM_PROVIDER="gemini", GEMINI_MODEL="gemini-2.5-flash")
    assert active_model_name() == "gemini:gemini-2.5-flash"


def test_role_override_beats_provider_default(env):
    env(
        LLM_PROVIDER="mistral",
        MISTRAL_API_KEY="k",
        MISTRAL_MODEL="mistral-small-latest",
        PLANNER_MODEL="mistral-large-latest",
        EXECUTOR_MODEL="",
        EVALUATOR_MODEL="",
    )
    assert build_llm("planner").model == "mistral-large-latest"
    assert build_llm("executor").model == "mistral-small-latest"
    assert build_llm("evaluator").model == "mistral-small-latest"
    from src.tools.llm_factory import active_model_name

    assert active_model_name() == "mistral:mistral-large-latest/mistral-small-latest"
