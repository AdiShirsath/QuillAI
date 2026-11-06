"""One place that decides which LLM provider QuillAI talks to (LLM_PROVIDER=groq|mistral|gemini)."""

from src.configs.settings import get_settings
from src.tools.token_tracker import tracker

SUPPORTED_PROVIDERS = ("groq", "mistral", "gemini")


def model_for(provider: str, role: str) -> str:
    """Role override (PLANNER_MODEL etc.) if set, otherwise the provider's default model."""
    s = get_settings()
    override = {"planner": s.planner_model, "executor": s.executor_model, "evaluator": s.evaluator_model}.get(role, "")
    default = {"groq": s.groq_model, "mistral": s.mistral_model, "gemini": s.gemini_model}[provider]
    return (override or "").strip() or default


def build_llm(role: str = "executor", temperature: float = 0.0):
    """Return a chat model for `role` ("planner" | "executor" | "evaluator") on the provider in LLM_PROVIDER."""
    s = get_settings()
    provider = (s.llm_provider or "groq").strip().lower()

    if provider == "mistral":
        if not s.mistral_api_key:
            raise ValueError("LLM_PROVIDER=mistral but MISTRAL_API_KEY is empty (see .env.example)")
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=model_for(provider, role),
            temperature=temperature,
            api_key=s.mistral_api_key,
            max_retries=6,
            callbacks=[tracker],
        )

    if provider == "gemini":
        if not s.gemini_api_key:
            raise ValueError("LLM_PROVIDER=gemini but GEMINI_API_KEY is empty (see .env.example)")
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {}
        if "2.5-flash" in model_for(provider, role):  # these models "think" by default; 0 saves tokens and time
            kwargs["thinking_budget"] = 0
        return ChatGoogleGenerativeAI(
            model=model_for(provider, role),
            temperature=temperature,
            google_api_key=s.gemini_api_key,
            max_retries=6,
            callbacks=[tracker],
            **kwargs,
        )

    if provider == "groq":
        if not s.groq_api_key:
            raise ValueError("LLM_PROVIDER=groq but GROQ_API_KEY is empty (see .env.example)")
        from langchain_groq import ChatGroq

        model = model_for(provider, role)
        kwargs = {}
        if "gpt-oss" in model:  # only reasoning models accept this; others would reject the request
            kwargs["reasoning_effort"] = "low"
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=s.groq_api_key,
            max_retries=6,
            callbacks=[tracker],
            **kwargs,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(SUPPORTED_PROVIDERS)}")


def active_model_name() -> str:
    """Provider and the model used for each role (shown in /health)."""
    s = get_settings()
    provider = (s.llm_provider or "groq").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        return f"{provider}:?"
    models = {model_for(provider, r) for r in ("planner", "executor", "evaluator")}
    return f"{provider}:{'/'.join(sorted(models))}"
