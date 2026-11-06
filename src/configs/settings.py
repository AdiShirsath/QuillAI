# settings file for all configs/keys


from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # === LLM ===
    openai_api_key: str = Field("..", env="OPENAI_API_KEY")
    # Which provider to use: (groq, mistral, gemini). Switch by changing this one line in .env.
    llm_provider: str = Field("groq", env="LLM_PROVIDER")
    groq_api_key: str = Field("", env="GROQ_API_KEY")
    groq_model: str = Field("openai/gpt-oss-20b", env="GROQ_MODEL")  # used for every role
    mistral_api_key: str = Field("", env="MISTRAL_API_KEY")
    mistral_model: str = Field("ministral-3b-2512", env="MISTRAL_MODEL")  # used for every role
    gemini_api_key: str = Field("", env="GEMINI_API_KEY")  # Google AI Studio key (aistudio.google.com)
    gemini_model: str = Field("gemini-3.5-flash-lite", env="GEMINI_MODEL")  # used for every role

    # OPTIONAL per-role overrides (empty = use the active provider's default model above).
    # Model names are provider-specific: set these only for the provider you are using.
    planner_model: str = Field("", env="PLANNER_MODEL")
    executor_model: str = Field("", env="EXECUTOR_MODEL")
    evaluator_model: str = Field("", env="EVALUATOR_MODEL")

    # === Code Execution Sandbox ===
    e2b_api_key: Optional[str] = Field(None, env="E2B_API_KEY")  # Cloud sandbox
    use_local_sandbox: bool = Field(True, env="USE_LOCAL_SANDBOX")  # Fallback to local

    # === Memory ===
    chroma_persist_dir: str = Field("./data/memory", env="CHROMA_PERSIST_DIR")
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    max_working_memory_steps: int = Field(50, env="MAX_WORKING_MEMORY")
    # Turn off on small/ephemeral hosts: skips ChromaDB and its embedding model (~100-200 MB RAM)
    enable_episodic_memory: bool = Field(True, env="ENABLE_EPISODIC_MEMORY")

    # === Guardrails ===
    # The static code policy is always on. The LLM judge is an optional extra layer (off by default).
    enable_llm_guardrail: bool = Field(False, env="ENABLE_LLM_GUARDRAIL")
    # NeMo Guardrails (needs `pip install nemoguardrails`). When on, it replaces the plain LLM judge.
    enable_nemo_guardrails: bool = Field(False, env="ENABLE_NEMO_GUARDRAILS")
    audit_log_dir: str = Field("./data/audit", env="AUDIT_LOG_DIR")

    # === API limits ===
    max_concurrent_tasks: int = Field(2, env="MAX_CONCURRENT_TASKS")
    max_upload_mb: int = Field(10, env="MAX_UPLOAD_MB")

    # === Agent Behavior ===
    max_iterations: int = Field(15, env="MAX_ITERATIONS")  # Max steps before stopping
    max_retries_per_step: int = Field(2, env="MAX_RETRIES_PER_STEP")  # Retries on error
    reasoning_effort: str = Field("low", env="REASONING_EFFORT")
    confidence_threshold: float = Field(0.7, env="CONFIDENCE_THRESHOLD")
    enable_self_correction: bool = Field(True, env="ENABLE_SELF_CORRECTION")

    # === Evaluation ===
    eval_results_dir: str = Field("./data/eval_results", env="EVAL_RESULTS_DIR")

    # === API ===
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
