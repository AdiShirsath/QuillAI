# settings file for all configs/keys


from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):

    # === LLM ===
    openai_api_key: str = Field("..", env="OPENAI_API_KEY")
    groq_api_key: str = Field("", env="GROQ_API_KEY" )
    planner_model: str = Field("llama-3.1-8b-instant", env="PLANNER_MODEL")        # Smart, plans tasks
    executor_model: str = Field("llama3-70b-8192", env="EXECUTOR_MODEL")      # Executes each step
    evaluator_model: str = Field("llama3-70b-8192", env="EVALUATOR_MODEL")    # Judges output quality

    # === Code Execution Sandbox ===
    e2b_api_key: Optional[str] = Field(None, env="E2B_API_KEY")      # Cloud sandbox
    use_local_sandbox: bool = Field(True, env="USE_LOCAL_SANDBOX")   # Fallback to local

    # === Memory ===
    chroma_persist_dir: str = Field("./data/memory", env="CHROMA_PERSIST_DIR")
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    max_working_memory_steps: int = Field(50, env="MAX_WORKING_MEMORY")

    # === Agent Behavior ===
    max_iterations: int = Field(15, env="MAX_ITERATIONS")            # Max steps before stopping
    max_retries_per_step: int = Field(3, env="MAX_RETRIES_PER_STEP") # Retries on error
    confidence_threshold: float = Field(0.7, env="CONFIDENCE_THRESHOLD")
    enable_self_correction: bool = Field(True, env="ENABLE_SELF_CORRECTION")

    # === Evaluation ===
    eval_results_dir: str = Field("./data/eval_results", env="EVAL_RESULTS_DIR")

    # === API ===
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # === Monitoring ===
    langfuse_public_key: Optional[str] = Field(None, env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(None, env="LANGFUSE_SECRET_KEY")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
