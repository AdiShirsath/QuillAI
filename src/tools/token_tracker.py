from langchain_core.callbacks import BaseCallbackHandler


class TokenTracker(BaseCallbackHandler):
    """Counts LLM calls and tokens. Attach with ChatGroq(callbacks=[tracker])."""

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)

    def summary(self) -> str:
        total = self.input_tokens + self.output_tokens
        return f"{self.calls} LLM calls, {self.input_tokens} in + {self.output_tokens} out = {total} tokens"


tracker = TokenTracker()  # one shared counter for the process
