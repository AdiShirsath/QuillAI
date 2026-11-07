import json

from src.tools import guardrails as g


def test_static_policy_blocks_and_allows():
    assert not g.check_code_policy("import os").allowed
    assert not g.check_code_policy("x = ().__class__").allowed
    assert g.check_code_policy("import pandas as pd\nprint(1)").allowed


def test_judge_fails_closed_on_llm_error():
    class Boom:
        def invoke(self, *_a, **_k):
            raise RuntimeError("down")

    r = g.check_code("print(1)", "s1", Boom(), use_llm_judge=True)
    assert not r.allowed


def test_judge_cannot_override_static_block():
    class Yes:
        def invoke(self, *_a, **_k):
            import types

            return types.SimpleNamespace(content=json.dumps({"safe": True, "reason": "fine"}))

    r = g.check_code("import os", "s1", Yes(), use_llm_judge=True)
    assert not r.allowed


def test_audit_log_hashes_and_redacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path))
    from src.configs.settings import get_settings

    get_settings.cache_clear()
    g.check_code("import os  # api_key=sk-SECRET123456", "s9", None, use_llm_judge=False)
    files = list(tmp_path.glob("*.jsonl"))
    assert files, "audit file not written"
    text = files[0].read_text()
    rec = json.loads(text.splitlines()[-1])
    assert len(rec.get("code_sha256", "")) == 64
    assert "SECRET123456" not in text
    get_settings.cache_clear()
