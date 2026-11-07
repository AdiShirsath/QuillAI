import io
import time

import pytest
from fastapi.testclient import TestClient

CSV = "x,y\n1,3\n2,2\n3,1\n"


@pytest.fixture
def client(make_agent, monkeypatch):
    from src.server import main

    agent = make_agent()
    monkeypatch.setattr(main, "DataAnalysisAgent", lambda: agent)
    main._active_tasks = 0
    with TestClient(main.app) as c:
        yield c, main


def test_health(client):
    c, _ = client
    j = c.get("/health").json()
    assert j["status"] == "healthy" and j["agent_ready"] is True


def test_upload_and_sync_analyze(client):
    c, _ = client
    up = c.post("/upload", files={"file": ("d.csv", io.BytesIO(CSV.encode()), "text/csv")}).json()
    r = c.post("/analyze/sync", json={"goal": "Compute the mean of x", "file_key": up["file_key"]})
    assert r.status_code == 200 and r.json()["steps_taken"] == 2


def test_upload_rejects_bad_type_and_big_file(client, monkeypatch):
    c, main = client
    assert c.post("/upload", files={"file": ("a.exe", io.BytesIO(b"x"), "x/y")}).status_code == 400
    monkeypatch.setattr(main.settings, "max_upload_mb", 0)
    assert c.post("/upload", files={"file": ("d.csv", io.BytesIO(b"x,y\n1,2\n"), "text/csv")}).status_code == 413


def test_missing_file_key_404_both_endpoints(client):
    c, _ = client
    body = {"goal": "Compute the mean of x", "file_key": "nope"}
    assert c.post("/analyze", json=body).status_code == 404
    assert c.post("/analyze/sync", json=body).status_code == 404
    assert c.get("/health").json()["active_analyses"] == 0  # no leaked slot


def test_429_when_busy(client):
    c, main = client
    main._active_tasks = main.settings.max_concurrent_tasks
    r = c.post("/analyze", json={"goal": "Compute the mean of x", "sample_data": CSV})
    assert r.status_code == 429


def test_task_status_not_completed_early_and_slot_released(client):
    c, main = client
    tid = c.post("/analyze", json={"goal": "Compute the mean of x", "sample_data": CSV}).json()["task_id"]
    for _ in range(100):
        t = c.get(f"/task/{tid}").json()
        if t["status"] in ("completed", "error"):
            break
        time.sleep(0.05)
    assert t["status"] == "completed" and "executive_summary" in t["result"] or t["result"]
    assert c.get("/health").json()["active_analyses"] == 0


def test_ws_replay(client):
    c, _ = client
    tid = c.post("/analyze", json={"goal": "Compute the mean of x", "sample_data": CSV}).json()["task_id"]
    for _ in range(100):
        if c.get(f"/task/{tid}").json()["status"] == "completed":
            break
        time.sleep(0.05)
    with c.websocket_connect(f"/ws/{tid}") as ws:
        assert ws.receive_json()["event_type"] == "complete"


@pytest.mark.parametrize("path", ["/", "/docs", "/dashboard", "/ws-test", "/openapi.json"])
def test_ui_pages_are_served(client, path):
    c, _ = client
    r = c.get(path)
    assert r.status_code == 200 and len(r.text) > 200
