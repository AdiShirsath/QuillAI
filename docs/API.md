# DataWright — API Reference

Base URL: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`


## The standard flow

Every analysis follows this 3-step flow:

```
1. POST /upload      →  file_key
2. POST /analyze     →  task_id
3. GET  /task/{id}   →  result
```

Or watch it live:
```
1. POST /upload      →  file_key
2. POST /analyze     →  task_id + websocket_url
3. WS   /ws/{id}     →  live events as agent works
```

---

## Endpoints

---

### GET /health

Check if the server, agent, and Redis are all running correctly. Use this first to confirm your setup is working.

**Request**
```bash
curl http://localhost:8000/health
```

**Response**
```json
{
  "status": "healthy",
  "agent_ready": true,
  "redis": "healthy",
  "redis_info": {
    "used_memory": "1.2M",
    "total_keys": 4,
    "active_tasks": 1
  },
  "active_websockets": 0
}
```

**What to check**
- `agent_ready` must be `true` before calling any other endpoint
- `redis` should be `"healthy"` — if `"unavailable"`, tasks will still work but won't survive server restarts
- If both are good, you're ready to use the API

---

### POST /upload

Upload a data file. Returns a `file_key` you pass to `/analyze`.

The file is saved to disk and the path is stored in Redis for 1 hour. After 1 hour the key expires and you'll need to re-upload.

**Supported formats:** `.csv` `.xlsx` `.xls` `.json` `.parquet`

**Request**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@your_data.csv"
```

**Response**
```json
{
  "file_key": "a3f2b1c4",
  "filename": "telco_churn.csv",
  "expires_in": "1 hour",
  "preview": {
    "rows": 7043,
    "columns": ["customerID", "tenure", "MonthlyCharges", "Contract", "Churn"],
    "dtypes": {
      "customerID": "object",
      "tenure": "int64",
      "MonthlyCharges": "float64",
      "Contract": "object",
      "Churn": "object"
    },
    "sample": [
      {"customerID": "7590-VHVEG", "tenure": 1, "MonthlyCharges": 29.85}
    ]
  }
}
```

**Save the `file_key`** — you need it in the next step.

**Errors**
| Status | Reason |
|---|---|
| `400` | File type not supported |
| `503` | Agent not ready yet |

---

### POST /analyze

Start an analysis task. Returns immediately with a `task_id` — the agent runs in the background.

Use this for any analysis that takes more than a few seconds. Connect to `/ws/{task_id}` to watch it live, or poll `/task/{task_id}` to get the result when done.

**Request body**
```json
{
  "goal": "Find the top 3 factors driving customer churn and build a logistic regression model",
  "file_key": "a3f2b1c4"
}
```

| Field | Required | Description |
|---|---|---|
| `goal` | Yes (min 10 chars) | What you want to know — natural language |
| `file_key` | One of these | Key from `/upload` |
| `sample_data` | One of these | Paste raw CSV text directly (for quick tests) |

**Request**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Find the top 3 factors driving customer churn and build a logistic regression model",
    "file_key": "a3f2b1c4"
  }'
```

**Response**
```json
{
  "task_id": "9e2d7f1a",
  "status": "queued",
  "websocket_url": "/ws/9e2d7f1a",
  "tip": "Connect to ws://localhost:8000/ws/9e2d7f1a for live updates"
}
```

**Save the `task_id`** — use it to check results or connect WebSocket.

**Using `sample_data` instead of a file (quick testing)**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Find patterns in this data",
    "sample_data": "name,age,score\nAlice,30,85\nBob,25,92\nCarol,35,78"
  }'
```

**Errors**
| Status | Reason |
|---|---|
| `404` | `file_key` not found or expired — re-upload |
| `503` | Agent not ready yet |

---

### GET /task/{task_id}

Get the current status and result of a task. Poll this every few seconds, or use the WebSocket for live updates.

Task data is stored in Redis and expires after 2 hours.

**Request**
```bash
curl http://localhost:8000/task/9e2d7f1a
```

**Response — while running**
```json
{
  "task_id": "9e2d7f1a",
  "goal": "Find the top 3 factors driving customer churn",
  "status": "running",
  "last_event": {
    "event_type": "step_complete",
    "data": {
      "title": "Calculate churn rates by segment",
      "status": "success",
      "confidence_score": 0.89
    }
  }
}
```

**Response — when completed**
```json
{
  "task_id": "9e2d7f1a",
  "goal": "Find the top 3 factors driving customer churn",
  "status": "completed",
  "result": {
    "executive_summary": "Month-to-month contracts and high monthly charges are the strongest predictors of churn...",
    "key_findings": [
      "Month-to-month customers churn at 42% vs 3% for two-year contracts",
      "Customers with tenure under 12 months churn at 47%",
      "Logistic regression achieved 81% accuracy on the test set"
    ],
    "detailed_analysis": "...",
    "methodology": "The agent executed 6 steps...",
    "limitations": [],
    "confidence_score": 0.87,
    "steps_taken": 6,
    "self_corrections": 1,
    "total_latency_ms": 34200,
    "figure_count": 3,
    "figures": ["<base64 PNG>", "<base64 PNG>", "<base64 PNG>"]
  }
}
```

**Status values**
| Status | Meaning |
|---|---|
| `queued` | Task created, agent starting |
| `running` | Agent is actively working |
| `completed` | Done — `result` field has everything |
| `error` | Something failed — `error` field has details |

**Polling script**
```python
import httpx, time

task_id = "9e2d7f1a"
while True:
    r = httpx.get(f"http://localhost:8000/task/{task_id}")
    task = r.json()
    print(f"Status: {task['status']}")
    if task["status"] == "completed":
        print(task["result"]["executive_summary"])
        break
    if task["status"] == "error":
        print(f"Error: {task.get('error')}")
        break
    time.sleep(3)
```

**Errors**
| Status | Reason |
|---|---|
| `404` | Task not found or expired (2 hour TTL) |

---

### WS /ws/{task_id}

WebSocket endpoint — receive every agent event the moment it happens. Connect right after calling `/analyze`.

If the task already completed before you connect, the final result is replayed immediately from Redis.

**Connect**
```javascript
// Browser
const ws = new WebSocket("ws://localhost:8000/ws/9e2d7f1a")
ws.onmessage = (e) => console.log(JSON.parse(e.data))
```

```python
# Python
import asyncio, websockets, json

async def watch():
    async with websockets.connect("ws://localhost:8000/ws/9e2d7f1a") as ws:
        async for msg in ws:
            event = json.loads(msg)
            print(event["event_type"], event.get("data", {}).get("title", ""))

asyncio.run(watch())
```

**Event types received**

| event_type | When | Key fields in `data` |
|---|---|---|
| `planning_start` | Agent begins planning | — |
| `planning_complete` | Plan is ready | `steps[]`, `complexity` |
| `step_start` | A step begins | `title`, `step_number`, `total_steps` |
| `step_complete` | A step finishes | `status`, `confidence_score`, `key_findings[]`, `figures[]` |
| `self_correction` | Agent fixed its own code | `explanation` |
| `replan_start` | Agent is revising the plan | `reason` |
| `replan_complete` | New plan is ready | `new_steps` |
| `complete` | All done | Full result (same as `/task/{id}`) |
| `error` | Something failed | `message` |
| `heartbeat` | Keep-alive ping every 30s | — |

**Example events**
```json
{"event_type": "planning_complete", "data": {
  "steps": [
    {"step_id": "s1", "step_type": "think", "title": "Understand the goal"},
    {"step_id": "s2", "step_type": "code",  "title": "Explore the dataset"},
    {"step_id": "s3", "step_type": "code",  "title": "Calculate churn rates"},
    {"step_id": "s4", "step_type": "code",  "title": "Build logistic regression"},
    {"step_id": "s5", "step_type": "summarize", "title": "Write final report"}
  ],
  "complexity": "medium"
}}

{"event_type": "step_complete", "data": {
  "title": "Calculate churn rates",
  "status": "success",
  "confidence_score": 0.89,
  "key_findings": ["Month-to-month churn rate is 42%"],
  "has_figures": true,
  "figures": ["<base64 PNG>"],
  "latency_ms": 4200
}}

{"event_type": "self_correction", "data": {
  "explanation": "Fixed KeyError on 'Contract_Type' — correct column is 'Contract'"
}}

{"event_type": "complete", "data": {
  "executive_summary": "...",
  "key_findings": ["...", "..."],
  "steps_taken": 5,
  "self_corrections": 1,
  "confidence_score": 0.87
}}
```

---

### POST /analyze/sync

Simpler version — runs the agent and waits for the full result before returning. No WebSocket or polling needed.

Use this for scripts, testing, or any task under ~60 seconds. For longer tasks use `/analyze` + WebSocket instead to avoid HTTP timeout.

**Request**
```bash
curl -X POST http://localhost:8000/analyze/sync \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Describe the dataset and find the 3 most interesting patterns",
    "file_key": "a3f2b1c4"
  }' | python3 -m json.tool
```

**Response** — same fields as the `result` object inside `/task/{id}`
```json
{
  "task_id": "b1c4d5e6",
  "goal": "Describe the dataset and find the 3 most interesting patterns",
  "executive_summary": "The dataset contains 7,043 telecom customers...",
  "key_findings": [
    "26.5% overall churn rate",
    "Fiber optic customers have the highest monthly charges",
    "Senior citizens churn at 41% vs 24% for non-seniors"
  ],
  "detailed_analysis": "...",
  "methodology": "The agent executed 5 steps (4 successful)...",
  "limitations": [],
  "confidence_score": 0.84,
  "steps_taken": 5,
  "self_corrections": 0,
  "total_latency_ms": 28400.0,
  "figure_count": 2,
  "figures": ["<base64 PNG>", "<base64 PNG>"]
}
```

**Decode figures to PNG files (Python)**
```python
import base64, httpx

r = httpx.post("http://localhost:8000/analyze/sync", json={
    "goal": "Find churn patterns",
    "file_key": "a3f2b1c4"
}, timeout=120)

result = r.json()
for i, fig in enumerate(result.get("figures", [])):
    with open(f"figure_{i+1}.png", "wb") as f:
        f.write(base64.b64decode(fig))
    print(f"Saved figure_{i+1}.png")
```

**Errors**
| Status | Reason |
|---|---|
| `404` | `file_key` not found or expired |
| `503` | Agent not ready yet |

---

### GET /memory/stats

Inspect what is currently stored in the agent's two memory systems — useful for debugging and demos.

**Request**
```bash
curl http://localhost:8000/memory/stats
```

**Response**
```json
{
  "episodic_memory": {
    "total_entries": 12,
    "persist_dir": "./data/memory",
    "enabled": true
  },
  "redis": {
    "total_keys": 8,
    "active_tasks": 2,
    "working_memory_entries": 5,
    "uploaded_files": 1
  }
}
```

| Field | What it means |
|---|---|
| `episodic_memory.total_entries` | Past task summaries + code patterns stored in ChromaDB |
| `redis.active_tasks` | Number of `task:*` keys currently in Redis |
| `redis.working_memory_entries` | Number of `agent:*` scratchpad keys (per-task findings, step outputs) |
| `redis.uploaded_files` | Number of `file:*` keys (uploaded files not yet expired) |

---

## Full workflow example

```python
import httpx, time, base64, json

SERVER = "http://localhost:8000"

# 1. Check health
r = httpx.get(f"{SERVER}/health")
assert r.json()["agent_ready"], "Agent not ready"
print("Server healthy")

# 2. Upload file
with open("telco_churn.csv", "rb") as f:
    r = httpx.post(f"{SERVER}/upload", files={"file": f})
file_key = r.json()["file_key"]
print(f"Uploaded → file_key: {file_key}")

# 3. Start analysis
r = httpx.post(f"{SERVER}/analyze", json={
    "goal": "Find the top 3 churn drivers and build a logistic regression model",
    "file_key": file_key,
})
task_id = r.json()["task_id"]
print(f"Task started → task_id: {task_id}")

# 4. Poll until done
while True:
    r = httpx.get(f"{SERVER}/task/{task_id}")
    task = r.json()
    print(f"  Status: {task['status']}")
    if task["status"] == "completed":
        break
    if task["status"] == "error":
        print(f"Error: {task.get('error')}")
        raise SystemExit(1)
    time.sleep(3)

# 5. Print results
result = task["result"]
print(f"\nSummary: {result['executive_summary']}")
print(f"\nKey findings:")
for i, f in enumerate(result["key_findings"], 1):
    print(f"  {i}. {f}")
print(f"\nConfidence: {result['confidence_score']:.0%}")
print(f"Steps: {result['steps_taken']}  Corrections: {result['self_corrections']}")

# 6. Save figures
for i, fig in enumerate(result.get("figures", [])):
    with open(f"figure_{i+1}.png", "wb") as f:
        f.write(base64.b64decode(fig))
print(f"\nSaved {len(result.get('figures', []))} figures")
```

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `503 Agent not ready` | Server just started | Wait 5 seconds and retry |
| `404 file_key not found` | Key expired (1 hour TTL) or Redis was restarted | Re-upload the file |
| `404 Task not found` | Task expired (2 hour TTL) | Re-run the analysis |
| `400 File type not supported` | Wrong file format | Use `.csv`, `.xlsx`, `.json` |
| `WebSocket connection refused` | Wrong URL or task_id | Check task_id from `/analyze` response |
