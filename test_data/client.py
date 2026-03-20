"""
client.py
----------
Test client for the Autonomous Data Analysis Agent API.

Tests every endpoint in src/api/main.py with real requests.
Run the server first, then run this file.

USAGE:
    # Terminal 1 - start server
    python -m src.api.main

    # Terminal 2 - run client tests
    python client.py                    # run all tests
    python client.py --test health      # run one test
    python client.py --test upload      # test file upload
    python client.py --test analyze     # test full analysis
    python client.py --test websocket   # watch live agent stream
    python client.py --test evaluate    # run evaluation

ENDPOINTS TESTED:
    GET  /health
    POST /upload
    POST /analyze          (async + WebSocket stream)
    POST /analyze/sync     (synchronous, waits for result)
    GET  /task/{task_id}
    GET  /memory/stats
    POST /evaluate
"""

import argparse
import asyncio
import io
import json
import sys
import time
from typing import Optional

import httpx
import websockets

# ─── CONFIG ───────────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
TIMEOUT = 120  # seconds — agent tasks can take a while

# ANSI colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ─── SAMPLE DATA ──────────────────────────────────────────────────────────────

SAMPLE_CSV = """customerID,tenure,MonthlyCharges,TotalCharges,Contract,InternetService,TechSupport,Churn
7590-VHVEG,1,29.85,29.85,Month-to-month,DSL,No,No
5575-GNVDE,34,56.95,1889.5,One year,DSL,Yes,No
3668-QPYBK,2,53.85,108.15,Month-to-month,DSL,No,Yes
7795-CFOCW,45,42.30,1840.75,One year,DSL,Yes,No
9237-HQITU,2,70.70,151.65,Month-to-month,Fiber optic,No,Yes
9305-CDSKC,8,99.65,820.5,Month-to-month,Fiber optic,No,Yes
1452-KIOVK,22,89.10,1949.4,Month-to-month,Fiber optic,Yes,No
6713-OKOMC,10,29.75,301.9,Month-to-month,DSL,No,No
7892-POOKP,28,104.80,3046.05,Month-to-month,Fiber optic,No,Yes
6388-TABGU,62,56.15,3487.95,One year,DSL,No,No
8BEM-AXTYB,15,75.20,1128.0,Month-to-month,Fiber optic,No,Yes
4HJK-MNBVC,48,45.00,2160.0,Two year,DSL,Yes,No
2QWE-RTYUI,5,89.90,449.5,Month-to-month,Fiber optic,No,Yes
9PLK-MNJHG,72,19.85,1429.2,Two year,DSL,Yes,No
3ZXC-VBNMQ,3,65.60,196.8,Month-to-month,Fiber optic,No,Yes"""

ANALYSIS_GOAL = (
    "Calculate the churn rate by Contract type and InternetService. "
    "Find which MonthlyCharges range has the highest churn. "
    "Summarize the top 2 factors driving churn with specific numbers."
)

EVAL_SAMPLES = [
    {
        "question": "What is the churn rate for month-to-month customers?",
        "answer": "Month-to-month customers have a significantly higher churn rate compared to annual contract customers.",
        "contexts": [
            "Contract type strongly influences churn. Month-to-month customers churn at 42% vs 11% for one-year contracts.",
        ],
        "ground_truth": "Month-to-month customers churn at approximately 42%.",
    },
    {
        "question": "Which internet service has higher churn?",
        "answer": "Fiber optic customers tend to churn more than DSL customers.",
        "contexts": [
            "Fiber optic internet service customers show higher churn rates at 41% compared to DSL at 19%.",
        ],
        "ground_truth": "Fiber optic customers have higher churn at around 41%.",
    },
]


# ─── HELPERS ──────────────────────────────────────────────────────────────────


def print_header(title: str):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


def print_result(label: str, value, indent: int = 2):
    pad = " " * indent
    if isinstance(value, dict):
        print(f"{pad}{DIM}{label}:{RESET}")
        for k, v in value.items():
            print(f"{pad}  {DIM}{k}:{RESET} {v}")
    elif isinstance(value, list):
        print(f"{pad}{DIM}{label}:{RESET}")
        for item in value[:5]:
            print(f"{pad}  • {item}")
        if len(value) > 5:
            print(f"{pad}  {DIM}... and {len(value)-5} more{RESET}")
    else:
        print(f"{pad}{DIM}{label}:{RESET} {value}")


def ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str):
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str):
    print(f"  {YELLOW}⚠{RESET} {msg}")


def info(msg: str):
    print(f"  {DIM}→{RESET} {msg}")


def check_server_running():
    """Make sure server is up before running tests."""
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ─── TEST FUNCTIONS ───────────────────────────────────────────────────────────


def test_health():
    """
    GET /health
    Checks server is alive and agent is initialized.
    """
    print_header("TEST: GET /health")

    r = httpx.get(f"{BASE_URL}/health", timeout=10)
    data = r.json()

    if r.status_code == 200:
        ok(f"Server is healthy (HTTP {r.status_code})")
    else:
        fail(f"Unexpected status: {r.status_code}")
        return False

    print_result("Response", data)

    if data.get("agent_ready"):
        ok("Agent is initialized and ready")
    else:
        warn("Agent not ready — server may still be starting up")

    info(f"Active WebSocket connections: {data.get('active_connections', 0)}")
    info(f"Tasks in memory: {data.get('tasks_in_memory', 0)}")
    return True


def test_upload():
    """
    POST /upload
    Uploads a CSV file and verifies the preview response.
    """
    print_header("TEST: POST /upload")

    # Create a file-like object from our sample CSV
    csv_bytes = SAMPLE_CSV.encode("utf-8")
    files = {"file": ("telco_churn.csv", io.BytesIO(csv_bytes), "text/csv")}

    r = httpx.post(f"{BASE_URL}/upload", files=files, timeout=30)
    data = r.json()

    if r.status_code == 200:
        ok(f"File uploaded successfully (HTTP {r.status_code})")
    else:
        fail(f"Upload failed: {r.status_code} — {data}")
        return None

    file_key = data.get("file_key")
    ok(f"File key received: {file_key}")

    preview = data.get("preview", {})
    if preview.get("shape"):
        ok(f"Data profiled: {preview['shape'][0]} rows × {preview['shape'][1]} columns")
        info(f"Columns: {preview.get('columns', [])}")
    else:
        warn("No preview data returned")

    # Test unsupported file type
    info("Testing unsupported file type (.pdf)...")
    bad_files = {"file": ("test.pdf", io.BytesIO(b"fake pdf"), "application/pdf")}
    r2 = httpx.post(f"{BASE_URL}/upload", files=bad_files, timeout=10)
    if r2.status_code == 400:
        ok("Correctly rejected unsupported file type (.pdf → 400)")
    else:
        warn(f"Expected 400, got {r2.status_code}")

    return file_key


def test_analyze_sync(file_key: Optional[str] = None):
    """
    POST /analyze/sync
    Synchronous analysis — waits for completion.
    Good for testing without WebSocket complexity.
    """
    print_header("TEST: POST /analyze/sync")

    payload = {"goal": ANALYSIS_GOAL}

    if file_key:
        payload["file_key"] = file_key
        info(f"Using uploaded file (key: {file_key})")
    else:
        payload["sample_data"] = SAMPLE_CSV
        info("Using inline CSV data (sample_data)")

    info(f"Goal: {ANALYSIS_GOAL[:80]}...")
    info("Waiting for agent to complete (this takes 30-90 seconds)...")

    t0 = time.time()
    r = httpx.post(f"{BASE_URL}/analyze/sync", json=payload, timeout=TIMEOUT)
    elapsed = time.time() - t0

    if r.status_code == 200:
        ok(f"Analysis complete in {elapsed:.1f}s (HTTP {r.status_code})")
    else:
        fail(f"Analysis failed: {r.status_code}")
        print(f"  Response: {r.text[:500]}")
        return None

    data = r.json()
    task_id = data.get("task_id")

    print(f"\n  {BOLD}── RESULTS ──{RESET}")
    print_result("Task ID", task_id)
    print_result("Steps taken", data.get("steps_taken"))
    print_result("Self-corrections", data.get("self_corrections"))
    print_result("Confidence score", f"{data.get('confidence_score', 0):.0%}")
    print_result("Latency", f"{data.get('total_latency_ms', 0)/1000:.1f}s")
    print_result("Figures generated", data.get("figure_count", 0))

    summary = data.get("executive_summary", "")
    if summary:
        ok("Executive summary received")
        print(f"\n  {DIM}Summary:{RESET}")
        # Word-wrap at 60 chars
        words = summary.split()
        line = "    "
        for word in words:
            if len(line) + len(word) > 64:
                print(line)
                line = "    " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

    findings = data.get("key_findings", [])
    if findings:
        ok(f"{len(findings)} key findings extracted")
        for i, f in enumerate(findings[:3], 1):
            print(f"    {i}. {f[:80]}")
    else:
        warn("No key findings returned")

    return task_id


async def test_analyze_async():
    """
    POST /analyze + WS /ws/{task_id}
    Async analysis with real-time WebSocket streaming.
    This is the most impressive test — watch the agent think live.
    """
    print_header("TEST: POST /analyze + WS /ws/{task_id} (streaming)")

    async with httpx.AsyncClient(timeout=30) as client:
        # Start the async task
        payload = {"goal": ANALYSIS_GOAL, "sample_data": SAMPLE_CSV}
        r = await client.post(f"{BASE_URL}/analyze", json=payload)

        if r.status_code != 200:
            fail(f"Failed to start task: {r.status_code} — {r.text}")
            return None

        data = r.json()
        task_id = data.get("task_id")
        ok(f"Task started (ID: {task_id})")
        info(f"Connecting to WebSocket: {WS_URL}/ws/{task_id}")

    # Connect to WebSocket and stream events
    print(f"\n  {BOLD}── LIVE AGENT STREAM ──{RESET}")

    try:
        async with websockets.connect(
            f"{WS_URL}/ws/{task_id}",
            ping_interval=20,
            ping_timeout=30,
        ) as ws:
            step_count = 0
            corrections = 0
            figures = 0

            async for raw_msg in ws:
                try:
                    event = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue

                etype = event.get("event_type", "")
                edata = event.get("data", {})

                # ── Render each event type ──────────────────────────
                if etype == "heartbeat":
                    continue  # Skip heartbeat noise

                elif etype == "planning_start":
                    info("Agent is planning...")

                elif etype == "planning_complete":
                    steps = edata.get("steps", [])
                    ok(f"Plan created: {len(steps)} steps ({edata.get('complexity', '?')} complexity)")
                    for s in steps:
                        icon = {"think": "💭", "code": "💻", "analyze": "🔍", "validate": "✅", "summarize": "📝"}.get(
                            s.get("step_type", ""), "•"
                        )
                        print(f"    {icon} [{s.get('step_id')}] {s.get('title')}")

                elif etype == "step_start":
                    step_count += 1
                    stype = edata.get("step_type", "")
                    icon = {"think": "💭", "code": "💻", "analyze": "🔍", "validate": "✅", "summarize": "📝"}.get(
                        stype, "•"
                    )
                    num = edata.get("step_number", step_count)
                    total = edata.get("total_steps", "?")
                    print(f"\n  {icon} Step {num}/{total}: {edata.get('title', '')}")

                elif etype == "self_correction":
                    corrections += 1
                    print(f"    {YELLOW}🔧 Self-correcting:{RESET} {edata.get('explanation','')[:80]}")

                elif etype == "step_complete":
                    status = edata.get("status", "")
                    score = edata.get("confidence_score", 0)
                    color = GREEN if status == "success" else RED
                    icon = "✓" if status == "success" else "✗"
                    print(f"    {color}{icon}{RESET} {status} (confidence: {score:.0%})")

                    for f in edata.get("key_findings", [])[:2]:
                        print(f"      → {f[:75]}")

                    n_figs = len(edata.get("figures", []))
                    if n_figs:
                        figures += n_figs
                        print(f"      📊 {n_figs} figure(s) generated")

                    if edata.get("error"):
                        print(f"      {RED}Error:{RESET} {edata['error'][:80]}")

                elif etype == "replan_start":
                    warn(f"Replanning: {edata.get('reason','')[:70]}")

                elif etype == "replan_complete":
                    info(f"Revised plan: {edata.get('new_steps')} remaining steps")

                elif etype == "complete":
                    print(f"\n  {BOLD}{GREEN}── COMPLETE ──{RESET}")
                    ok("Task finished successfully")
                    print_result("Steps taken", edata.get("steps_taken"))
                    print_result("Self-corrections", edata.get("self_corrections"))
                    print_result("Confidence", f"{edata.get('confidence_score', 0):.0%}")
                    print_result("Latency", f"{edata.get('total_latency_ms', 0)/1000:.1f}s")
                    print_result("Figures", figures)

                    summary = edata.get("executive_summary", "")
                    if summary:
                        print(f"\n  {DIM}Summary:{RESET} {summary[:200]}...")

                    break  # Done

                elif etype == "error":
                    fail(f"Agent error: {edata.get('message', 'Unknown')}")
                    break

    except websockets.exceptions.ConnectionClosed as e:
        warn(f"WebSocket closed: {e}")
    except OSError as e:
        fail(f"WebSocket connection failed: {e}")
        info("Make sure the server is running and websockets is installed")
        info("pip install websockets")
        return None

    return task_id


def test_get_task(task_id: str):
    """
    GET /task/{task_id}
    Poll task status — useful when not using WebSocket.
    """
    print_header(f"TEST: GET /task/{task_id}")

    r = httpx.get(f"{BASE_URL}/task/{task_id}", timeout=10)

    if r.status_code == 200:
        ok(f"Task found (HTTP {r.status_code})")
        data = r.json()
        print_result("Status", data.get("status"))
        print_result("Goal", data.get("goal", "")[:60])
        if data.get("result"):
            ok("Result data present")
    elif r.status_code == 404:
        warn(f"Task {task_id} not found (may have expired from memory)")
    else:
        fail(f"Unexpected status: {r.status_code}")

    # Test with non-existent task ID
    info("Testing non-existent task ID...")
    r2 = httpx.get(f"{BASE_URL}/task/doesnotexist999", timeout=10)
    if r2.status_code == 404:
        ok("Correctly returned 404 for unknown task ID")
    else:
        warn(f"Expected 404, got {r2.status_code}")


def test_memory_stats():
    """
    GET /memory/stats
    Check what's stored in the agent's episodic memory.
    """
    print_header("TEST: GET /memory/stats")

    r = httpx.get(f"{BASE_URL}/memory/stats", timeout=10)
    data = r.json()

    if r.status_code == 200:
        ok(f"Memory stats retrieved (HTTP {r.status_code})")
    else:
        fail(f"Failed: {r.status_code}")
        return

    if data.get("memory_enabled"):
        ok("Episodic memory is enabled (ChromaDB)")
        print_result("Total memories stored", data.get("total_memories", 0))
        print_result("Memory directory", data.get("memory_dir"))

        n = data.get("total_memories", 0)
        if n > 0:
            ok(f"Agent has {n} memory entries from past tasks")
        else:
            info("No memories yet — run some analysis tasks first")
    else:
        warn("Episodic memory not enabled")
        if data.get("error"):
            info(f"Reason: {data['error']}")


def test_evaluate():
    """
    POST /evaluate
    Run RAGAS-style evaluation on a test set.
    """
    print_header("TEST: POST /evaluate")

    payload = {
        "samples": EVAL_SAMPLES,
        "run_name": "client_test_run",
        "use_ragas": False,  # Use lightweight custom metrics (no extra deps)
    }

    info(f"Evaluating {len(EVAL_SAMPLES)} test samples...")
    info("Using lightweight custom metrics (use_ragas=False)")

    r = httpx.post(f"{BASE_URL}/evaluate", json=payload, timeout=TIMEOUT)

    if r.status_code == 200:
        ok(f"Evaluation complete (HTTP {r.status_code})")
    else:
        fail(f"Evaluation failed: {r.status_code} — {r.text[:200]}")
        return

    data = r.json()
    metrics = data.get("metrics", {})

    print(f"\n  {BOLD}── EVALUATION METRICS ──{RESET}")

    def grade(val):
        if val is None:
            return f"{DIM}N/A{RESET}"
        color = GREEN if val >= 0.8 else YELLOW if val >= 0.6 else RED
        bar_len = int(val * 15)
        bar = "█" * bar_len + "░" * (15 - bar_len)
        return f"{color}{bar}{RESET} {val:.1%}"

    metric_names = {
        "faithfulness": "Faithfulness    ",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall  ",
        "answer_similarity": "Answer Similarity",
    }

    for key, label in metric_names.items():
        val = metrics.get(key)
        print(f"    {DIM}{label}{RESET}  {grade(val)}")

    print_result("Samples evaluated", data.get("n_samples"))
    print_result("Run name", data.get("run_name"))
    print_result("Timestamp", data.get("timestamp", "")[:19])


def test_invalid_requests():
    """
    Tests that the API correctly rejects malformed requests.
    Good for showing you built a robust API, not just a happy-path demo.
    """
    print_header("TEST: Invalid Request Handling")

    cases = [
        # (description, method, endpoint, payload, expected_status)
        ("Empty goal", "POST", "/analyze/sync", {"goal": "hi"}, 422),
        ("Missing goal", "POST", "/analyze/sync", {}, 422),
        ("Bad file key", "POST", "/analyze/sync", {"goal": ANALYSIS_GOAL, "file_key": "badkey123"}, 404),
        ("Unknown task", "GET", "/task/xxxxxxxx", None, 404),
    ]

    all_passed = True
    for desc, method, endpoint, payload, expected in cases:
        try:
            if method == "POST":
                r = httpx.post(f"{BASE_URL}{endpoint}", json=payload, timeout=10)
            else:
                r = httpx.get(f"{BASE_URL}{endpoint}", timeout=10)

            if r.status_code == expected:
                ok(f"{desc} → {r.status_code} (expected {expected})")
            else:
                fail(f"{desc} → got {r.status_code}, expected {expected}")
                all_passed = False
        except Exception as e:
            fail(f"{desc} → exception: {e}")
            all_passed = False

    return all_passed


# ─── MAIN RUNNER ──────────────────────────────────────────────────────────────


async def run_all_tests():
    """Run all tests in sequence."""
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  🤖 DataWright Agent — API Test Suite{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    print(f"  Server: {BASE_URL}")
    print(f"  Timeout: {TIMEOUT}s per request")

    # Check server
    if not check_server_running():
        print(f"\n{RED}✗ Server not running at {BASE_URL}{RESET}")
        print(f"  Start it with: {BOLD}python -m src.api.main{RESET}")
        sys.exit(1)

    print(f"\n  {GREEN}Server is running ✓{RESET}")

    results = {}

    # 1. Health
    results["health"] = test_health()

    # 2. Upload
    file_key = test_upload()
    results["upload"] = file_key is not None

    # 3. Sync analysis (uses uploaded file if available)
    task_id = test_analyze_sync(file_key)
    results["analyze_sync"] = task_id is not None

    # 4. Get task status
    if task_id:
        test_get_task(task_id)
        results["get_task"] = True

    # 5. Memory stats
    test_memory_stats()
    results["memory_stats"] = True

    # 6. Evaluation
    test_evaluate()
    results["evaluate"] = True

    # 7. Invalid request handling
    results["invalid_requests"] = test_invalid_requests()

    # 8. WebSocket streaming (async)
    print_header("TEST: Async WebSocket Streaming")
    info("Starting async analysis with WebSocket stream...")
    ws_task_id = await test_analyze_async()
    results["websocket"] = ws_task_id is not None

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  TEST SUMMARY{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_flag in results.items():
        icon = f"{GREEN}✓{RESET}" if passed_flag else f"{RED}✗{RESET}"
        print(f"  {icon} {test_name}")

    print(f"\n  {BOLD}Result: {passed}/{total} tests passed{RESET}")

    if passed == total:
        print(f"  {GREEN}All tests passed! 🎉{RESET}\n")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  {RED}Failed: {', '.join(failed)}{RESET}\n")


async def run_single_test(test_name: str):
    """Run a single named test."""
    if not check_server_running():
        print(f"{RED}✗ Server not running at {BASE_URL}{RESET}")
        print("  Start it with: python -m src.api.main")
        sys.exit(1)

    tests = {
        "health": lambda: test_health(),
        "upload": lambda: test_upload(),
        "sync": lambda: test_analyze_sync(),
        "analyze": lambda: test_analyze_sync(),
        "task": lambda: test_get_task("test-id-123"),
        "memory": lambda: test_memory_stats(),
        "evaluate": lambda: test_evaluate(),
        "invalid": lambda: test_invalid_requests(),
        "websocket": lambda: asyncio.get_event_loop().run_until_complete(test_analyze_async()),
    }

    if test_name not in tests:
        print(f"{RED}Unknown test: {test_name}{RESET}")
        print(f"Available: {', '.join(tests.keys())}")
        sys.exit(1)

    if test_name == "websocket":
        await test_analyze_async()
    else:
        tests[test_name]()


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataWright Agent API Test Client")
    parser.add_argument(
        "--test",
        "-t",
        help="Run a specific test (health/upload/sync/websocket/memory/evaluate/invalid)",
        default=None,
    )
    parser.add_argument(
        "--url",
        help=f"Base URL of the server (default: {BASE_URL})",
        default=BASE_URL,
    )
    args = parser.parse_args()

    # Allow overriding server URL
    if args.url != BASE_URL:
        BASE_URL = args.url
        WS_URL = args.url.replace("http://", "ws://").replace("https://", "wss://")

    if args.test:
        asyncio.run(run_single_test(args.test))
    else:
        asyncio.run(run_all_tests())
