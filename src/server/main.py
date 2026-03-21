import asyncio
import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.agent import DataAnalysisAgent
from src.configs.settings import get_settings
from src.server.ui_pages import ui_router  # ← all UI page routes
from src.tools.redis_client import (  # ← new import
    file_get,
    file_set,
    get_redis,
    task_count,
    task_get,
    task_set,
    task_update,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

_redis = None
_agent: Optional[DataAnalysisAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _redis
    _redis = get_redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=getattr(settings, "redis_password", None),
    )
    logger.info("🤖 Initializing Autonomous Data Analysis Agent...")
    _agent = DataAnalysisAgent()
    logger.info("✅ Agent ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="QuillAI: Autonomous Data Analysis Agent API",
    description="""
    An autonomous AI agent that analyzes datasets by:
    - 🧠 Planning: Creates a multi-step analysis plan
    - 💻 Executing: Writes and runs Python code
    - 🔧 Self-Correcting: Fixes errors automatically
    - 📊 Evaluating: Scores its own output quality

    Connect via WebSocket to see the agent think in real-time.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ui pages to make front end/, /docs, /dashboard, /ws-test

app.include_router(ui_router)


# ─── REQUEST SCHEMAS ──────────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    goal: str = Field(..., min_length=10, description="What you want to analyze")
    file_key: Optional[str] = Field(None, description="File key from /upload endpoint")
    sample_data: Optional[str] = Field(None, description="CSV data as string (for testing)")

    class Config:
        json_schema_extra = {
            "example": {
                "goal": "Find the top 3 factors driving customer churn. Build a model and explain the results.",
                "file_key": "abc123",
            }
        }


class EvalRequest(BaseModel):
    tasks: List[Dict[str, Any]] = Field(..., description="List of {goal, file_key, expected_findings}")
    run_name: str = Field("api_eval", description="Name for this eval run")


# ─── WEBSOCKET MANAGER ────────────────────────────────────────────────────────


class ConnectionManager:
    """Manages WebSocket connections for streaming agent events."""

    def __init__(self):
        self.active: Dict[str, WebSocket] = {}  # task_id -> websocket

    async def connect(self, task_id: str, ws: WebSocket):
        await ws.accept()
        self.active[task_id] = ws

    def disconnect(self, task_id: str):
        self.active.pop(task_id, None)

    async def send(self, task_id: str, event: Dict):
        ws = self.active.get(task_id)
        if ws:
            try:
                await ws.send_json(event)
            except Exception:
                self.disconnect(task_id)


manager = ConnectionManager()


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    redis_status = "unavailable"
    if _redis:
        try:
            _redis.ping()
            redis_status = "healthy"
        except Exception:
            redis_status = "error"

    return {
        "status": "healthy",
        "agent_ready": _agent is not None,
        "redis": redis_status,
        "active_connections": len(manager.active),
        "tasks_in_memory": task_count(_redis),
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a data file. Returns a file_key to use in /analyze."""
    allowed = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed}")

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    content = await file.read()
    tmp.write(content)
    tmp.flush()

    file_key = str(uuid.uuid4())[:8]
    file_set(_redis, file_key, tmp.name, ttl=3600)

    # Quick preview
    try:
        if ext == ".csv":
            df = pd.read_csv(tmp.name)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(tmp.name)
        else:
            df = pd.read_json(tmp.name)

        preview = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample": df.head(3).to_dict(orient="records"),
        }
    except Exception as e:
        preview = {"error": str(e)}

    return {
        "file_key": file_key,
        "filename": file.filename,
        "preview": preview,
    }


@app.post("/analyze")
async def start_analysis(request: AnalyzeRequest):
    """
    Start an async analysis task.
    Returns task_id immediately. Connect to /ws/{task_id} for live updates.
    """
    if not _agent:
        raise HTTPException(503, "Agent not initialized")

    task_id = str(uuid.uuid4())[:8]
    task_set(_redis, task_id, {"status": "queued", "goal": request.goal})

    # Resolve file path
    file_path = None
    df = None

    if request.file_key:
        file_path = file_get(_redis, request.file_key)
        if not file_path:
            raise HTTPException(404, f"File key '{request.file_key}' not found. Upload file first.")

    elif request.sample_data:
        import io

        df = pd.read_csv(io.StringIO(request.sample_data))

    # Run agent in background
    asyncio.create_task(_run_agent_task(task_id, request.goal, file_path, df))

    return {
        "task_id": task_id,
        "status": "started",
        "websocket_url": f"/ws/{task_id}",
        "message": f"Connect to WebSocket /ws/{task_id} for live updates",
    }


async def _run_agent_task(
    task_id: str,
    goal: str,
    file_path: Optional[str],
    df: Optional[pd.DataFrame],
):
    """Background coroutine that runs the agent and broadcasts events."""
    task_update(_redis, task_id, {"status": "running"})

    try:
        async for event in _agent.run_streaming(
            goal=goal,
            file_path=file_path,
            dataframe=df,
            task_id=task_id,
        ):
            event_dict = event.to_dict()
            task_update(_redis, task_id, {"status": "completed", "result": event.data})

            # Send to WebSocket if connected
            await manager.send(task_id, event_dict)

            if event.event_type == "complete":
                task_update(_redis, task_id, {"status": "completed", "result": event.data})
            elif event.event_type == "error":
                task_update(_redis, task_id, {"status": "error", "error": event.data})

    except Exception as e:

        task_update(_redis, task_id, {"status": "error", "error": str(e)})

        await manager.send(task_id, {"event_type": "error", "data": {"message": str(e)}})


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time agent event streaming.

    If the task already completed before connecting,
    replays the final result immediately.
    """
    await manager.connect(task_id, websocket)

    try:
        # If task already done, send final result immediately
        task = task_get(_redis, task_id) or {}
        if task.get("status") == "completed" and "result" in task:
            await websocket.send_json(
                {
                    "event_type": "complete",
                    "data": task["result"],
                    "replayed": True,
                }
            )
            return

        # Keep connection alive until task completes
        while True:
            try:
                # Ping to keep alive (client should pong back)
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"event_type": "heartbeat"})
            except WebSocketDisconnect:
                break

            # Check if task finished
            current = task_get(_redis, task_id) or {}
            if current.get("status") in ("completed", "error"):
                break

    finally:
        manager.disconnect(task_id)


@app.post("/analyze/sync")
async def analyze_sync(request: AnalyzeRequest):
    """
    Synchronous analysis — waits for completion.
    Use for simple testing; prefer async + WebSocket for production.
    """
    if not _agent:
        raise HTTPException(503, "Agent not initialized")

    file_path = None
    df = None

    if request.file_key:
        file_path = file_get(_redis, request.file_key)
    elif request.sample_data:
        import io

        df = pd.read_csv(io.StringIO(request.sample_data))

    report = await _agent.run(
        goal=request.goal,
        file_path=file_path,
        dataframe=df,
    )

    return {
        "task_id": report.task_id,
        "executive_summary": report.executive_summary,
        "key_findings": report.key_findings,
        "detailed_analysis": report.detailed_analysis,
        "methodology": report.methodology,
        "limitations": report.limitations,
        "confidence_score": report.confidence_score,
        "steps_taken": report.steps_taken,
        "self_corrections": report.self_corrections,
        "total_latency_ms": report.total_latency_ms,
        "has_figures": len(report.figures) > 0,
        "figure_count": len(report.figures),
    }


@app.get("/task/{task_id}")
async def get_task(task_id: str):
    """Get the current status and result of a task."""
    task = task_get(_redis, task_id)
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")
    return task


@app.get("/memory/stats")
async def memory_stats():
    """Show what's stored in episodic memory."""
    if not _agent:
        raise HTTPException(503, "Agent not initialized")

    try:
        collection = _agent.episodic_memory._collection
        if not collection:
            return {"total_memories": 0, "memory_enabled": False}

        total = collection.count()
        return {
            "total_memories": total,
            "memory_enabled": True,
            "memory_dir": settings.chroma_persist_dir,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
