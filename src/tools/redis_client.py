"""
src/db/redis_client.py
-----------------------
All Redis logic in one place.
Import get_redis() and the helpers into any file that needs them.
"""

import json
import logging
from typing import Optional, Dict

import redis as redis_lib

logger = logging.getLogger(__name__)


def get_redis(host: str, port: int, db: int, password: Optional[str] = None) -> Optional[redis_lib.Redis]:
    """Connect to Redis. Returns None silently if Redis is not running."""
    try:
        client = redis_lib.Redis(
            host=host,
            port=port,
            db=db,
            password=password or None,
            decode_responses=True,
            socket_connect_timeout=3,
        )
        client.ping()
        logger.info(f"Redis connected at {host}:{port}")
        return client
    except Exception as e:
        logger.warning(f"Redis not available: {e} — using in-memory dicts as fallback")
        return None


# ─── FALLBACK STORES (used when Redis is not available) ───────────────────────
_fallback_tasks: Dict[str, str] = {}
_fallback_files: Dict[str, str] = {}


# ─── TASK HELPERS ─────────────────────────────────────────────────────────────

def task_set(r: Optional[redis_lib.Redis], task_id: str, data: dict, ttl: int = 7200):
    value = json.dumps(data, default=str)
    if r:
        r.setex(f"task:{task_id}", ttl, value)
    else:
        _fallback_tasks[task_id] = value


def task_get(r: Optional[redis_lib.Redis], task_id: str) -> Optional[dict]:
    raw = r.get(f"task:{task_id}") if r else _fallback_tasks.get(task_id)
    return json.loads(raw) if raw else None


def task_update(r: Optional[redis_lib.Redis], task_id: str, updates: dict, ttl: int = 7200):
    existing = task_get(r, task_id) or {}
    existing.update(updates)
    task_set(r, task_id, existing, ttl)


def task_count(r: Optional[redis_lib.Redis]) -> int:
    return len(r.keys("task:*")) if r else len(_fallback_tasks)


# ─── FILE HELPERS ─────────────────────────────────────────────────────────────

def file_set(r: Optional[redis_lib.Redis], file_key: str, file_path: str, ttl: int = 3600):
    if r:
        r.setex(f"file:{file_key}", ttl, file_path)
    else:
        _fallback_files[file_key] = file_path


def file_get(r: Optional[redis_lib.Redis], file_key: str) -> Optional[str]:
    return r.get(f"file:{file_key}") if r else _fallback_files.get(file_key)