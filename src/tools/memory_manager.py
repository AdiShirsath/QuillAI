import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory — either a finding or a task summary."""

    memory_id: str
    memory_type: str  # "finding", "task_summary", "error_pattern", "approach"
    content: str  # The actual memory text
    metadata: Dict[str, Any]  # task_id, timestamp, confidence, etc.
    embedding_text: str = ""  # What to embed for similarity search


class WorkingMemory:
    """
    Fast in-memory storage for a single task execution.
    Stores intermediate results, computed values, and context.

    Think of it as the agent's "scratchpad" during a task.
    """

    def __init__(self, task_id: str, use_redis: bool = False):
        self.task_id = task_id
        self._store: Dict[str, Any] = {}
        self.use_redis = use_redis
        self._redis = None

        if use_redis:
            try:
                import redis

                from src.configs.settings import get_settings

                settings = get_settings()
                self._redis = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)
                self._redis.ping()
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}. Using in-memory dict.")
                self.use_redis = False

    def set(self, key: str, value: Any):
        """Store a value in working memory."""
        if self.use_redis and self._redis:
            try:
                self._redis.setex(f"agent:{self.task_id}:{key}", 3600, json.dumps(value, default=str))  # 1 hour TTL
                return
            except Exception:
                pass
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from working memory."""
        if self.use_redis and self._redis:
            try:
                raw = self._redis.get(f"agent:{self.task_id}:{key}")
                if raw:
                    return json.loads(raw)
            except Exception:
                pass
        return self._store.get(key, default)

    def update_findings(self, new_findings: List[str]):
        """Accumulate key findings throughout the task."""
        existing = self.get("key_findings", [])
        existing.extend(new_findings)
        self.set("key_findings", existing[-20:])  # Keep last 20

    def get_findings(self) -> List[str]:
        return self.get("key_findings", [])

    def store_step_output(self, step_id: str, output: str):
        """Store the text output of a step for later reference."""
        self.set(f"step_output:{step_id}", output)

    def get_step_output(self, step_id: str) -> Optional[str]:
        return self.get(f"step_output:{step_id}")

    def clear(self):
        """Clear all working memory for this task."""
        if self.use_redis and self._redis:
            keys = self._redis.keys(f"agent:{self.task_id}:*")
            if keys:
                self._redis.delete(*keys)
        self._store.clear()

    def to_dict(self) -> Dict:
        """Export full working memory state."""
        return dict(self._store)


class EpisodicMemory:
    """
    Long-term memory that persists across tasks.
    Uses ChromaDB for semantic search over past experiences.

    Stores:
    - Task summaries (what was analyzed, what was found)
    - Successful code patterns (approaches that worked)
    - Error patterns (what failed and why)
    - Domain insights (patterns observed across datasets)

    At the start of each new task, the agent queries this memory
    for relevant past experience.
    """

    def __init__(self, persist_dir: str = "./data/memory"):
        self._collection = None
        self._persist_dir = persist_dir
        self._init_store()

    def _init_store(self):
        """Initialize ChromaDB collection for episodic memory."""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = client.get_or_create_collection(
                name="agent_episodic_memory", metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Episodic memory loaded: {self._collection.count()} entries")
        except Exception as e:
            logger.warning(f"ChromaDB unavailable: {e}. Episodic memory disabled.")

    def remember(self, entry: MemoryEntry):
        """Store a new memory entry."""
        if not self._collection:
            return
        try:
            self._collection.add(
                ids=[entry.memory_id],
                documents=[entry.content],
                metadatas=[
                    {
                        "memory_type": entry.memory_type,
                        "timestamp": datetime.utcnow().isoformat(),
                        **{k: str(v) for k, v in entry.metadata.items()},
                    }
                ],
            )
        except Exception as e:
            logger.warning(f"Failed to store memory: {e}")

    def recall(self, query: str, n_results: int = 3, memory_type: Optional[str] = None) -> List[MemoryEntry]:
        """
        Search episodic memory for relevant past experiences.
        Returns most semantically similar memories.
        """
        if not self._collection or self._collection.count() == 0:
            return []

        try:
            where = {"memory_type": memory_type} if memory_type else None
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, self._collection.count()),
                where=where,
            )

            memories = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                memories.append(
                    MemoryEntry(
                        memory_id=results["ids"][0][i],
                        memory_type=meta.get("memory_type", "unknown"),
                        content=doc,
                        metadata=meta,
                    )
                )
            return memories
        except Exception as e:
            logger.warning(f"Memory recall failed: {e}")
            return []

    def remember_task(self, task_id: str, goal: str, summary: str, key_findings: List[str]):
        """Store a completed task summary in episodic memory."""
        content = f"TASK: {goal}\n\nSUMMARY: {summary}\n\nKEY FINDINGS:\n" + "\n".join(f"- {f}" for f in key_findings)

        self.remember(
            MemoryEntry(
                memory_id=f"task_{task_id}",
                memory_type="task_summary",
                content=content,
                metadata={"task_id": task_id, "goal": goal[:100]},
            )
        )

    def remember_successful_approach(self, task_id: str, description: str, code: str):
        """Store a code pattern that successfully solved a problem."""
        content = f"APPROACH: {description}\n\nCODE:\n{code[:500]}"
        self.remember(
            MemoryEntry(
                memory_id=f"approach_{uuid.uuid4().hex[:8]}",
                memory_type="approach",
                content=content,
                metadata={"task_id": task_id, "description": description[:100]},
            )
        )

    def remember_error_pattern(self, error_type: str, context: str, resolution: str):
        """Store an error and its fix for future reference."""
        content = f"ERROR TYPE: {error_type}\nCONTEXT: {context}\nRESOLUTION: {resolution}"
        self.remember(
            MemoryEntry(
                memory_id=f"error_{uuid.uuid4().hex[:8]}",
                memory_type="error_pattern",
                content=content,
                metadata={"error_type": error_type},
            )
        )

    def get_relevant_context(self, goal: str, data_description: str) -> str:
        """
        At task start: retrieve most relevant past experiences.
        Returns formatted string for injection into agent context.
        """
        query = f"{goal} {data_description}"
        memories = self.recall(query, n_results=3)

        if not memories:
            return ""

        lines = ["RELEVANT PAST EXPERIENCE:"]
        for m in memories:
            lines.append(f"\n[{m.memory_type.upper()}] {m.content[:300]}...")

        return "\n".join(lines)
