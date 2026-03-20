import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class StepType(Enum):
    """What kind of action a step performs."""

    THINK = "think"  # Pure reasoning, no external calls
    CODE = "code"  # Write + execute Python code
    ANALYZE = "analyze"  # Interpret code output / data
    SEARCH = "search"  # Search for external information (RAG or web)
    VALIDATE = "validate"  # Self-check: is the current answer correct?
    SUMMARIZE = "summarize"  # Synthesize results into a report
    CLARIFY = "clarify"  # Ask user for more information


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class AgentStatus(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    SELF_CORRECTING = "self_correcting"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_FOR_USER = "waiting_for_user"


@dataclass
class DataContext:
    """
    Describes the data available to the agent.
    Loaded when a file is uploaded or data is provided.
    """

    source: str  # filename or description
    file_type: str  # csv, xlsx, json, text
    shape: Optional[tuple] = None  # (rows, cols) for tabular data
    columns: Optional[List[str]] = None  # column names
    dtypes: Optional[Dict] = None  # column -> dtype mapping
    sample: Optional[str] = None  # first 5 rows as string
    stats: Optional[str] = None  # describe() output
    file_path: Optional[str] = None  # path to actual file


@dataclass
class Step:
    step_id: str
    step_type: StepType
    title: str  # Human-readable name
    description: str  # What to do
    rationale: str  # Why this step is needed
    expected_output: str  # What success looks like
    depends_on: List[str] = field(default_factory=list)  # step_ids this needs
    status: StepStatus = StepStatus.PENDING
    retry_count: int = 0


@dataclass
class StepResult:
    step_id: str
    step_type: StepType
    status: StepStatus

    # Content
    code_written: Optional[str] = None  # Python code (for CODE steps)
    code_output: Optional[str] = None  # stdout from execution
    figures: List[str] = field(default_factory=list)  # base64 encoded plots
    dataframes: List[Dict] = field(default_factory=list)  # table data
    interpretation: str = ""  # Agent's analysis of output
    key_findings: List[str] = field(default_factory=list)

    # Quality
    confidence_score: float = 0.0  # 0-1, how confident the agent is
    self_correction_applied: bool = False  # Did it fix itself?
    correction_explanation: str = ""

    # Error handling
    error: Optional[str] = None
    error_type: Optional[str] = None  # SyntaxError, KeyError, etc.

    # Metadata
    latency_ms: float = 0.0
    tokens_used: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Plan:
    """
    The agent's full execution plan for a goal.

    Created by the Planner before any execution starts.
    The plan can be revised mid-execution if the agent
    discovers the data doesn't match its assumptions.
    """

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    steps: List[Step] = field(default_factory=list)
    reasoning: str = ""  # Why this plan structure
    estimated_complexity: str = "medium"  # low/medium/high
    requires_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AgentState:
    """
    Complete snapshot of agent state at any point in time.

    This is the "world state" of the agent. It contains:
    - The original goal
    - The plan
    - All step results so far
    - Current status
    - Memory of key findings

    WHY THIS MATTERS:
    - Enables resumability: save/load state to continue interrupted tasks
    - Enables debugging: full replay of every decision
    - Enables evaluation: compare planned vs actual execution
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    data_context: Optional[DataContext] = None
    plan: Optional[Plan] = None
    step_results: List[StepResult] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    current_step_index: int = 0
    iteration_count: int = 0

    # Accumulated knowledge as agent progresses
    working_memory: Dict[str, Any] = field(default_factory=dict)
    key_findings: List[str] = field(default_factory=list)
    errors_encountered: List[str] = field(default_factory=list)

    # Final output
    final_report: Optional[str] = None
    final_answer: Optional[str] = None
    figures: List[str] = field(default_factory=list)  # all generated plots

    # Metrics
    total_tokens_used: int = 0
    total_latency_ms: float = 0.0
    self_corrections_count: int = 0
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None

    def get_context_summary(self) -> str:
        """
        Summarize current state for LLM context.
        Passed to LLM at each step so it knows what happened before.
        """
        lines = [f"GOAL: {self.goal}", f"STATUS: {self.status.value}"]

        if self.data_context:
            lines.append(f"DATA: {self.data_context.source} ({self.data_context.shape})")

        if self.key_findings:
            lines.append("\nKEY FINDINGS SO FAR:")
            for f in self.key_findings[-5:]:  # Last 5 to stay within context
                lines.append(f"  - {f}")

        if self.errors_encountered:
            lines.append(f"\nRECENT ERRORS: {self.errors_encountered[-2:]}")

        completed = [r for r in self.step_results if r.status == StepStatus.SUCCESS]
        lines.append(f"\nCOMPLETED STEPS: {len(completed)}/{len(self.plan.steps) if self.plan else '?'}")

        return "\n".join(lines)


@dataclass
class FinalReport:
    """Structured final output delivered to the user."""

    task_id: str
    goal: str
    executive_summary: str  # 2-3 sentence TL;DR
    key_findings: List[str]  # Bulleted insights
    detailed_analysis: str  # Full narrative
    methodology: str  # What the agent did and why
    figures: List[str]  # base64 plots
    tables: List[Dict]  # structured data
    limitations: List[str]  # What the agent couldn't determine
    confidence_score: float  # Overall confidence 0-1
    steps_taken: int
    self_corrections: int
    total_latency_ms: float
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
