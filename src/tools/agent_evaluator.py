
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.configs.settings import get_settings
from src.agent.models import FinalReport, AgentState, StepStatus

logger = logging.getLogger(__name__)


@dataclass
class TaskEvalResult:
    """Evaluation result for a single task run."""
    task_id: str
    goal: str

    # Task-level metrics
    task_completed: bool = False        # Did agent reach COMPLETED status?
    steps_taken: int = 0
    steps_succeeded: int = 0
    steps_failed: int = 0
    self_corrections: int = 0
    successful_corrections: int = 0    # Corrections that fixed the problem
    latency_ms: float = 0.0

    # Quality metrics (from LLM judge)
    answer_quality_score: float = 0.0  # 1-5 scale, normalized to 0-1
    plan_quality_score: float = 0.0    # Was the plan appropriate?
    answer_grounded: float = 0.0       # Is answer based on actual data?

    # Computed metrics
    step_success_rate: float = 0.0
    self_correction_success_rate: float = 0.0
    plan_efficiency: float = 0.0       # useful_steps / total_steps

    # Reference comparison (if ground truth provided)
    expected_findings: List[str] = field(default_factory=list)
    findings_coverage: float = 0.0    # % of expected findings found

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    run_name: str = ""


@dataclass
class BenchmarkResult:
    """Aggregated results across multiple tasks."""
    run_name: str
    n_tasks: int
    timestamp: str

    # Aggregate metrics
    task_completion_rate: float = 0.0
    avg_step_success_rate: float = 0.0
    avg_self_correction_success_rate: float = 0.0
    avg_answer_quality: float = 0.0
    avg_plan_quality: float = 0.0
    avg_plan_efficiency: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    # Totals
    total_self_corrections: int = 0
    total_steps_taken: int = 0

    individual_results: List[TaskEvalResult] = field(default_factory=list)


class AgentEvaluator:
    """
    Evaluates agent performance using a combination of:
    - Deterministic metrics (step counts, success rates, timing)
    - LLM-as-judge (answer quality, plan quality, grounding)
    - Reference comparison (when ground truth is available)
    """

    def __init__(self, run_name: str = "default"):
        self.settings = get_settings()
        self.run_name = run_name
        self.results_dir = Path(self.settings.eval_results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.judge_llm = ChatGroq(
            model=self.settings.evaluator_model,
            temperature=0.0, # this makes model consistent
            api_key=self.settings.groq_api_key,
        )

    # ─── PUBLIC API ───────────────────────────────────────────────────────────

    def evaluate_task(
        self,
        report: FinalReport,
        state: AgentState,
        expected_findings: Optional[List[str]] = None,
    ) -> TaskEvalResult:
        """
        Evaluate a single completed task.

        Args:
            report: The FinalReport produced by the agent
            state: The full AgentState (for step-level metrics)
            expected_findings: Ground truth findings (optional)

        Returns:
            TaskEvalResult with all metrics
        """
        result = TaskEvalResult(
            task_id=report.task_id,
            goal=report.goal,
            run_name=self.run_name,
            task_completed=state.status.value == "completed",
            steps_taken=len(state.step_results),
            steps_succeeded=sum(1 for r in state.step_results if r.status == StepStatus.SUCCESS),
            steps_failed=sum(1 for r in state.step_results if r.status == StepStatus.FAILED),
            self_corrections=state.self_corrections_count,
            latency_ms=report.total_latency_ms,
            expected_findings=expected_findings or [],
        )

        # Compute deterministic metrics
        if result.steps_taken > 0:
            result.step_success_rate = result.steps_succeeded / result.steps_taken

        # Plan efficiency: skip THINK steps (non-executable) in denominator
        executable_steps = sum(
            1 for s in state.step_results
            if s.step_type.value != "think"
        )
        if executable_steps > 0:
            result.plan_efficiency = result.steps_succeeded / executable_steps

        # Self-correction success rate
        corrected_steps = [r for r in state.step_results if r.self_correction_applied]
        successful_corrections = [r for r in corrected_steps if r.status == StepStatus.SUCCESS]
        result.successful_corrections = len(successful_corrections)
        if corrected_steps:
            result.self_correction_success_rate = len(successful_corrections) / len(corrected_steps)

        # LLM judge metrics
        result.answer_quality_score = self._judge_answer_quality(
            goal=report.goal,
            answer=report.detailed_analysis,
        )
        result.plan_quality_score = self._judge_plan_quality(
            goal=report.goal,
            steps=state.plan.steps if state.plan else [],
            complexity=state.plan.estimated_complexity if state.plan else "medium",
        )
        result.answer_grounded = self._judge_grounding(
            answer=report.detailed_analysis,
            findings=report.key_findings,
        )

        # Reference comparison
        if expected_findings:
            result.findings_coverage = self._compute_findings_coverage(
                actual=report.key_findings,
                expected=expected_findings,
            )

        return result

    def evaluate_batch(
        self,
        task_reports: List[tuple],  # (FinalReport, AgentState, Optional[List[str]])
    ) -> BenchmarkResult:
        """
        Evaluate multiple tasks and aggregate metrics.
        Use this for A/B testing different agent configurations.
        """
        individual_results = []

        for i, task_data in enumerate(task_reports):
            report, state, expected = task_data[0], task_data[1], task_data[2] if len(task_data) > 2 else None
            logger.info(f"Evaluating task {i+1}/{len(task_reports)}: {report.goal[:60]}...")
            result = self.evaluate_task(report, state, expected)
            individual_results.append(result)

        benchmark = self._aggregate_results(individual_results)
        self._save_results(benchmark)
        self._print_report(benchmark)
        return benchmark

    # ─── LLM JUDGE METHODS ────────────────────────────────────────────────────

    def _judge_answer_quality(self, goal: str, answer: str) -> float:
        """Use LLM to score answer quality on 0-1 scale."""
        prompt = f"""Rate the quality of this analytical answer on a scale of 1-5.

GOAL: {goal}

ANSWER: {answer[:1500]}

Criteria:
- 5: Comprehensive, specific with numbers, addresses all aspects of the goal
- 4: Mostly complete, specific, minor gaps
- 3: Partially addresses goal, some vague claims  
- 2: Superficial, missing key aspects
- 1: Does not address the goal

Respond with ONLY a single integer (1-5):"""

        try:
            response = self.judge_llm.invoke(prompt)
            score = int(re.search(r'\d', response.content).group())
            return max(1, min(5, score)) / 5  # Normalize to 0-1
        except Exception:
            return 0.5

    def _judge_plan_quality(self, goal: str, steps: List, complexity: str) -> float:
        """Evaluate if the plan was appropriate for the goal."""
        if not steps:
            return 0.0

        steps_desc = "\n".join(f"- [{s.step_type.value}] {s.title}" for s in steps[:10])

        prompt = f"""Rate this analysis plan's quality on a scale of 1-5.

GOAL: {goal}
COMPLEXITY: {complexity}

PLAN STEPS:
{steps_desc}

Criteria:
- 5: Logical sequence, appropriate depth, covers all aspects of goal
- 4: Good sequence, minor inefficiencies
- 3: Reasonable but missing some important steps
- 2: Poorly structured, missing critical steps
- 1: Wrong approach for the goal

Respond with ONLY a single integer (1-5):"""

        try:
            response = self.judge_llm.invoke(prompt)
            score = int(re.search(r'\d', response.content).group())
            return max(1, min(5, score)) / 5
        except Exception:
            return 0.5

    def _judge_grounding(self, answer: str, findings: List[str]) -> float:
        """Check if the answer is grounded in actual findings vs hallucinated."""
        if not findings:
            return 0.5

        findings_text = "\n".join(f"- {f}" for f in findings)

        prompt = f"""Rate 0-1: How well is this answer grounded in the stated findings?
1.0 = Every claim is directly supported by a finding
0.0 = Answer contains claims not in the findings (hallucinated)

FINDINGS:
{findings_text}

ANSWER: {answer[:1000]}

Respond with ONLY a decimal between 0 and 1:"""

        try:
            response = self.judge_llm.invoke(prompt)
            score = float(re.search(r'0?\.\d+|[01]', response.content).group())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5

    def _compute_findings_coverage(self, actual: List[str], expected: List[str]) -> float:
        """
        Compute what fraction of expected findings were discovered.
        Uses LLM to check semantic match (not just exact string match).
        """
        if not expected or not actual:
            return 0.0

        covered = 0
        for exp in expected:
            prompt = f"""Does any item in this list capture the same insight as: "{exp}"?

LIST:
{chr(10).join(f"- {a}" for a in actual[:10])}

Answer YES or NO:"""
            try:
                response = self.judge_llm.invoke(prompt)
                if "yes" in response.content.lower():
                    covered += 1
            except Exception:
                pass

        return covered / len(expected)

    # ─── REPORTING ────────────────────────────────────────────────────────────

    def _aggregate_results(self, results: List[TaskEvalResult]) -> BenchmarkResult:
        """Aggregate individual task results into benchmark metrics."""
        n = len(results)
        if n == 0:
            return BenchmarkResult(run_name=self.run_name, n_tasks=0, timestamp=datetime.utcnow().isoformat())

        latencies = sorted([r.latency_ms for r in results])

        return BenchmarkResult(
            run_name=self.run_name,
            n_tasks=n,
            timestamp=datetime.utcnow().isoformat(),
            task_completion_rate=sum(1 for r in results if r.task_completed) / n,
            avg_step_success_rate=sum(r.step_success_rate for r in results) / n,
            avg_self_correction_success_rate=sum(r.self_correction_success_rate for r in results) / n,
            avg_answer_quality=sum(r.answer_quality_score for r in results) / n,
            avg_plan_quality=sum(r.plan_quality_score for r in results) / n,
            avg_plan_efficiency=sum(r.plan_efficiency for r in results) / n,
            avg_latency_ms=sum(r.latency_ms for r in results) / n,
            p95_latency_ms=latencies[int(0.95 * n)] if n >= 20 else max(latencies),
            total_self_corrections=sum(r.self_corrections for r in results),
            total_steps_taken=sum(r.steps_taken for r in results),
            individual_results=results,
        )

    def _save_results(self, benchmark: BenchmarkResult):
        """Save benchmark results to JSON."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = self.results_dir / f"benchmark_{self.run_name}_{ts}.json"
        data = asdict(benchmark)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved benchmark results to {path}")

    def _print_report(self, b: BenchmarkResult):
        """Print a formatted benchmark report."""
        def bar(val, width=20):
            filled = int(val * width)
            return "█" * filled + "░" * (width - filled) + f" {val:.1%}"

        def grade(val):
            if val >= 0.9: return "🟢"
            if val >= 0.7: return "🟡"
            if val >= 0.5: return "🟠"
            return "🔴"

        print("\n" + "═"*65)
        print(f"  🤖 AGENT BENCHMARK REPORT — {b.run_name}")
        print("═"*65)
        print(f"  Tasks evaluated: {b.n_tasks} | Run: {b.timestamp[:10]}")
        print("─"*65)

        metrics = [
            ("Task Completion Rate", b.task_completion_rate),
            ("Step Success Rate", b.avg_step_success_rate),
            ("Self-Correction Success", b.avg_self_correction_success_rate),
            ("Answer Quality (LLM judge)", b.avg_answer_quality),
            ("Plan Quality", b.avg_plan_quality),
            ("Plan Efficiency", b.avg_plan_efficiency),
        ]

        for name, val in metrics:
            print(f"  {grade(val)} {name:<35} {bar(val)}")

        print("─"*65)
        print(f"  ⏱  Avg Latency:  {b.avg_latency_ms/1000:.1f}s")
        print(f"  ⏱  P95 Latency:  {b.p95_latency_ms/1000:.1f}s")
        print(f"  🔧 Self-Corrections: {b.total_self_corrections} total")
        print(f"  📊 Steps Taken: {b.total_steps_taken} total")
        print("═"*65 + "\n")
