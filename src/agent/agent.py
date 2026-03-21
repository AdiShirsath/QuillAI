import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional

import pandas as pd

from src.agent.executor import Executor
from src.agent.models import AgentState, AgentStatus, DataContext, FinalReport, StepStatus, StepType
from src.agent.planner import Planner
from src.configs.settings import get_settings
from src.tools.memory_manager import EpisodicMemory, WorkingMemory

logger = logging.getLogger(__name__)


# ─── AGENT EVENT ──────────────────────────────────────────────────────────────


class AgentEvent:
    """An event emitted during agent execution for streaming to UI."""

    def __init__(self, event_type: str, data: Dict):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
        }


# ─── MAIN AGENT ───────────────────────────────────────────────────────────────


class DataAnalysisAgent:
    """
    Autonomous Data Analysis Agent.

    Takes a natural language goal and a dataset, produces a complete
    analytical report through autonomous multi-step reasoning and execution.

    Example:
        agent = DataAnalysisAgent()
        report = await agent.run(
            goal="Find the top factors driving customer churn and build a predictive model",
            file_path="data/customers.csv"
        )
    """

    def __init__(self):
        self.settings = get_settings()
        self.planner = Planner()
        self.executor = Executor()
        self.episodic_memory = EpisodicMemory(persist_dir=self.settings.chroma_persist_dir)

    # ─── PUBLIC API ───────────────────────────────────────────────────────────

    async def run(
        self,
        goal: str,
        file_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        task_id: Optional[str] = None,
    ) -> FinalReport:
        """
        Main entry point. Run a full analysis task.

        Args:
            goal: Natural language analysis goal
            file_path: Path to data file (CSV, Excel, JSON)
            dataframe: Pre-loaded DataFrame (alternative to file_path)
            task_id: Optional ID for tracking (auto-generated if not provided)

        Returns:
            FinalReport with all findings, figures, and methodology
        """
        task_id = task_id or str(uuid.uuid4())[:8]
        state = AgentState(task_id=task_id, goal=goal)
        events = []

        async for event in self._run_streaming(state, file_path, dataframe):
            events.append(event)
            if state.status == AgentStatus.COMPLETED:
                break

        return self._build_final_report(state)

    async def run_streaming(
        self,
        goal: str,
        file_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        task_id: Optional[str] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Streaming version — yields events as they happen.
        Use this for WebSocket connections to show live progress.
        """
        task_id = task_id or str(uuid.uuid4())[:8]
        state = AgentState(task_id=task_id, goal=goal)
        async for event in self._run_streaming(state, file_path, dataframe):
            yield event

    # ─── CORE EXECUTION LOOP ──────────────────────────────────────────────────

    async def _run_streaming(
        self,
        state: AgentState,
        file_path: Optional[str],
        dataframe: Optional[pd.DataFrame],
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        The main event loop. Yields AgentEvents as execution progresses.
        This is the heart of the agent.
        """
        t_start = time.time()
        working_memory = WorkingMemory(task_id=state.task_id, use_redis=True)

        # ── PHASE 1: Data Loading ──────────────────────────────────────────────
        try:
            df, data_context = self._load_and_profile_data(file_path, dataframe)
            state.data_context = data_context
            if df is not None:
                working_memory.set("dataframe", df.to_json())
        except Exception as e:
            state.status = AgentStatus.FAILED
            yield AgentEvent("error", {"message": f"Data loading failed: {e}"})
            return

        # ── PHASE 2: Episodic Memory Recall ───────────────────────────────────
        past_context = ""
        if data_context:
            past_context = self.episodic_memory.get_relevant_context(
                goal=state.goal, data_description=f"{data_context.source} {data_context.columns}"
            )
            if past_context:
                logger.info("Recalled relevant past experience")
                state.working_memory["past_context"] = past_context

        # ── PHASE 3: Planning ──────────────────────────────────────────────────
        state.status = AgentStatus.PLANNING
        yield AgentEvent("planning_start", {"goal": state.goal})

        state.plan = self.planner.create_plan(
            goal=state.goal,
            data_context=data_context,
        )

        # Handle clarification needed
        if state.plan.requires_clarification:
            state.status = AgentStatus.WAITING_FOR_USER
            yield AgentEvent(
                "clarification_needed",
                {
                    "questions": state.plan.clarification_questions,
                    "plan_reasoning": state.plan.reasoning,
                },
            )
            # In a real system, you'd pause here and wait for user input
            # For now, proceed with best-effort plan

        yield AgentEvent(
            "planning_complete",
            {
                "plan": {
                    "steps": [
                        {
                            "step_id": s.step_id,
                            "title": s.title,
                            "step_type": s.step_type.value,
                            "rationale": s.rationale,
                        }
                        for s in state.plan.steps
                    ],
                    "reasoning": state.plan.reasoning,
                    "complexity": state.plan.estimated_complexity,
                    "past_context_used": bool(past_context),
                }
            },
        )

        # ── PHASE 4: Execution Loop ────────────────────────────────────────────
        state.status = AgentStatus.EXECUTING
        data_runtime = {"df": df} if df is not None else {}

        for i, step in enumerate(state.plan.steps):
            state.current_step_index = i

            # Check if dependencies are satisfied
            if not self._dependencies_met(step, state):
                logger.warning(f"Skipping step {step.step_id} — dependencies not met")
                step.status = StepStatus.SKIPPED
                continue

            # Emit step start event
            yield AgentEvent(
                "step_start",
                {
                    "step_id": step.step_id,
                    "step_type": step.step_type.value,
                    "title": step.title,
                    "description": step.description,
                    "step_number": i + 1,
                    "total_steps": len(state.plan.steps),
                },
            )

            # Execute the step
            result = self.executor.execute_step(
                step=step,
                state=state,
                data_context=data_runtime,
            )

            # Update step status
            step.status = result.status
            state.step_results.append(result)
            state.total_tokens_used += result.tokens_used
            state.total_latency_ms += result.latency_ms
            state.iteration_count += 1

            if result.self_correction_applied:
                state.self_corrections_count += 1

            # Accumulate findings
            if result.key_findings:
                state.key_findings.extend(result.key_findings)
                working_memory.update_findings(result.key_findings)

            # Accumulate figures
            if result.figures:
                state.figures.extend(result.figures)

            # Store step output in working memory for dependency access
            if result.interpretation:
                working_memory.store_step_output(step.step_id, result.interpretation)

            # Emit step result event
            yield AgentEvent(
                "step_complete",
                {
                    "step_id": step.step_id,
                    "status": result.status.value,
                    "interpretation": result.interpretation,
                    "key_findings": result.key_findings,
                    "confidence_score": result.confidence_score,
                    "self_correction_applied": result.self_correction_applied,
                    "correction_explanation": result.correction_explanation,
                    "has_figures": len(result.figures) > 0,
                    "figures": result.figures,
                    "code_written": result.code_written,
                    "code_output": result.code_output,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                },
            )

            if result.self_correction_applied:
                state.status = AgentStatus.SELF_CORRECTING
                yield AgentEvent(
                    "self_correction",
                    {
                        "step_id": step.step_id,
                        "explanation": result.correction_explanation,
                    },
                )
                state.status = AgentStatus.EXECUTING

            # Track errors
            if result.status == StepStatus.FAILED:
                state.errors_encountered.append(f"Step {step.step_id}: {result.error_type} - {result.error}")

            # ── ADAPTIVE REPLANNING ────────────────────────────────────────────
            should_replan, reason = self.planner.assess_if_replan_needed(state, result)

            if should_replan and i < len(state.plan.steps) - 2:
                state.status = AgentStatus.PLANNING
                yield AgentEvent("replan_start", {"reason": reason})

                revised_plan = self.planner.replan(state=state, reason=reason)

                # Replace remaining steps with revised steps
                state.plan.steps = state.plan.steps[: i + 1] + revised_plan.steps
                state.status = AgentStatus.EXECUTING

                yield AgentEvent(
                    "replan_complete",
                    {
                        "reason": reason,
                        "new_steps": len(revised_plan.steps),
                    },
                )

            # ── STORE IN EPISODIC MEMORY (for successful code) ─────────────────
            if result.status == StepStatus.SUCCESS and result.code_written and result.confidence_score > 0.8:
                self.episodic_memory.remember_successful_approach(
                    task_id=state.task_id,
                    description=step.description,
                    code=result.code_written,
                )

            if result.status == StepStatus.FAILED and result.self_correction_applied and result.correction_explanation:
                self.episodic_memory.remember_error_pattern(
                    error_type=result.error_type or "Unknown",
                    context=step.description[:100],
                    resolution=result.correction_explanation,
                )

        # ── PHASE 5: Final Report ──────────────────────────────────────────────
        state.status = AgentStatus.COMPLETED
        state.completed_at = datetime.utcnow().isoformat()
        state.total_latency_ms = (time.time() - t_start) * 1000

        final_report = self._build_final_report(state)
        state.final_report = final_report.detailed_analysis
        state.final_answer = final_report.executive_summary

        # Store task in episodic memory for future reference
        self.episodic_memory.remember_task(
            task_id=state.task_id,
            goal=state.goal,
            summary=final_report.executive_summary,
            key_findings=final_report.key_findings,
        )

        yield AgentEvent(
            "complete",
            {
                "task_id": state.task_id,
                "executive_summary": final_report.executive_summary,
                "key_findings": final_report.key_findings,
                "detailed_analysis": final_report.detailed_analysis,
                "methodology": final_report.methodology,
                "figures": final_report.figures,
                "tables": final_report.tables,
                "limitations": final_report.limitations,
                "confidence_score": final_report.confidence_score,
                "steps_taken": final_report.steps_taken,
                "self_corrections": final_report.self_corrections,
                "total_latency_ms": final_report.total_latency_ms,
            },
        )

        working_memory.clear()

    # ─── DATA LOADING ─────────────────────────────────────────────────────────

    def _load_and_profile_data(
        self,
        file_path: Optional[str],
        dataframe: Optional[pd.DataFrame],
    ) -> tuple[Optional[pd.DataFrame], Optional[DataContext]]:
        """
        Load data and create a DataContext description.
        The DataContext is what gets passed to the Planner for plan creation.
        """
        df = None

        if dataframe is not None:
            df = dataframe
            source = "provided_dataframe"
            file_type = "dataframe"

        elif file_path:
            path = Path(file_path)
            source = path.name
            file_type = path.suffix.lstrip(".").lower()

            if file_type == "csv":
                df = pd.read_csv(file_path)
            elif file_type in ("xlsx", "xls"):
                df = pd.read_excel(file_path)
            elif file_type == "json":
                df = pd.read_json(file_path)
            elif file_type == "parquet":
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        else:
            return None, None

        # Profile the dataframe
        data_context = DataContext(
            source=source,
            file_type=file_type,
            shape=df.shape,
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            sample=df.head(3).to_string(),
            stats=df.describe(include="all").to_string(),
            file_path=file_path,
        )

        logger.info(f"Loaded data: {source} ({df.shape[0]:,} rows × {df.shape[1]} cols)")
        return df, data_context

    # ─── HELPERS ──────────────────────────────────────────────────────────────

    def _dependencies_met(self, step, state: AgentState) -> bool:
        """Check if all steps this step depends on have completed successfully."""
        completed_ids = {r.step_id for r in state.step_results if r.status == StepStatus.SUCCESS}
        return all(dep_id in completed_ids for dep_id in step.depends_on)

    def _build_final_report(self, state: AgentState) -> FinalReport:
        """Synthesize all step results into a final structured report."""
        successful_steps = [r for r in state.step_results if r.status == StepStatus.SUCCESS]

        # Get the summarize step output if exists
        summary_result = next(
            (
                r
                for r in reversed(state.step_results)
                if r.step_type == StepType.SUMMARIZE and r.status == StepStatus.SUCCESS
            ),
            None,
        )

        executive_summary = (
            summary_result.interpretation[:500]
            if summary_result
            else (state.key_findings[0] if state.key_findings else "Analysis complete.")
        )

        detailed_analysis = (
            summary_result.interpretation
            if summary_result
            else "\n\n".join(
                f"**{r.step_type.value.title()}**: {r.interpretation}" for r in successful_steps if r.interpretation
            )
        )

        methodology = (
            f"The agent executed {len(state.step_results)} steps "
            f"({len(successful_steps)} successful) following an "
            f"{state.plan.estimated_complexity if state.plan else 'medium'}-complexity plan. "
            + (
                f"Self-corrections were applied {state.self_corrections_count} time(s). "
                if state.self_corrections_count > 0
                else ""
            )
        )

        limitations = []
        if state.errors_encountered:
            limitations.append(
                f"Some analysis steps encountered errors: {len(state.errors_encountered)} step(s) failed."
            )
        failed_steps = [r for r in state.step_results if r.status == StepStatus.FAILED]
        for fr in failed_steps:
            limitations.append(f"Could not complete: {fr.step_type.value} ({fr.error_type})")

        avg_confidence = (
            sum(r.confidence_score for r in successful_steps) / len(successful_steps) if successful_steps else 0.5
        )

        all_tables = []
        for r in successful_steps:
            all_tables.extend(r.dataframes)

        return FinalReport(
            task_id=state.task_id,
            goal=state.goal,
            executive_summary=executive_summary,
            key_findings=state.key_findings[:10],
            detailed_analysis=detailed_analysis,
            methodology=methodology,
            figures=state.figures,
            tables=all_tables[:5],
            limitations=limitations,
            confidence_score=round(avg_confidence, 3),
            steps_taken=len(state.step_results),
            self_corrections=state.self_corrections_count,
            total_latency_ms=state.total_latency_ms,
        )
