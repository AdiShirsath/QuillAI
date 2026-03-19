import json
import logging
import re
import uuid
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.configs.settings import get_settings
from src.agent.models import (
    Plan, Step, StepType, StepStatus,
    DataContext, AgentState
)

logger = logging.getLogger(__name__)


# Prompts:
PLANNER_SYSTEM = """You are a senior data scientist and analytical planner.

Your job: Given a user's analysis goal and their dataset description, create a precise,
ordered execution plan that an AI agent will follow step by step.

STEP TYPES AVAILABLE:
- THINK: Pure reasoning — formulate hypotheses, decide approach, no code
- CODE: Write and execute Python code (pandas, matplotlib, scipy, sklearn)
- ANALYZE: Interpret the output of a CODE step — find patterns, anomalies, insights
- VALIDATE: Self-check — verify findings are statistically sound, not artifacts
- SUMMARIZE: Synthesize all findings into a final report

PLANNING RULES:
1. Every CODE step must be followed by an ANALYZE step
2. Complex goals need a THINK step first to formulate approach
3. Statistical claims need a VALIDATE step
4. Always end with SUMMARIZE
5. Keep plans to 5-10 steps — not too shallow, not too verbose
6. Be specific: "Calculate correlation matrix between numeric columns" not "analyze data"
7. Each step's rationale must explain WHY, not just WHAT

OUTPUT: Respond ONLY with valid JSON matching this exact schema:
{{
  "reasoning": "Why you chose this plan structure",
  "estimated_complexity": "low|medium|high",
  "requires_clarification": false,
  "clarification_questions": [],
  "steps": [
    {{
      "step_id": "s1",
      "step_type": "THINK|CODE|ANALYZE|VALIDATE|SUMMARIZE",
      "title": "Short title",
      "description": "Exactly what to do",
      "rationale": "Why this step is needed",
      "expected_output": "What success looks like",
      "depends_on": []
    }}
  ]
}}"""

PLANNER_HUMAN = """USER GOAL: {goal}

DATASET INFORMATION:
{data_context}

Create the optimal analysis plan:"""

REPLANNER_SYSTEM = """You are replanning mid-execution because the situation changed.

Review what has been done, what was discovered, and create a REVISED plan
for the REMAINING steps only. Do not re-plan completed steps.

Keep the same JSON format. Be adaptive — if the data doesn't support
the original approach, pivot to what IS achievable."""

REPLANNER_HUMAN = """ORIGINAL GOAL: {goal}

CURRENT STATE:
{state_summary}

REASON FOR REPLANNING: {reason}

Create revised plan for remaining steps:"""


class Planner:
    """
    Creates and revises execution plans for the agent.

    Two modes:
    1. Initial planning: full plan from scratch
    2. Replanning: revise mid-execution based on new information
    """

    def __init__(self):
        self.settings= get_settings()
        self.llm =  ChatGroq(
            model=self.settings.planner_model,
            temperature=0.0, # this makes model consistent
            api_key=self.settings.groq_api_key,
        )

        self.initial_prompt = ChatPromptTemplate.from_messages([
            ("system" , PLANNER_SYSTEM),
            ("human", PLANNER_HUMAN),
        ])
        self.replan_prompt = ChatPromptTemplate.from_messages([
            ("system", REPLANNER_SYSTEM),
            ("human", REPLANNER_HUMAN),
        ])



    def create_plan(self, goal: str, data_context:Optional[DataContext]= None) -> Plan:
        """
        Create initial execution plan.

        Args:
            goal: What the user wants to know/do
            data_context: Description of available data (columns, shape, sample)

        Returns:
            Plan with ordered list of Steps
        """
        logger.info(f"Creating plan for goal: {goal[:80]}...")
        data_desc = self._format_data_context(data_context)

        response = self.llm.invoke(
            self.initial_prompt.format_messages(
                goal=goal,
                data_context=data_desc
            )
        )

        plan = self._parse_plan_response(response.content, goal)
        plan_len =len(plan.steps)
        estimated_complexity =plan.estimated_complexity
        requires_clarification =plan.requires_clarification
        logger.info(
            f"Plan created: {plan_len} steps, "
            f"complexity={estimated_complexity}, "
            f"needs_clarification={requires_clarification}"
        )
        return plan

    def replan(self, state: AgentState, reason: str) -> Plan:
        """
        Revise the plan mid-execution.

        Called when:
        - A step fails and the approach needs to change
        - Data doesn't match original assumptions
        - New information discovered changes the analysis direction

        Args:
            state: Current agent state (what's been done, findings so far)
            reason: Why replanning is needed

        Returns:
            Revised Plan for remaining steps
        """
        logger.info(f"Replanning: {reason}")

        response = self.llm.invoke(
            self.replan_prompt.format_messages(
                goal=state.goal,
                state_summary=state.get_context_summary(),
                reason=reason,
            )
        )

        revised_plan = self._parse_plan_response(response.content, state.goal)
        logger.info(f"Revised plan: {len(revised_plan.steps)} remaining steps")
        return revised_plan

    def assess_if_replan_needed(self, state: AgentState, last_result) -> tuple[bool, str]:

        """
        Decide if the current plan is still valid after a step result.

        Returns (should_replan, reason)

        Checks for:
        - Multiple consecutive failures
        - Data structure mismatch (expected column doesn't exist)
        - Confidence score too low to continue current approach
        """

        # too many errors in row
        recent_results = state.step_results[-3:] if len(state.step_results) >= 3 else state.step_results
        recent_failures = sum(1 for r in recent_results if r.status.value== "failed")

        # if recent failures more than 2 worng approch
        if recent_failures >= 2:
            return True, f"Multiple consecutive failures suggest wrong approach: {[r.error for r in recent_results if r.error]}"

        # Very low confidence — findings aren't reliable
        if last_result and last_result.confidence_score < 0.3:
            return True, f"Confidence too low ({last_result.confidence_score:.2f}): {last_result.interpretation[:100]}"

        return False, ""

    # format data

    def _format_data_context(self, ctx: Optional[DataContext]) -> str:

        if not ctx:
            return "No structured dataset provided. Work with general knowledge."

        lines = [
            f"File: {ctx.source} ({ctx.file_type.upper()})",
        ]
        if ctx.shape:
            lines.append(f"Shape: {ctx.shape[0]:,} rows × {ctx.shape[1]} columns")
        if ctx.columns:
            lines.append(f"Columns: {', '.join(ctx.columns)}")
        if ctx.dtypes:
            lines.append(f"Data types: {ctx.dtypes}")
        if ctx.sample:
            lines.append(f"\nSample (first 3 rows):\n{ctx.sample}")
        if ctx.stats:
            lines.append(f"\nStatistics:\n{ctx.stats}")

        return "\n".join(lines)

    def _parse_plan_response(self, response_text: str, goal: str) -> Plan:

        # Strip markdown code blocks if present
        text = re.sub(r'```json\n?|\n?```', '', response_text).strip()

        # Find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            logger.warning("Could not find JSON in planner response, using fallback plan")
            return self._fallback_plan(goal)

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, using fallback plan")
            return self._fallback_plan(goal)

       # Build Steps
        steps = []
        for s in data.get("steps", []):
            try:
                step_type = StepType[s.get("step_type", "THINK").upper()]
            except KeyError:
                step_type = StepType.THINK

            step = Step(
                step_id=s.get("step_id", f"s{len(steps)+1}"),
                step_type=step_type,
                title=s.get("title", "Step"),
                description=s.get("description", ""),
                rationale=s.get("rationale", ""),
                expected_output=s.get("expected_output", ""),
                depends_on=s.get("depends_on", []),
                status=StepStatus.PENDING,
            )
            steps.append(step)


        return Plan(
            goal=goal,
            steps=steps,
            reasoning=data.get("reasoning", ""),
            estimated_complexity=data.get("estimated_complexity", "medium"),
            requires_clarification=data.get("requires_clarification", False),
            clarification_questions=data.get("clarification_questions", []),
        )


    def _fallback_plan(self, goal: str) -> Plan:
        """
        Minimal plan when JSON parsing fails.
        """
        return Plan(
            goal=goal,
            reasoning="Fallback plan due to parsing error",
            steps=[
                Step(
                    step_id="s1", step_type=StepType.THINK,
                    title="Analyze goal",
                    description=f"Think through how to approach: {goal}",
                    rationale="Need to understand the task before executing",
                    expected_output="Clear analysis approach"
                ),
                Step(
                    step_id="s2", step_type=StepType.CODE,
                    title="Explore data",
                    description="Load data and perform exploratory analysis",
                    rationale="Need to understand data structure",
                    expected_output="Data summary and initial insights",
                    depends_on=["s1"]
                ),
                Step(
                    step_id="s3", step_type=StepType.SUMMARIZE,
                    title="Report findings",
                    description="Summarize all findings",
                    rationale="Deliver answer to user",
                    expected_output="Complete analysis report",
                    depends_on=["s2"]
                ),
            ]
        )
