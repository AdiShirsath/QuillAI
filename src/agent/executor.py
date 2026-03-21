import json
import logging
import re
import time
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.agent.models import AgentState, Step, StepResult, StepStatus, StepType
from src.configs.settings import get_settings
from src.tools.code_executor import CodeExecutor, ExecutionResult

logger = logging.getLogger(__name__)

## Prompts:

CODE_WRITER_SYSTEM = """You are an expert Python data scientist.

Write clean, executable Python code to complete the given analysis step.

RULES:
1. Do NOT include any import statements — everything is pre-imported.
2. The dataframe is available as 'df' if data was provided.
3. Print key results explicitly: print(f"Result: {{value:.3f}}")
4. Create clear, labeled matplotlib figures for visualizations. Never call plt.show().
5. Add comments explaining what each section does.
6. Handle edge cases: check if columns exist before using them.
7. Output ONLY the Python code — no explanation, no markdown, no code fences.
8. Never modify 'df' directly. Always work on a copy: df_work = df.copy()
9. Before calling train_test_split, always verify:
   print(f"Class distribution: {{y.value_counts().to_dict()}}`")
   assert y.nunique() >= 2, "Target must have at least 2 classes"
10. Never call pd.get_dummies on 'df' directly. Use: df_encoded = pd.get_dummies(df.copy(), ...)

PRE-AVAILABLE VARIABLES & MODULES:
- Data         : df (main DataFrame)
- Core         : pd, np
- Visualization: plt, sns
- Stats        : stats (scipy.stats)
- sklearn      : train_test_split, LogisticRegression, LinearRegression,
                 RandomForestClassifier, RandomForestRegressor,
                 StandardScaler, LabelEncoder, Pipeline,
                 accuracy_score, classification_report,
                 confusion_matrix, mean_squared_error, r2_score

CONTEXT:
{context_summary}

DATA AVAILABLE:
{data_info}

PREVIOUS FINDINGS:
{previous_findings}
"""
CODE_WRITER_HUMAN = """STEP: {step_title}
DESCRIPTION: {step_description}
EXPECTED OUTPUT: {expected_output}

Write the Python code:"""

CODE_FIXER_SYSTEM = """You are debugging Python code that failed.

Analyze the error carefully and fix the code.
Output ONLY the corrected Python code, no explanation.

IMPORTANT:
- If KeyError: check available columns and use the correct ones
- If AttributeError: check variable types
- If the error is fundamental (wrong approach), rewrite from scratch
- Keep the same goal, just fix the implementation"""

CODE_FIXER_HUMAN = """ORIGINAL CODE:
```python
{code}
```

ERROR:
{error}

AVAILABLE DATA INFO:
{data_info}

Write the fixed Python code:"""


INTERPRETER_SYSTEM = """You are a senior data analyst interpreting code output.

Given the output of a data analysis step, provide:
1. A clear interpretation of what the results mean
2. Key findings (specific numbers, patterns, anomalies)
3. Confidence score (0.0-1.0) based on data quality and result clarity
4. Any caveats or limitations

Respond in JSON:
{{
  "interpretation": "detailed explanation of results",
  "key_findings": ["specific finding 1", "specific finding 2"],
  "confidence_score": 0.85,
  "caveats": ["caveat 1"],
  "suggests_followup": "optional: what to investigate next"
}}"""

INTERPRETER_HUMAN = """STEP: {step_title}
GOAL: {goal}

CODE OUTPUT:
{output}

FIGURES GENERATED: {n_figures}
DATAFRAMES CAPTURED: {n_dataframes}

Interpret these results:"""

THINKER_SYSTEM = """You are a data scientist doing analytical reasoning.

For THINK steps: reason through the problem carefully.
For ANALYZE steps: interpret data in context of the full goal.
For VALIDATE steps: critically assess if findings are statistically sound.
For SUMMARIZE steps: synthesize all findings into a coherent narrative.

Be specific with numbers when available. Flag uncertainty.

Respond in JSON:
{{
  "output": "your reasoning/analysis/summary",
  "key_findings": ["finding 1", "finding 2"],
  "confidence_score": 0.8,
  "next_actions": ["optional recommendation"]
}}"""

THINKER_HUMAN = """{step_description}

CONTEXT:
{context_summary}

KEY FINDINGS SO FAR:
{findings}

Provide your analysis:"""


class Executor:
    """
    Executes individual steps in the agent's plan.
    Handles CODE, THINK, ANALYZE, VALIDATE, and SUMMARIZE step types.
    Implements self-correction loop for failed code.
    """

    def __init__(self):
        # get all settings and llm ready
        self.settings = get_settings()

        self.llm = ChatGroq(
            model=self.settings.executor_model,
            temperature=0.1,  # Slight creativity for code writing
            api_key=self.settings.groq_api_key,
        )

        self.code_executor = CodeExecutor(
            timeout_seconds=30, use_e2b=bool(self.settings.e2b_api_key and not self.settings.use_local_sandbox)
        )

        # Prompts
        self.code_writer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CODE_WRITER_SYSTEM),
                ("human", CODE_WRITER_HUMAN),
            ]
        )
        self.code_fixer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CODE_FIXER_SYSTEM),
                ("human", CODE_FIXER_HUMAN),
            ]
        )
        self.interpreter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INTERPRETER_SYSTEM),
                ("human", INTERPRETER_HUMAN),
            ]
        )
        self.thinker_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", THINKER_SYSTEM),
                ("human", THINKER_HUMAN),
            ]
        )

    def execute_step(
        self,
        step: Step,
        state: AgentState,
        data_context: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """
        Execute a single step and return the result.

        Routes to appropriate handler based on step type.
        All handlers return a StepResult.
        """
        t0 = time.time()
        logger.info(f"Executing step [{step.step_id}] {step.step_type.value}: {step.title}")

        try:
            if step.step_type == StepType.CODE:
                result = self._execute_code_step(step, state, data_context)
            else:
                # THINK, ANALYZE, VALIDATE, SUMMARIZE all use the thinker
                result = self._execute_think_step(step, state)

            result.latency_ms = (time.time() - t0) * 1000
            return result

        except Exception as e:
            logger.error(f"Step {step.step_id} crashed: {e}", exc_info=True)
            return StepResult(
                step_id=step.step_id,
                step_type=step.step_type,
                status=StepStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=(time.time() - t0) * 1000,
            )

    def _execute_code_step(
        self,
        step: Step,
        state: AgentState,
        data_context: Optional[Dict] = None,
    ) -> StepResult:
        """
        Handle CODE steps with self-correction loop.

        Flow:
        1. Write code (LLM)
        2. Execute code (sandbox)
        3. If failed: fix code (LLM) and retry (up to max_retries)
        4. If succeeded: interpret output (LLM)
        5. Return StepResult
        """
        max_retries = self.settings.max_retries_per_step
        data_info = self._format_data_info(state.data_context, data_context)

        # Step 1: Write initial code does not excute it
        code = self._write_code(step, state, data_info)

        exec_result = None
        self_correction_applied = False
        correction_explanation = ""

        # Self-correction retry loop
        for attempt in range(max_retries):
            logger.debug(f"Code execution attempt {attempt + 1}/{max_retries}")

            # Step 2: Execute code using sandbox
            ## because of errors of imports lines in code we need to remove those
            # in sandbox they are already imported

            # Remove all top-level import statements
            code = re.sub(r"^\s*import .*$", "", code, flags=re.MULTILINE)
            code = re.sub(r"^\s*from .* import .*$", "", code, flags=re.MULTILINE)
            exec_result = self.code_executor.execute(
                code=code,
                data_context=data_context,
            )

            if exec_result.success:
                break  # Code worked!

            # Step 3: Self-correction
            if attempt < max_retries - 1:
                logger.info(
                    f"Step {step.step_id} attempt {attempt+1} failed " f"({exec_result.error_type}). Self-correcting..."
                )
                fixed_code = self._fix_code(code, exec_result.error, data_info)
                if fixed_code and fixed_code != code:
                    correction_explanation = (
                        f"Fixed {exec_result.error_type} on attempt {attempt+1}: "
                        f"{exec_result.error[:100] if exec_result.error else 'unknown error'}"
                    )
                    code = fixed_code
                    self_correction_applied = True
                else:
                    logger.warning("Self-correction produced no change. Stopping retries.")
                    break

        # Step 4: Interpret output
        if exec_result and exec_result.success:
            interpretation_data = self._interpret_output(
                step=step,
                goal=state.goal,
                exec_result=exec_result,
            )

            return StepResult(
                step_id=step.step_id,
                step_type=step.step_type,
                status=StepStatus.SUCCESS,
                code_written=code,
                code_output=exec_result.stdout,
                figures=exec_result.figures,
                dataframes=exec_result.dataframes,
                interpretation=interpretation_data.get("interpretation", ""),
                key_findings=interpretation_data.get("key_findings", []),
                confidence_score=interpretation_data.get("confidence_score", 0.7),
                self_correction_applied=self_correction_applied,
                correction_explanation=correction_explanation,
            )
        else:
            # All retries failed
            return StepResult(
                step_id=step.step_id,
                step_type=step.step_type,
                status=StepStatus.FAILED,
                code_written=code,
                code_output=exec_result.stdout if exec_result else "",
                error=exec_result.error if exec_result else "Unknown execution failure",
                error_type=exec_result.error_type if exec_result else "UnknownError",
                self_correction_applied=self_correction_applied,
                correction_explanation=correction_explanation,
            )

    def _write_code(self, step: Step, state: AgentState, data_info: str) -> str:
        """Ask LLM to write Python code for this step."""
        findings_str = "\n".join(f"- {f}" for f in state.key_findings[-5:]) or "None yet"

        response = self.llm.invoke(
            self.code_writer_prompt.format_messages(
                context_summary=state.get_context_summary(),
                data_info=data_info,
                previous_findings=findings_str,
                step_title=step.title,
                step_description=step.description,
                expected_output=step.expected_output,
            )
        )

        # Strip markdown code fences if present
        code = response.content
        code = re.sub(r"^```python\n?|^```\n?|\n?```$", "", code, flags=re.MULTILINE).strip()
        return code

    def _fix_code(self, code: str, error: Optional[str], data_info: str) -> Optional[str]:
        """Ask LLM to fix broken code given the error message."""
        response = self.llm.invoke(
            self.code_fixer_prompt.format_messages(
                code=code,
                error=error or "Unknown error",
                data_info=data_info,
            )
        )
        fixed = response.content
        fixed = re.sub(r"^```python\n?|^```\n?|\n?```$", "", fixed, flags=re.MULTILINE).strip()
        return fixed if fixed else None

    def _interpret_output(
        self,
        step: Step,
        goal: str,
        exec_result: ExecutionResult,
    ) -> Dict:
        """Ask LLM to interpret the output of a code execution."""
        output_text = exec_result.stdout or "No printed output"
        if len(output_text) > 3000:
            output_text = output_text[:3000] + "\n... [truncated]"

        response = self.llm.invoke(
            self.interpreter_prompt.format_messages(
                step_title=step.title,
                goal=goal,
                output=output_text,
                n_figures=len(exec_result.figures),
                n_dataframes=len(exec_result.dataframes),
            )
        )

        return self._parse_json_response(
            response.content,
            {
                "interpretation": output_text,
                "key_findings": [],
                "confidence_score": 0.6,
            },
        )

    def _execute_think_step(self, step: Step, state: AgentState) -> StepResult:
        """
        Handle THINK, ANALYZE, VALIDATE, SUMMARIZE steps.
        These are pure reasoning steps — no code execution.
        """
        findings_str = "\n".join(f"- {f}" for f in state.key_findings) or "No findings yet"

        # For SUMMARIZE, enrich with all code outputs
        description = step.description
        if step.step_type == StepType.SUMMARIZE:
            all_outputs = []
            for r in state.step_results:
                if r.status == StepStatus.SUCCESS and r.code_output:
                    all_outputs.append(f"[{r.step_id}] {r.interpretation}")
            if all_outputs:
                description += "\n\nAll analysis results to synthesize:\n" + "\n\n".join(all_outputs)

        response = self.llm.invoke(
            self.thinker_prompt.format_messages(
                step_description=description,
                context_summary=state.get_context_summary(),
                findings=findings_str,
            )
        )

        data = self._parse_json_response(
            response.content,
            {
                "output": response.content,
                "key_findings": [],
                "confidence_score": 0.7,
            },
        )

        return StepResult(
            step_id=step.step_id,
            step_type=step.step_type,
            status=StepStatus.SUCCESS,
            interpretation=data.get("output", ""),
            key_findings=data.get("key_findings", []),
            confidence_score=data.get("confidence_score", 0.7),
        )

    def _format_data_info(
        self,
        data_context,
        runtime_data: Optional[Dict] = None,
    ) -> str:
        """Format data information for code writer prompts."""
        lines = []
        if data_context:
            lines.append(f"DataFrame 'df': {data_context.shape}")
            lines.append(f"Columns: {data_context.columns}")
            lines.append(f"Dtypes: {data_context.dtypes}")
            if data_context.sample:
                lines.append(f"Sample:\n{data_context.sample}")
        elif runtime_data:
            import pandas as pd

            for name, val in runtime_data.items():
                if isinstance(val, pd.DataFrame):
                    lines.append(f"DataFrame '{name}': {val.shape}, columns={list(val.columns)}")
        return "\n".join(lines) or "No structured data available"

    def _parse_json_response(self, text: str, fallback: Dict) -> Dict:
        """Parse JSON from LLM response with fallback."""
        text = re.sub(r"```json\n?|\n?```", "", text).strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return fallback
