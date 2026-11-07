import base64  # Capturing matplotlib figures (converted to base64 PNG)
import contextlib
import io
import logging
import signal
import sys
import textwrap
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tools.guardrails import (  # noqa: F401  (BLOCKED/ALLOWED_IMPORTS re-exported for compatibility)
    ALLOWED_IMPORTS,
    BLOCKED_IMPORTS,
    check_code,
    check_code_policy,
    is_blocked_attr,
)

logger = logging.getLogger(__name__)


class SandboxTimeout(BaseException):
    """Raised inside sandboxed code when the time limit is hit.

    Deliberately a BaseException (not an Exception): generated code that wraps a loop in
    `try/except Exception` must not be able to swallow the timeout and keep running.
    """


# Exception classes sandboxed code may catch (LLM-written code uses try/except routinely)
_SAFE_EXCEPTIONS = (
    "Exception",
    "ValueError",
    "KeyError",
    "TypeError",
    "IndexError",
    "LookupError",
    "AttributeError",
    "ArithmeticError",
    "ZeroDivisionError",
    "OverflowError",
    "FloatingPointError",
    "AssertionError",
    "RuntimeError",
    "NotImplementedError",
    "StopIteration",
)


@dataclass
class ExecutionResult:
    """
    Full result of a code execution attempt.
    Captures everything: output, figures, errors, timing.
    """

    success: bool
    stdout: str = ""  # printed text output
    stderr: str = ""  # error output
    error: Optional[str] = None  # formatted error message
    error_type: Optional[str] = None  # e.g. "KeyError", "TypeError"
    error_line: Optional[int] = None  # line number of error
    figures: List[str] = field(default_factory=list)  # base64 PNG strings
    dataframes: List[Dict] = field(default_factory=list)  # {name, html, shape}
    variables: Dict[str, Any] = field(default_factory=dict)  # captured variables
    execution_time_ms: float = 0.0
    code_executed: str = ""


class CodeExecutor:
    """
    Sandboxed Python code executor for data analysis tasks.

    Usage:
        executor = CodeExecutor()
        result = executor.execute(code_string, data_context={"df": my_dataframe})
    """

    def __init__(self, timeout_seconds: int = 30, use_e2b: bool = False):
        self.timeout = timeout_seconds
        self.use_e2b = use_e2b

        # Try to import E2B if requested
        self.e2b_sandbox = None
        if use_e2b:
            try:
                from e2b_code_interpreter import Sandbox

                self.E2BSandbox = Sandbox
                logger.info("E2B cloud sandbox enabled")
            except ImportError:
                logger.warning("e2b-code-interpreter not installed. Falling back to local sandbox.")
                self.use_e2b = False

    # decide to use e2b or local
    def execute(
        self,
        code: str,
        data_context: Optional[Dict[str, Any]] = None,
        capture_vars: Optional[List[str]] = None,
        step_id: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute Python code in a sandbox.

        Every call passes the guardrails first (static policy, optional LLM judge, audit log).
        A blocked call returns ExecutionResult(success=False, error_type="SecurityError") instead of
        raising, so the agent's self-correction loop can rewrite the code.

        Args:
            code: Python code string to execute
            data_context: Dict of variables to inject (e.g., {"df": dataframe})
            capture_vars: Variable names to capture from execution namespace
            step_id: Optional plan step id, recorded in the audit log

        Returns:
            ExecutionResult with output, figures, errors
        """
        verdict = check_code(code, step_id=step_id)
        if not verdict.allowed:
            error_type = "SyntaxError" if verdict.kind == "syntax" else "SecurityError"
            return ExecutionResult(
                success=False,
                error=verdict.reason if error_type == "SyntaxError" else f"Blocked by guardrails: {verdict.reason}",
                error_type=error_type,
                code_executed=code,
            )

        if self.use_e2b:
            return self._execute_e2b(code, data_context)
        else:
            return self._execute_local(code, data_context, capture_vars)

    def validate_code(self, code: str) -> tuple[bool, str]:
        """
        Static analysis: check code for dangerous patterns before running.
        Returns (is_safe, reason_if_unsafe). Pure check: no LLM call, no audit log entry.
        """
        result = check_code_policy(code)
        return result.allowed, result.reason

    def _execute_local(
        self,
        code: str,
        data_context: Optional[Dict] = None,
        capture_vars: Optional[List[str]] = None,
    ) -> ExecutionResult:
        """
        Local sandbox execution using restricted namespace.

        Injects matplotlib figure capture and dataframe capture automatically.
        """
        t0 = time.time()

        # Static validation (execute() already ran the full guardrails; this keeps direct callers safe too)
        is_safe, reason = self.validate_code(code)
        if not is_safe:
            return ExecutionResult(
                success=False,
                error=reason,
                error_type="SyntaxError" if reason.startswith("SyntaxError") else "SecurityError",
                code_executed=code,
            )

        # Build execution namespace
        # We inject safe versions of common modules
        import matplotlib
        import numpy as np
        import pandas as pd

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        namespace = {
            "__builtins__": self._safe_builtins(),
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
            "plt": plt,
            "matplotlib": matplotlib,
            "sns": sns,
            "seaborn": sns,
            "print": print,
        }

        # Try to inject scipy, sklearn if available
        # Inject scipy
        try:
            import scipy
            from scipy import stats

            namespace["scipy"] = scipy
            namespace["stats"] = stats
        except ImportError:
            pass

        # Inject sklearn submodules directly
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                GradientBoostingRegressor,
                RandomForestClassifier,
                RandomForestRegressor,
            )
            from sklearn.feature_selection import f_classif, mutual_info_classif, mutual_info_regression
            from sklearn.inspection import permutation_importance
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.metrics import (
                accuracy_score,
                classification_report,
                confusion_matrix,
                mean_squared_error,
                r2_score,
            )
            from sklearn.model_selection import cross_val_score, train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

            namespace.update(
                {
                    "sklearn": __import__("sklearn"),
                    "train_test_split": train_test_split,
                    "LogisticRegression": LogisticRegression,
                    "LinearRegression": LinearRegression,
                    "RandomForestClassifier": RandomForestClassifier,
                    "RandomForestRegressor": RandomForestRegressor,
                    "StandardScaler": StandardScaler,
                    "LabelEncoder": LabelEncoder,
                    "accuracy_score": accuracy_score,
                    "classification_report": classification_report,
                    "confusion_matrix": confusion_matrix,
                    "mean_squared_error": mean_squared_error,
                    "r2_score": r2_score,
                    "KMeans": KMeans,
                    "PCA": PCA,
                    "GradientBoostingClassifier": GradientBoostingClassifier,
                    "GradientBoostingRegressor": GradientBoostingRegressor,
                    "DecisionTreeClassifier": DecisionTreeClassifier,
                    "DecisionTreeRegressor": DecisionTreeRegressor,
                    "permutation_importance": permutation_importance,
                    "mutual_info_classif": mutual_info_classif,
                    "mutual_info_regression": mutual_info_regression,
                    "f_classif": f_classif,
                    "cross_val_score": cross_val_score,
                    "Pipeline": Pipeline,
                }
            )
        except ImportError:
            pass

        # Inject user data — always copy DataFrames to prevent steps from
        # corrupting shared state (e.g. pd.get_dummies modifying df in-place)
        if data_context:
            safe_context = {}
            for key, val in data_context.items():
                if isinstance(val, pd.DataFrame):
                    safe_context[key] = val.copy()  # fresh copy every step
                else:
                    safe_context[key] = val
            namespace.update(safe_context)

        # Capture stdout
        stdout_capture = io.StringIO()
        figures = []
        dataframes = []

        # Add figure-capture wrapper
        preamble = textwrap.dedent(
            """
        import warnings
        warnings.filterwarnings('ignore')
        # Auto-capture: any plt.show() will be intercepted
        """
        )

        full_code = preamble + "\n" + code

        try:
            with contextlib.redirect_stdout(stdout_capture):
                with self._timeout(self.timeout):
                    exec(full_code, namespace)

            # Capture any open matplotlib figures
            import matplotlib.pyplot as plt

            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
                buf.seek(0)
                figures.append(base64.b64encode(buf.read()).decode("utf-8"))
                plt.close(fig)

            # Capture DataFrames in namespace
            for var_name, val in namespace.items():
                if var_name.startswith("_"):
                    continue
                if isinstance(val, pd.DataFrame) and len(val) > 0:
                    dataframes.append(
                        {
                            "name": var_name,
                            "shape": list(val.shape),
                            "columns": list(val.columns),
                            "data": val.head(20).to_dict(orient="records"),
                            "html": val.head(10).to_html(classes="dataframe", border=0),
                        }
                    )

            # Capture requested variables
            captured_vars = {}
            if capture_vars:
                for var in capture_vars:
                    if var in namespace:
                        captured_vars[var] = namespace[var]

            return ExecutionResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                figures=figures,
                dataframes=dataframes,
                variables=captured_vars,
                execution_time_ms=(time.time() - t0) * 1000,
                code_executed=code,
            )

        except (TimeoutError, SandboxTimeout):
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {self.timeout} seconds",
                error_type="TimeoutError",
                stdout=stdout_capture.getvalue(),
                execution_time_ms=(time.time() - t0) * 1000,
                code_executed=code,
            )

        except Exception as e:
            tb = traceback.format_exc()
            error_type = type(e).__name__
            # Extract line number from traceback
            error_line = self._extract_error_line(tb)

            # Generate helpful error message with suggestions
            error_msg = self._format_error(e, tb, code)

            return ExecutionResult(
                success=False,
                error=error_msg,
                error_type=error_type,
                error_line=error_line,
                stderr=tb,
                stdout=stdout_capture.getvalue(),
                execution_time_ms=(time.time() - t0) * 1000,
                code_executed=code,
            )

    def _safe_builtins(self) -> dict:
        """Return a safe subset of Python builtins."""
        safe = [
            "abs",
            "all",
            "any",
            "bin",
            "bool",
            "bytes",
            "callable",
            "chr",
            "dict",
            "divmod",
            "enumerate",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "hasattr",
            "hash",
            "hex",
            "id",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "map",
            "max",
            "min",
            "next",
            "oct",
            "ord",
            "pow",
            "print",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "str",
            "sum",
            "super",
            "tuple",
            "type",
            "zip",
            "True",
            "False",
            "None",
        ]
        import builtins

        results = {name: getattr(builtins, name) for name in safe if hasattr(builtins, name)}
        for exc_name in _SAFE_EXCEPTIONS:
            results[exc_name] = getattr(builtins, exc_name)

        # `vars`, `dir`, `help`, `eval`, `exec`, `compile`, `open`, `globals`, `locals` are deliberately absent.
        # The remaining reflective builtins are replaced by guarded versions, and the import hook
        # only lets allow-listed modules through (the static policy alone cannot see __import__('os')).
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level != 0:
                raise ImportError("Relative imports are not allowed in the sandbox")
            if name.split(".")[0] not in ALLOWED_IMPORTS:
                raise ImportError(f"Import of '{name}' is not allowed in the sandbox")
            return real_import(name, globals, locals, fromlist, level)

        def guarded_getattr(obj, name, *default):
            if is_blocked_attr(name) or name.startswith("_"):
                raise AttributeError(f"Access to attribute '{name}' is not allowed in the sandbox")
            return getattr(obj, name, *default)

        def guarded_hasattr(obj, name):
            if is_blocked_attr(name) or name.startswith("_"):
                return False
            return hasattr(obj, name)

        def guarded_setattr(obj, name, value):
            if is_blocked_attr(name) or name.startswith("_"):
                raise AttributeError(f"Access to attribute '{name}' is not allowed in the sandbox")
            return setattr(obj, name, value)

        results["__import__"] = guarded_import
        results["getattr"] = guarded_getattr
        results["hasattr"] = guarded_hasattr
        results["setattr"] = guarded_setattr

        return results

    @contextlib.contextmanager
    def _timeout(self, seconds: int):
        """Context manager for execution timeout using SIGALRM (Unix only).

        Uses an interval timer so the timeout keeps firing every second after the first expiry,
        and raises SandboxTimeout (a BaseException) so `except Exception` in generated code
        cannot swallow it.
        """

        def handler(signum, frame):
            raise SandboxTimeout(f"Execution exceeded {seconds}s limit")

        # Only the *setup* is wrapped for AttributeError (platforms without SIGALRM, e.g. Windows).
        # The body must stay outside it: an AttributeError raised by the generated code used to be
        # caught here and re-yielded, surfacing as "RuntimeError: generator didn't stop after throw()".
        timer_armed = False
        old_trace = None
        trace_armed = False
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGALRM, handler)
                signal.setitimer(signal.ITIMER_REAL, seconds, 1.0)
                timer_armed = True
            except AttributeError:
                logger.warning("SIGALRM unavailable on this platform - code runs without a time limit")
        else:
            # Signals only work in the main thread (e.g. a threadpool worker or test client).
            # Fall back to a deadline checked on every line of the generated code. It cannot
            # interrupt one long C call (a huge sort, say), but stops loops in generated Python.
            deadline = time.monotonic() + seconds

            def local_trace(frame, event, arg):
                if time.monotonic() > deadline:
                    raise SandboxTimeout(f"Execution exceeded {seconds}s limit")
                return local_trace

            def global_trace(frame, event, arg):
                return local_trace if frame.f_code.co_filename == "<string>" else None

            old_trace = sys.gettrace()
            sys.settrace(global_trace)
            trace_armed = True

        try:
            yield
        finally:
            if timer_armed:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if trace_armed:
                sys.settrace(old_trace)

    def _extract_error_line(self, traceback_str: str) -> Optional[int]:
        """Extract the line number from a traceback string."""
        import re

        matches = re.findall(r"line (\d+)", traceback_str)
        if matches:
            return int(matches[-1])
        return None

    def _format_error(self, error: Exception, tb: str, code: str) -> str:
        """Format an error with context and suggestions."""
        error_type = type(error).__name__
        base_msg = f"{error_type}: {str(error)}"

        # Add helpful suggestions for common errors
        suggestions = {
            "KeyError": "Column name may not exist. Check df.columns to see available columns.",
            "AttributeError": "Variable may not be the expected type. Check variable names.",
            "ValueError": "Data type mismatch or invalid operation. Check data types with df.dtypes.",
            "NameError": "Variable not defined. Make sure to load data before using it.",
            "TypeError": "Wrong type passed to function. Check argument types.",
            "ImportError": "Module not available in sandbox. Use pre-imported: pd, np, plt, sns.",
        }
        suggestion = suggestions.get(error_type, "Review the code logic above.")
        return f"{base_msg}\n\nSuggestion: {suggestion}"

        # ─── E2B CLOUD SANDBOX ────────────────────────────────────────────────────

    def _execute_e2b(self, code: str, data_context: Optional[Dict] = None) -> ExecutionResult:
        """
        Execute code in E2B cloud sandbox.
        More secure than local execution for production.
        """
        t0 = time.time()
        try:
            with self.E2BSandbox() as sandbox:
                # If data context contains DataFrames, serialize and inject them
                setup_code = ""
                if data_context:
                    setup_code = self._build_data_injection_code(data_context)

                full_code = setup_code + "\n" + code
                execution = sandbox.run_code(full_code)

                figures = []
                for result in execution.results:
                    if result.png:
                        figures.append(result.png)

                return ExecutionResult(
                    success=not execution.error,
                    stdout=execution.logs.stdout or "",
                    stderr=execution.logs.stderr or "",
                    error=execution.error,
                    figures=figures,
                    execution_time_ms=(time.time() - t0) * 1000,
                    code_executed=code,
                )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=(time.time() - t0) * 1000,
                code_executed=code,
            )

    def _build_data_injection_code(self, data_context: Dict) -> str:
        """Serialize DataFrames to CSV strings for injection into E2B sandbox."""
        import pandas as pd

        lines = ["import pandas as pd", "import io"]
        for name, val in data_context.items():
            if isinstance(val, pd.DataFrame):
                csv_str = val.to_csv(index=False).replace("'", "\\'")
                lines.append(f"{name} = pd.read_csv(io.StringIO('''{csv_str}'''))")
        return "\n".join(lines)
