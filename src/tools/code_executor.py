import ast
import base64  # Capturing matplotlib figures (converted to base64 PNG)
import contextlib
import io
import logging
import signal
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Imports that are dangerous in sandboxed environments
BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "urllib",
    "requests",
    "httpx",
    "ftplib",
    "smtplib",
    "pickle",
    "shelve",
    "importlib",
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",  # file system
}

# Safe imports that data analysis code needs
ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scipy",
    "sklearn",
    "plotly",
    "statsmodels",
    "json",
    "math",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "re",
    "string",
    "random",
    "statistics",
    "warnings",
}


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
    ) -> ExecutionResult:
        """
        Execute Python code in a sandbox.

        Args:
            code: Python code string to execute
            data_context: Dict of variables to inject (e.g., {"df": dataframe})
            capture_vars: Variable names to capture from execution namespace

        Returns:
            ExecutionResult with output, figures, errors
        """
        if self.use_e2b:
            return self._execute_e2b(code, data_context)
        else:
            return self._execute_local(code, data_context, capture_vars)

    def validate_code(self, code: str) -> tuple[bool, str]:
        """
        Static analysis: check code for dangerous patterns before running.
        Returns (is_safe, reason_if_unsafe)
        """
        # Check for syntax errors first
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

        # Walk AST looking for dangerous patterns
        for node in ast.walk(tree):
            # Block dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base_module = alias.name.split(".")[0]
                    if base_module in BLOCKED_IMPORTS:
                        return False, f"Blocked import: '{alias.name}' is not allowed in sandbox"

            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in BLOCKED_IMPORTS:
                    return False, f"Blocked import: 'from {node.module}' is not allowed"

            # Block direct exec/eval calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "compile"}:
                    return False, f"Direct call to '{node.func.id}' is not allowed"

        return True, ""

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

        # Static validation

        is_safe, reason = self.validate_code(code)
        if not is_safe:
            return ExecutionResult(
                success=False,
                error=reason,
                error_type="SecurityError",
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
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.metrics import (
                accuracy_score,
                classification_report,
                confusion_matrix,
                mean_squared_error,
                r2_score,
            )
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import LabelEncoder, StandardScaler

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

        except TimeoutError:
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
            "dir",
            "divmod",
            "enumerate",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "hasattr",
            "hash",
            "help",
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
            "vars",
            "zip",
            "True",
            "False",
            "None",
        ]
        import builtins

        results = {name: getattr(builtins, name) for name in safe if hasattr(builtins, name)}
        results["__import__"] = builtins.__import__

        return results

    @contextlib.contextmanager
    def _timeout(self, seconds: int):
        """Context manager for execution timeout using SIGALRM (Unix only)."""

        def handler(signum, frame):
            raise TimeoutError(f"Execution exceeded {seconds}s limit")

        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            yield
        except AttributeError:
            yield
        finally:
            try:
                signal.alarm(0)
            except AttributeError:
                pass

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
