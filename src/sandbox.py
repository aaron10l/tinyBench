"""Subprocess-based Python sandbox for LLM tool calling."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SANDBOX_TIMEOUT = 30
SANDBOX_MAX_OUTPUT = 4_000

_VENV_PYTHON = Path(__file__).parent.parent / ".venv" / "bin" / "python3"


class PythonSandbox:
    """Execute untrusted Python code in a subprocess with df pre-loaded."""

    def __init__(self, timeout: int = SANDBOX_TIMEOUT, max_output: int = SANDBOX_MAX_OUTPUT) -> None:
        self.timeout = timeout
        self.max_output = max_output
        self._python = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

    def ping(self) -> bool:
        """Return True if the sandbox subprocess can be launched successfully."""
        try:
            result = subprocess.run(
                [self._python, "-c", "print('ok')"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0 and result.stdout.strip() == "ok"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def run(self, code: str, csv_path: Path | str) -> str:
        """Run code with df pre-loaded from csv_path. Always returns a string."""
        setup = (
            "import pandas as pd\nimport numpy as np\nimport scipy\n"
            f"df = pd.read_csv({repr(str(csv_path))})\n"
        )
        # Auto-print trailing expression, REPL-style (models write `result` not `print(result)`)
        import ast
        try:
            tree = ast.parse(code)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                rest = ast.unparse(ast.Module(body=tree.body[:-1], type_ignores=[]))
                last = ast.unparse(tree.body[-1].value)
                exec_rest = f"exec(compile({repr(rest)}, '<model_code>', 'exec'))\n" if rest.strip() else ""
                auto_print = f"_result = ({last})\nif _result is not None: print(_result)\n"
                wrapper = setup + exec_rest + auto_print
            else:
                wrapper = setup + f"exec(compile({repr(code)}, '<model_code>', 'exec'))\n"
        except SyntaxError:
            wrapper = setup + f"exec(compile({repr(code)}, '<model_code>', 'exec'))\n"
        try:
            result = subprocess.run(
                [self._python, "-c", wrapper],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {self.timeout}s"
        except FileNotFoundError:
            return f"ERROR: Python not found at {self._python}"

        if result.returncode != 0:
            err = result.stderr.strip()
            if len(err) > self.max_output:
                err = err[: self.max_output] + "\n[truncated]"
            return f"ERROR:\n{err}"

        out = result.stdout
        if len(out) > self.max_output:
            out = out[: self.max_output] + "\n[output truncated]"
        return out or "(no output)"
