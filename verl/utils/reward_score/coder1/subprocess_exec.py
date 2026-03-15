import os
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory

from .utils import _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS


def code_exec_subprocess(code, stdin=None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest=None):
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]

    if pytest:
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)
            result = subprocess.run(
                ["python3", "-m", "pytest", tmpdir],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=timeout,
                check=False,
            )
    else:
        with NamedTemporaryFile(suffix=".py", mode="w", delete=True) as tmp:
            tmp.write(code)
            tmp.flush()
            result = subprocess.run(
                ["python3", tmp.name],
                input=stdin.encode() if stdin else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=timeout,
                check=False,
            )

    stderr = result.stderr.decode().strip()
    stdout = result.stdout.decode()

    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"