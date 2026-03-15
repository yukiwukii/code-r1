import subprocess, tempfile
from pathlib import Path
from .utils import _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS

IMAGE = "/home/users/ntu/elim078/scratch/code-r1-yuki/scipy.sif" 

def code_exec_singularity(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS):
    # memory_limit_prefix = "import resource\nresource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))\n"
    memory_limit_prefix = "import os\nos.environ['OPENBLAS_NUM_THREADS'] = '1'\nos.environ['OMP_NUM_THREADS'] = '1'\nos.environ['MKL_NUM_THREADS'] = '1'\nimport resource\nresource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))\n"

    code = memory_limit_prefix + code
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        script = td / "script.py"
        script.write_text(code)

        cmd = ["singularity", "exec", "--containall", "--cleanenv",
       "--bind", f"{td}:/work",
       IMAGE,
       "python", "/work/script.py"]

        try:
            p = subprocess.run(
                cmd,
                input=(stdin if stdin is not None else None),
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            logs = (p.stdout or "") + (p.stderr or "")
            return (p.returncode == 0), (logs if p.returncode == 0 else _ERROR_MSG_PREFIX + logs)
        except Exception as e:
            return False, _ERROR_MSG_PREFIX + str(e)
