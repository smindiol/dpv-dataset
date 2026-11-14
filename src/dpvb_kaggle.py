"""Kaggle entry point for dataset B (callable shape: runs in the runner's own process, so progress streams live
to the kernel log). The runner puts <clone>/src on sys.path; dpvgen lives at the clone root."""
import os
import subprocess
import sys
from pathlib import Path


def main(material, n=500, max_extra=100):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["JAX_PLATFORMS"] = "cpu"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "git+https://github.com/romanodev/deltapv@f874e24"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"], check=True)
    from dpvgen import generacion_b
    return generacion_b.run(material, n=n, max_extra=max_extra, salida=os.getcwd())
