"""Run submitted models in dependency order and keep generated files in outputs/."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["all", "q1", "q2", "q3", "q4", "validate"])
    args = parser.parse_args()
    jobs = {"q1": "问题1.py", "q2": "问题2.py", "q3": "问题3.py", "q4": "问题4.py"}
    if args.target == "all":
        scripts = list(jobs.values())
    elif args.target == "validate":
        if not (ROOT / "outputs/q4/q4_summary.txt").is_file():
            parser.error("Run python run.py all before validate; Q4 validation reads its baseline summary.")
        scripts = ["问题3数值验证.py", "问题4数值验证.py"]
    else:
        scripts = [jobs[args.target]]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    log_dir = ROOT / "outputs/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for script in scripts:
        print(f"Running {script}", flush=True)
        with (log_dir / (Path(script).stem + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"{script} failed ({result.returncode}); see outputs/logs.")
        print(f"Finished {script}", flush=True)

if __name__ == "__main__":
    main()
