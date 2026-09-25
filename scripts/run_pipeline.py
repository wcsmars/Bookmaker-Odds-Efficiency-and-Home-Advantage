"""Run the research scripts with the current Python, from any directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STEPS = {
    "home-advantage": ("home-advantage", (
        "02_clean.py", "03_analysis.py", "04_figures.py")),
    "forecasting": ("forecasting", (
        "01_features.py", "02_models.py", "03_evaluate.py",
        "04_backtest.py", "05_figures.py")),
}


def check_inputs(require_snapshot: bool) -> None:
    manifest = json.loads((ROOT / "data/source_manifest.json").read_text(encoding="utf-8"))
    raw = ROOT / "home-advantage/data/raw"
    missing, changed = [], []
    for item in manifest["files"]:
        path = raw / item["file"]
        if not path.is_file() or not path.stat().st_size:
            missing.append(item["file"])
        elif require_snapshot and hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            changed.append(item["file"])
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} source files under {raw}. "
            "See DATA_SOURCES.md for input acquisition. No files were downloaded.\n"
            + ", ".join(missing))
    if changed:
        raise SystemExit("Source snapshot differs from the recorded research inputs: "
                         + ", ".join(changed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["all", *STEPS], default="all")
    parser.add_argument("--verify-snapshot", action="store_true",
                        help="Require all source-file hashes to match the release manifest.")
    args = parser.parse_args()
    check_inputs(args.verify_snapshot)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, "1")
    start = time.monotonic()
    studies = list(STEPS) if args.study == "all" else [args.study]
    for study in studies:
        project, scripts = STEPS[study]
        for script in scripts:
            path = ROOT / project / "src" / script
            print(f"\nRunning {project}/src/{script}", flush=True)
            subprocess.run([sys.executable, "-u", str(path)], cwd=ROOT,
                           env=env, check=True)
    print(f"\nCompleted {args.study} in {time.monotonic() - start:.1f} seconds.")


if __name__ == "__main__":
    main()
