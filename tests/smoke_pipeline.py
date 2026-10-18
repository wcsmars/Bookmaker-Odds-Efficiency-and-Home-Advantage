"""Run both studies end to end on synthetic inputs in a temporary copy.

    python tests/smoke_pipeline.py [--seed N] [--keep]

Needs the research dependencies from requirements.txt. The repository is copied
to a temporary directory whose path contains a space, 60 synthetic
league-season files are generated there, scripts/run_pipeline.py runs from an
unrelated working directory, and the regenerated outputs must pass the
consistency layer of scripts/check_release.py. The committed outputs are never
touched. The synthetic data-generating process is described in
tests/make_synthetic_inputs.py; the run checks the code, not the findings.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
COPIED = ("forecasting/src", "home-advantage/src", "scripts", "data", "tests")
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc")


def publishable_files():
    """Tracked plus untracked-but-not-ignored files, or None outside a git checkout.

    Copying only these keeps private, git-ignored notes in a working folder out
    of the smoke copy, so they cannot affect its documentation check.
    """
    try:
        listing = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    files = [ROOT / p for p in listing.decode("utf-8").split("\0") if p]
    return [f for f in files if f.is_file()] or None


def make_copy(copy: Path) -> None:
    files = publishable_files()
    if files is None:
        for item in COPIED:
            shutil.copytree(ROOT / item, copy / item, ignore=IGNORED)
        for top in ROOT.iterdir():
            if top.is_file() and not top.name.startswith("."):
                shutil.copy(top, copy / top.name)
        (copy / "docs").mkdir()
        for doc in (ROOT / "docs").glob("*.md"):
            shutil.copy(doc, copy / "docs" / doc.name)
        for project in ("forecasting", "home-advantage"):
            shutil.copy(ROOT / project / "README.md", copy / project / "README.md")
        return
    for source in files:
        target = copy / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py"
                                                  if name != "check_release"
                                                  else ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep", action="store_true", help="Keep the temporary copy.")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="smoke run "))
    copy = tmp / "repo copy"
    try:
        make_copy(copy)
        make_inputs = load("make_synthetic_inputs")
        raw = copy / "home-advantage/data/raw"
        n = make_inputs.write_inputs(raw, args.seed)
        print(f"Synthetic inputs: {n} matches in {raw}")

        start = time.monotonic()
        subprocess.run([sys.executable, str(copy / "scripts/run_pipeline.py")],
                       cwd=tmp, check=True)
        print(f"\nPipeline finished in {time.monotonic() - start:.0f} seconds.")

        check = load("check_release")
        check.check_consistency(copy, release=False)
        forecast = check.load_json(copy / "forecasting/output/results.json")
        home = check.load_json(copy / "home-advantage/output/results.json")
        print(json.dumps({
            "matches": home["counts"]["matches"],
            "test_matches": forecast["pooled"]["market"]["n"],
            "market_rps": forecast["pooled"]["market"]["rps"],
            "ghost_margin": round(home["main"]["margin"]["coefs"]["ghost"]["coef"], 4),
        }))
        print("Smoke test passed: both studies ran on synthetic inputs and the outputs are consistent.")
    finally:
        if args.keep:
            print(f"Kept {copy}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
