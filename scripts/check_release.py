"""Offline consistency checks for the published research outputs (stdlib only)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "forecasting"
HOME = ROOT / "home-advantage"


def load_json(path: Path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value {value} in {path}")
    return json.loads(path.read_text(), parse_constant=invalid)


def close(actual, expected, tolerance=5.1e-6):
    if not math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0):
        raise AssertionError(f"{actual} does not match {expected}")


def check_forecast():
    result = load_json(FORECAST / "output/results.json")
    with (FORECAST / "output/tables/t1_pooled_metrics.csv").open() as stream:
        table = list(csv.DictReader(stream))
    assert {r["model"] for r in table} == set(result["pooled"])
    for row in table:
        model = row["model"]
        for metric in ("rps", "logloss", "brier", "acc", "n"):
            close(row[metric], result["pooled"][model][metric])
        assert int(float(row["n"])) == (14280 if model == "blend" else 16106)
        assert 0 <= float(row["rps"]) <= 1
        assert 0 <= float(row["acc"]) <= 1
    with (FORECAST / "output/tables/t2_rps_by_season.csv").open() as stream:
        for row in csv.DictReader(stream):
            model = row.pop("model")
            for season, score in row.items():
                if score:
                    close(score, result["rps_by_season"][model][season])
    assert result["market_on_blend_sample"]["n"] == result["pooled"]["blend"]["n"]
    close(result["pooled"]["blend"]["rps"] - result["market_on_blend_sample"]["rps"],
          result["dm_blend_vs_market"]["mean_gap"], tolerance=1.1e-5)
    rois = [entry["roi"] for model in result["backtest"].values()
            for threshold, entry in model.items() if threshold.startswith("theta_")]
    assert len(rois) == 16 and all(value < 0 for value in rois)
    close(min(rois), -0.1382091917591125, tolerance=0.0001)
    close(max(rois), -0.041223571749887535, tolerance=0.0001)
    print("PASS forecasting tables, samples, matched blend comparison and ROI range")


def check_home():
    result = load_json(HOME / "output/results.json")
    assert result["counts"]["matches"] == 21589
    expected = {"margin": -0.197755, "home_win": -0.050802, "yellow_diff": -0.247010}
    for outcome, value in expected.items():
        estimate = result["main"][outcome]["coefs"]["ghost"]
        close(estimate["coef"], value, tolerance=1e-5)
        assert estimate["ci_lo"] < estimate["coef"] < estimate["ci_hi"]
        assert 0 <= estimate["p"] <= 1
    with (HOME / "output/tables/table4_s1920.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    # Check the public CSV against the two-way inference reported in the README.
    labels = {"Goal margin (H-A)": "margin", "Home win (LPM)": "home_win",
              "Yellow cards (A-H)": "yellow_diff"}
    for row in rows:
        outcome = labels[row["Outcome"]]
        values = result["s1920_twoway"][outcome]
        assert int(row["N"]) == values["n"]
        ghost = row["ghost"]
        se = re.search(r"\(([0-9.]+)\)", ghost)
        assert se, f"Missing standard error in {ghost}"
        close(se.group(1), values["coefs"]["ghost"]["se"], tolerance=0.00051)
    print("PASS home-advantage estimates and within-season inference table")


def check_manifest():
    manifest = load_json(ROOT / "data/source_manifest.json")
    seasons = ("1415", "1516", "1617", "1718", "1819", "1920",
               "2021", "2122", "2223", "2324", "2425", "2526")
    expected = {f"{league}_{season}.csv" for league in ("E0", "D1", "SP1", "I1", "F1")
                for season in seasons}
    files = manifest["files"]
    assert len(files) == 60 and {f["file"] for f in files} == expected
    for item in files:
        assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        assert item["bytes"] > 0
        assert urlparse(item["url"]).hostname == "www.football-data.co.uk"
    print("PASS all 60 input source URLs and snapshot fingerprints")


def check_documentation():
    failures = []
    paths = [ROOT / "README.md", ROOT / "DATA_SOURCES.md"]
    paths += list((ROOT / "docs").glob("*.md"))
    paths += [FORECAST / "README.md", HOME / "README.md"]
    for path in paths:
        text = path.read_text()
        if "[Author Name]" in text or "/Users/" in text:
            failures.append(f"Unfinished or local-only content: {path.relative_to(ROOT)}")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            target = target.strip().split(' "')[0].strip("<>")
            if urlparse(target).scheme or target.startswith("#"):
                continue
            target = unquote(target.split("#")[0])
            if target and not (path.parent / target).exists():
                failures.append(f"Broken link in {path.relative_to(ROOT)}: {target}")
    assert not failures, "\n".join(failures)
    for project in (HOME, FORECAST):
        for image in (project / "output/figures").glob("*.png"):
            assert image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", str(image)
    print("PASS local documentation links, author fields and figure files")


def main():
    for check in (check_forecast, check_home, check_manifest, check_documentation):
        check()
    print("All offline release checks passed. Full fitting is a separate pipeline run.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, ValueError, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
