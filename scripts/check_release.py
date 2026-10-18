"""Offline consistency checks for the published research outputs (stdlib only).

Two layers:

* Consistency checks that hold for any pipeline run: every CSV table agrees
  with its result JSON, expected files exist, values are finite, standard
  errors are positive, the input manifest is well formed and documentation
  links resolve. `check_consistency(root)` runs them on any repository copy,
  including the synthetic-input copy built by tests/smoke_pipeline.py.
* Expectations for the published release: sample sizes and headline estimates.

Run from anywhere: `python scripts/check_release.py`.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]

LEAGUES = ("E0", "D1", "SP1", "I1", "F1")
SEASONS = ("1415", "1516", "1617", "1718", "1819", "1920",
           "2021", "2122", "2223", "2324", "2425", "2526")
TEST_SEASONS = ("2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
                "2022-23", "2023-24", "2024-25", "2025-26")
PERIODS = ("pre", "ghost", "post")
STANDALONE_MODELS = ("elo", "logit", "gbm", "dpois")
BACKTEST_MODELS = ("logit", "gbm", "dpois", "blend")
THETAS = ("theta_0.0", "theta_0.02", "theta_0.05", "theta_0.1")
FORECAST_FIGURES = ("fig1_rps_by_season", "fig2_reliability", "fig3_market_bias",
                    "fig4_importance", "fig5_economics")
HOME_FIGURES = ("fig1_rolling", "fig2_seasons", "fig3_referee", "fig4_coefplot")
# Coefficient tables: file -> (result section, regressors, outcome labels)
OUTCOME_LABELS = {
    "margin": "Goal margin (H-A)", "points_diff": "Points margin (H-A)",
    "home_win": "Home win (LPM)", "yellow_diff": "Yellow cards (A-H)",
    "foul_diff": "Fouls (A-H)", "red_diff": "Red cards (A-H)",
    "shots_diff": "Shots (H-A)", "sot_diff": "Shots on target (H-A)",
    "corner_diff": "Corners (H-A)",
}
POST_DUMMIES = ("post_2122", "post_2223", "post_2324", "post_2425", "post_2526")
# Gradient-boosting inputs (forecasting/src/02_models.py), the rows of t4
GBM_FEATURES = ("d_ppg5", "d_ppg10", "d_gf5", "d_ga5", "d_sotf5", "d_sota5", "d_cor5",
                "d_vppg", "d_vgd", "d_rest", "d_played", "elo_home", "elo_away", "elo_diff",
                "h_ppg5", "a_ppg5", "h_vppg", "a_vppg", "h_hist_n", "a_hist_n",
                *(f"lg_{league}" for league in LEAGUES))
# Text that must never reach a published Markdown file. The last marker flags a
# documentation step that only a rerun on the real source data can complete.
FORBIDDEN_TEXT = ("[Author Name]", "/Users/")
PENDING_MARKER = "REGENERATION PENDING"
COEF_TABLES = {
    "table2_main": ("main", ("ghost", "post"), tuple(OUTCOME_LABELS)),
    "table3_recovery": ("recovery", ("ghost", *POST_DUMMIES),
                        ("margin", "home_win", "yellow_diff")),
    "table4_strict": ("strict", ("ghost_strict", "ghost_loose", "post"),
                      ("margin", "home_win", "yellow_diff")),
    "table4_s1920": ("s1920", ("ghost",), ("margin", "home_win", "yellow_diff")),
    "table4_placebo": ("placebo", ("fake_ghost",), ("margin", "home_win", "yellow_diff")),
}
COEF_CELL = re.compile(r"^(-?\d+\.\d{3})(\*{0,3}) \((\d+\.\d{3})\)$")


class CheckFailure(Exception):
    """A release check found an inconsistency."""


def require(condition, message: str) -> None:
    """Fail with a message. Unlike `assert`, this survives `python -O`."""
    if not condition:
        raise CheckFailure(message)


def load_json(path: Path):
    def invalid(value):
        raise CheckFailure(f"Non-finite JSON value {value} in {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def read_table(path: Path) -> list[dict]:
    require(path.is_file(), f"Missing table {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows, f"Empty table {path}")
    return rows


def close(actual, expected, tolerance=5.1e-6, what="value") -> None:
    a, e = float(actual), float(expected)
    if not (math.isfinite(a) and math.isfinite(e)):
        raise CheckFailure(f"{what}: non-finite comparison {actual} vs {expected}")
    if not math.isclose(a, e, abs_tol=tolerance, rel_tol=0):
        raise CheckFailure(f"{what}: {actual} does not match {expected}")


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def check_values(node, path: str, allowed_null=frozenset()) -> None:
    """Every number in a result file is finite, nulls appear only where expected,
    every standard error is positive, every p-value lies in [0, 1] and every
    confidence interval contains its estimate."""
    if node is None:
        require(path in allowed_null, f"{path}: unexpected null value")
    elif isinstance(node, float):
        require(math.isfinite(node), f"{path}: non-finite value")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            check_values(value, f"{path}[{i}]", allowed_null)
    elif isinstance(node, dict):
        for key, value in node.items():
            check_values(value, f"{path}/{key}", allowed_null)
        if "se" in node:
            require(finite(node["se"], f"{path}/se") > 0, f"{path}: standard error must be positive")
        if "p" in node:
            require(0 <= finite(node["p"], f"{path}/p") <= 1, f"{path}: p-value out of range")
        if {"coef", "ci_lo", "ci_hi"} <= set(node):
            require(node["ci_lo"] < node["coef"] < node["ci_hi"], f"{path}: interval excludes estimate")


def finite(value, what: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{what}: expected a finite number, got {value!r}")
    return float(value)


# --------------------------------------------------------------------------
# Consistency layer: valid for any pipeline run
# --------------------------------------------------------------------------

def check_forecast_tables(root: Path = ROOT) -> None:
    project = root / "forecasting"
    result = load_json(project / "output/results.json")
    tables = project / "output/tables"
    for key in ("pooled", "rps_by_season", "rps_by_period", "ece", "market_bias_by_season",
                "market_on_blend_sample", "top_features", "backtest", "closing_vs_prematch",
                "demargin_robustness", "blend_gap_by_season"):
        require(key in result, f"forecasting results.json lacks '{key}'")
    # The blend starts a season later, so its first test-season score is the only gap.
    check_values(result, "forecasting", allowed_null={"forecasting/rps_by_season/blend/2017-18"})
    pooled = result["pooled"]
    require(set(pooled) == {*STANDALONE_MODELS, "market", "blend"},
            f"Unexpected model set in pooled metrics: {sorted(pooled)}")

    # t1: pooled metrics
    t1 = read_table(tables / "t1_pooled_metrics.csv")
    require({r["model"] for r in t1} == set(pooled), "t1 model rows differ from results.json")
    for row in t1:
        model = row["model"]
        for metric in ("rps", "logloss", "brier", "acc", "n"):
            close(row[metric], pooled[model][metric], what=f"t1 {model} {metric}")
        n = float(row["n"])
        require(n == int(n) and n > 0, f"t1 {model}: n must be a positive integer")
        require(0 <= float(row["rps"]) <= 1 and 0 <= float(row["acc"]) <= 1,
                f"t1 {model}: scores out of range")
    require(pooled["blend"]["n"] < pooled["market"]["n"],
            "The blend must cover a shorter sample than the market")
    require(result["market_on_blend_sample"]["n"] == pooled["blend"]["n"],
            "market_on_blend_sample must use the blend's sample size")

    # t2: RPS by season; a blank cell is valid only where the JSON holds null
    by_season = result["rps_by_season"]
    t2 = read_table(tables / "t2_rps_by_season.csv")
    require({r["model"] for r in t2} == set(by_season), "t2 model rows differ from results.json")
    seasons = [c for c in t2[0] if c != "model"]
    require(tuple(seasons) == TEST_SEASONS, f"t2 seasons {seasons} differ from {TEST_SEASONS}")
    for row in t2:
        model = row["model"]
        require(set(by_season[model]) == set(seasons), f"t2 {model}: season keys differ")
        for season in seasons:
            saved = by_season[model][season]
            if row[season] == "":
                require(saved is None, f"t2 {model} {season}: blank cell but JSON has {saved}")
            else:
                close(row[season], saved, what=f"t2 {model} {season}")
        blanks = sum(row[s] == "" for s in seasons)
        require(blanks == (1 if model == "blend" else 0),
                f"t2 {model}: {blanks} blank cells (blend has exactly one, others none)")

    # t3: RPS by crowd period, including the market on the blend's sample
    by_period = result["rps_by_period"]
    t3 = read_table(tables / "t3_rps_by_period.csv")
    require({r["model"] for r in t3} == set(by_period) == {*pooled, "market_blend_sample"},
            "t3 rows must be every model plus market_blend_sample")
    for row in t3:
        for period in PERIODS:
            close(row[period], by_period[row["model"]][period],
                  what=f"t3 {row['model']} {period}")

    # Clustered score-difference tests agree with the pooled means
    for model in STANDALONE_MODELS:
        close(result[f"dm_{model}_vs_market"]["mean_gap"],
              pooled[model]["rps"] - pooled["market"]["rps"], tolerance=1.1e-5,
              what=f"dm_{model}_vs_market mean gap")
        require(result[f"dm_{model}_vs_market"]["n"] == pooled[model]["n"],
                f"dm_{model}_vs_market n differs from pooled n")
    close(result["dm_blend_vs_market"]["mean_gap"],
          pooled["blend"]["rps"] - result["market_on_blend_sample"]["rps"], tolerance=1.1e-5,
          what="dm_blend_vs_market mean gap")
    require(set(result["blend_gap_by_season"]) == set(TEST_SEASONS[1:]),
            "blend_gap_by_season must cover the eight blend seasons")
    for season, gap in result["blend_gap_by_season"].items():
        close(gap["mean_gap"], by_season["blend"][season] - by_season["market"][season],
              tolerance=1.1e-5, what=f"blend gap {season}")
        require(0 <= finite(gap["p"], f"blend gap {season} p") <= 1, f"blend gap {season}: p out of range")

    # t4: one row per model feature, sorted, with the top ten mirrored in the JSON
    t4 = read_table(tables / "t4_permutation_importance.csv")
    require(sorted(r["feature"] for r in t4) == sorted(GBM_FEATURES),
            "t4 must list each model feature exactly once")
    importances = [finite(float(r["importance"]), f"t4 {r['feature']} importance") for r in t4]
    require(all(finite(float(r["std"]), f"t4 {r['feature']} std") >= 0 for r in t4),
            "t4 standard deviations must be non-negative")
    require(importances == sorted(importances, reverse=True), "t4 is not sorted by importance")
    top = result["top_features"]
    require(len(top) == 10 and len(t4) >= 10, "top_features must hold the ten leading rows")
    for row, saved in zip(t4, top):
        require(row["feature"] == saved["feature"], "t4 feature order differs from top_features")
        close(row["importance"], saved["importance"], tolerance=1e-9, what=f"t4 {row['feature']}")
        close(row["std"], saved["std"], tolerance=1e-9, what=f"t4 {row['feature']} std")

    # t5: market home-win bias by season, both de-margining conventions
    bias = result["market_bias_by_season"]
    t5 = read_table(tables / "t5_market_bias_by_season.csv")
    require([r["season"] for r in t5] == [b["season"] for b in bias] == list(TEST_SEASONS),
            "t5 seasons differ from results.json or the test seasons")
    for row, saved in zip(t5, bias):
        for column in ("implied", "realized", "bias", "ci_lo", "ci_hi",
                       "implied_basic", "bias_basic", "ci_lo_basic", "ci_hi_basic"):
            close(row[column], saved[column], tolerance=1e-9, what=f"t5 {row['season']} {column}")
        require(int(row["n"]) == saved["n"] > 0, f"t5 {row['season']}: n mismatch")
        for suffix in ("", "_basic"):
            close(saved["realized"] - saved["implied" + suffix], saved["bias" + suffix],
                  tolerance=1.1e-4, what=f"t5 {row['season']} bias{suffix} identity")
            require(saved["ci_lo" + suffix] <= saved["bias" + suffix] <= saved["ci_hi" + suffix],
                    f"t5 {row['season']}: bias{suffix} outside its interval")

    # Calibration tables: bins, sample sizes and the saved ECE values
    for model in ("market", "gbm", "blend"):
        for label, column in (("home", "pH"), ("away", "pA")):
            rows = read_table(tables / f"cal_{model}_{label}.csv")
            require([int(r["bin"]) for r in rows] == list(range(len(rows))) and 2 <= len(rows) <= 10,
                    f"cal_{model}_{label}: bins are not sequential")
            n = [int(r["n"]) for r in rows]
            require(sum(n) == pooled[model]["n"] and min(n) > 0,
                    f"cal_{model}_{label}: bin sizes must be positive and sum to the sample")
            means = [float(r["p_mean"]) for r in rows]
            rates = [float(r["o_rate"]) for r in rows]
            require(all(0 <= v <= 1 for v in means + rates), f"cal_{model}_{label}: value outside [0, 1]")
            require(all(a < b for a, b in zip(means, means[1:])),
                    f"cal_{model}_{label}: quantile-bin means must increase")
            ece = sum(k * abs(float(r["p_mean"]) - float(r["o_rate"])) for k, r in zip(n, rows)) / sum(n)
            close(ece, result["ece"][f"{model}_{label}_ece"], tolerance=1e-6,
                  what=f"cal_{model}_{label} ECE")

    # Backtest grid
    backtest = result["backtest"]
    for model in BACKTEST_MODELS:
        for theta in THETAS:
            cell = backtest[model][theta]
            n_bets = cell["n_bets"]
            require(isinstance(n_bets, int) and n_bets > 0, f"backtest {model} {theta}: no bets")
            roi = finite(cell["roi"], f"backtest {model} {theta} roi")
            require(cell["ci_lo"] <= roi <= cell["ci_hi"], f"backtest {model} {theta}: ROI outside CI")
            require(finite(cell["mean_odds"], f"backtest {model} {theta} mean_odds") > 1,
                    f"backtest {model} {theta}: mean odds must exceed one")
    for model in ("gbm", "blend"):
        require(set(backtest[model]["by_period"]) == set(PERIODS), f"backtest {model}: periods")
        require(sum(v["n_bets"] for v in backtest[model]["by_period"].values())
                == backtest[model]["theta_0.05"]["n_bets"],
                f"backtest {model}: by-period bet counts do not sum to the theta=0.05 count")
        require(0 < backtest[model]["clv"]["n"] <= backtest[model]["theta_0.05"]["n_bets"],
                f"backtest {model}: closing-line sample exceeds the bet count")
    require(0 < finite(backtest["mean_overround_test"], "mean overround") < 0.3,
            "Mean test-sample overround is implausible")
    closing = result["closing_vs_prematch"]
    require(closing["n"] > 0 and closing["n"] <= pooled["market"]["n"], "closing sample size")
    print("PASS forecasting tables, calibration, score tests and backtest grid agree with results.json")


def check_home_tables(root: Path = ROOT) -> None:
    project = root / "home-advantage"
    result = load_json(project / "output/results.json")
    tables = project / "output/tables"
    for key in ("descriptives", "league_margin_by_period", "season_stats", "main", "recovery",
                "recovery_clean", "strict", "s1920", "placebo", "by_league", "pooled_wild",
                "bookmaker", "counts"):
        require(key in result, f"home-advantage results.json lacks '{key}'")

    # Descriptive tables (rounded to 3 or 4 decimals in the CSV)
    t1 = read_table(tables / "table1_descriptives.csv")
    require([r["period"] for r in t1] == list(PERIODS), "table1 periods")
    for row in t1:
        for column, values in result["descriptives"].items():
            close(row[column], values[row["period"]], tolerance=5.1e-4,
                  what=f"table1 {row['period']} {column}")
    require(sum(int(float(r["Matches"])) for r in t1) == result["counts"]["matches"],
            "table1 match counts do not sum to counts.matches")
    a1 = read_table(tables / "tableA1_league_margin.csv")
    require({r["league"] for r in a1} == set(result["league_margin_by_period"]["pre"])
            and len(a1) == 5, "tableA1 leagues")
    for row in a1:
        for period in PERIODS:
            close(row[period], result["league_margin_by_period"][period][row["league"]],
                  tolerance=5.1e-4, what=f"tableA1 {row['league']} {period}")
    stats = read_table(tables / "season_stats.csv")
    require([r["season"] for r in stats] == list(result["season_stats"]["margin"]) and len(stats) == 12,
            "season_stats seasons")
    for row in stats:
        for column in ("home_win", "margin", "yellow_diff", "implied"):
            close(row[column], result["season_stats"][column][row["season"]], tolerance=5.1e-5,
                  what=f"season_stats {row['season']} {column}")

    # Coefficient tables: 'coef*** (se)' cells against the estimates
    for name, (section, regressors, outcomes) in COEF_TABLES.items():
        rows = read_table(tables / f"{name}.csv")
        require([r["Outcome"] for r in rows] == [OUTCOME_LABELS[o] for o in outcomes],
                f"{name}: outcome rows")
        require(set(rows[0]) == {"Outcome", "N", "home_adv_const", *regressors}, f"{name}: columns")
        for row, outcome in zip(rows, outcomes):
            spec = result[section][outcome]
            require(int(row["N"]) == spec["n"], f"{name} {outcome}: N")
            for regressor in ("home_adv_const", *regressors):
                match = COEF_CELL.match(row[regressor])
                require(match, f"{name} {outcome} {regressor}: unreadable cell {row[regressor]!r}")
                saved = spec["coefs"][regressor]
                close(match.group(1), saved["coef"], tolerance=5.1e-4, what=f"{name} {outcome} {regressor}")
                close(match.group(3), saved["se"], tolerance=5.1e-4, what=f"{name} {outcome} {regressor} se")
                require(match.group(2) == stars(saved["p"]), f"{name} {outcome} {regressor}: stars")

    # Every estimate and contrast: positive standard error, valid p-value, finite values
    check_values(result, "home-advantage")
    require(set(result["by_league"]) == set(result["league_margin_by_period"]["pre"]),
            "by_league must cover the five leagues")
    for league, specs in result["by_league"].items():
        for outcome, spec in specs.items():
            require(spec["n_clusters"] >= 100, f"by_league {league} {outcome}: too few clusters")
    for key, p in result["pooled_wild"].items():
        require(0 < finite(p, f"pooled_wild {key}") <= 1, f"pooled_wild {key}: p out of range")
    print("PASS home-advantage descriptive and coefficient tables agree with results.json")


def check_figures(root: Path = ROOT) -> None:
    expected = [root / "forecasting/output/figures" / f"{n}.png" for n in FORECAST_FIGURES]
    expected += [root / "home-advantage/output/figures" / f"{n}.png" for n in HOME_FIGURES]
    for image in expected:
        require(image.is_file(), f"Missing figure {image.relative_to(root)}")
        data = image.read_bytes()
        head = data[:24]
        require(head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR"
                and data.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82"),
                f"Not a complete PNG file: {image.relative_to(root)}")
        width, height = int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
        require(width > 100 and height > 100, f"Implausible image size in {image.relative_to(root)}")
    print(f"PASS all {len(expected)} figure files are present PNG images")


def check_manifest(root: Path = ROOT) -> None:
    manifest = load_json(root / "data/source_manifest.json")
    files = manifest["files"]
    expected = {f"{league}_{season}.csv" for league in LEAGUES for season in SEASONS}
    require(len(files) == 60 and {f["file"] for f in files} == expected,
            "Manifest must list exactly the 60 league-season files")
    for item in files:
        league, season = item["file"][:-4].split("_")
        require(item["url"] == f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv",
                f"Manifest URL for {item['file']} does not match its file name")
        require(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), f"Bad digest for {item['file']}")
        require(isinstance(item["bytes"], int) and item["bytes"] > 0, f"Bad size for {item['file']}")
    require(len({f["sha256"] for f in files}) == 60, "Manifest digests are not unique")
    print("PASS all 60 input source URLs and snapshot fingerprints")


def tracked_files(root: Path):
    """Paths git tracks under root, or None when git is absent or root is not a checkout.

    Whether root is a checkout is decided from the folder structure, not from
    git's (possibly translated) messages. Inside a checkout any git failure, for
    example one git refuses as having dubious ownership, is an error, so the
    tracked-files guarantee never lapses silently.
    """
    if not any((folder / ".git").exists() for folder in (root, *root.parents)):
        return None  # a downloaded archive or the smoke-test copy
    try:
        listing = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                                 capture_output=True, check=True,
                                 env={**os.environ, "LC_ALL": "C", "LANGUAGE": "C"})
    except OSError:
        return None
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", "replace").strip()
        raise CheckFailure(f"git could not list tracked files: {message}") from error
    tracked = {(root / p).resolve() for p in listing.stdout.decode("utf-8").split("\0") if p}
    return tracked or None


def check_documentation(root: Path = ROOT, release: bool = True) -> None:
    """Local links resolve to published files; no placeholder or local-only text.

    In a git checkout only tracked Markdown is scanned and a link counts only
    when its target is tracked, so private or ignored files never satisfy a
    link that would break once the repository is published. With
    release=False (synthetic smoke runs) the pending-regeneration marker is
    allowed, since only a real-data rerun can resolve it.
    """
    forbidden = FORBIDDEN_TEXT + ((PENDING_MARKER,) if release else ())
    root = root.resolve()
    tracked = tracked_files(root)

    def published(path: Path) -> bool:
        if tracked is None:
            return path.exists()
        if path in tracked:
            return True
        return path.is_dir() and any(path in t.parents for t in tracked)

    if tracked is None:
        paths = sorted(p for p in root.rglob("*.md")
                       if not any(part.startswith(".") for part in p.relative_to(root).parts))
    else:
        paths = sorted(p for p in tracked if p.suffix == ".md")
    require(paths, "No Markdown documentation found")
    failures = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                failures.append(f"Unfinished or local-only content ({marker}): {path.relative_to(root)}")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            target = target.strip().split(' "')[0].strip("<>")
            if urlparse(target).scheme or target.startswith("#"):
                continue
            target = unquote(target.split("#")[0])
            if target and not published((path.parent / target).resolve()):
                failures.append(f"Broken or unpublished link in {path.relative_to(root)}: {target}")
    require(not failures, "\n".join(failures))
    print(f"PASS local documentation links in {len(paths)} Markdown files resolve to published files")


def check_consistency(root: Path = ROOT, release: bool = True) -> None:
    for check in (check_forecast_tables, check_home_tables, check_figures, check_manifest):
        check(root)
    check_documentation(root, release=release)


# --------------------------------------------------------------------------
# Published release: expected sample sizes and headline estimates
# --------------------------------------------------------------------------

def check_published_forecast() -> None:
    result = load_json(ROOT / "forecasting/output/results.json")
    pooled = result["pooled"]
    for model in (*STANDALONE_MODELS, "market"):
        require(pooled[model]["n"] == 16106, f"{model}: expected 16,106 test matches")
    require(pooled["blend"]["n"] == 14280, "blend: expected 14,280 test matches")
    # The market benchmark depends only on the recorded source snapshot, not on the models.
    close(pooled["market"]["rps"], 0.19511, what="market RPS")
    close(result["market_on_blend_sample"]["rps"], 0.19578, what="market RPS on the blend sample")
    for model in STANDALONE_MODELS:
        require(pooled["market"]["rps"] < pooled[model]["rps"],
                f"The market must beat {model} in pooled RPS")
    # Model-dependent values (RPS, ROI range, test statistics) are tied to the
    # README text by tests/test_release.py, so they need no constants here.
    rois = [result["backtest"][model][theta]["roi"]
            for model in BACKTEST_MODELS for theta in THETAS]
    require(len(rois) == 16 and all(value < 0 for value in rois), "Every ROI must be negative")
    print("PASS forecasting sample sizes, market benchmark, ranking and negative returns")


def check_published_home() -> None:
    result = load_json(ROOT / "home-advantage/output/results.json")
    require(result["counts"]["matches"] == 21589, "Expected 21,589 matches")
    expected = {"margin": -0.197755, "home_win": -0.050802, "yellow_diff": -0.247010}
    for outcome, value in expected.items():
        estimate = result["main"][outcome]["coefs"]["ghost"]
        close(estimate["coef"], value, tolerance=1e-5, what=f"main {outcome} ghost")
        require(estimate["ci_hi"] < 0, f"main {outcome} ghost: interval must exclude zero")
    print("PASS home-advantage sample size and closed-period estimates")


def main() -> None:
    check_consistency(ROOT)
    check_published_forecast()
    check_published_home()
    print("All offline release checks passed. Full fitting is a separate pipeline run.")


if __name__ == "__main__":
    try:
        main()
    except (CheckFailure, KeyError, ValueError, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
