"""Offline tests for the saved results, release checks and pipeline runner (stdlib only).

Run from the repository root with: python -m unittest discover -s tests
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
STANDALONE_MODELS = ("elo", "logit", "gbm", "dpois")
MINUS = "−"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_release = load_script("check_release")
run_pipeline = load_script("run_pipeline")


class ReleaseChecks(unittest.TestCase):
    """Each offline release check passes on the included outputs."""

    def assert_passes(self, check, *args) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as output:
            check(*args)
        self.assertTrue(output.getvalue().startswith("PASS"), output.getvalue())

    def test_forecasting_tables_match_results(self):
        self.assert_passes(check_release.check_forecast_tables)

    def test_home_advantage_tables_match_results(self):
        self.assert_passes(check_release.check_home_tables)

    def test_all_figures_present(self):
        self.assert_passes(check_release.check_figures)

    def test_source_manifest_is_complete(self):
        self.assert_passes(check_release.check_manifest)

    def test_documentation_links_resolve(self):
        self.assert_passes(check_release.check_documentation)

    def test_published_expectations(self):
        self.assert_passes(check_release.check_published_forecast)
        self.assert_passes(check_release.check_published_home)


class ChecksDetectCorruption(unittest.TestCase):
    """The consistency layer rejects outputs that no longer agree with the JSON."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "copy"
        for item in ("forecasting/output", "home-advantage/output", "data"):
            shutil.copytree(ROOT / item, self.root / item)

    def tearDown(self):
        self.tmp.cleanup()

    def edit_csv(self, relative: str, row: int, column: str, value: str) -> None:
        path = self.root / relative
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows, fields = list(reader), reader.fieldnames
        rows[row][column] = value
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_changed_period_score_is_caught(self):
        self.edit_csv("forecasting/output/tables/t3_rps_by_period.csv", 0, "pre", "0.99999")
        with self.assertRaisesRegex(check_release.CheckFailure, "t3"):
            check_release.check_forecast_tables(self.root)

    def test_changed_coefficient_cell_is_caught(self):
        self.edit_csv("home-advantage/output/tables/table2_main.csv", 0, "ghost", "-0.098*** (0.050)")
        with self.assertRaisesRegex(check_release.CheckFailure, "table2_main margin ghost"):
            check_release.check_home_tables(self.root)

    def test_changed_calibration_table_is_caught(self):
        self.edit_csv("forecasting/output/tables/cal_market_home.csv", 0, "o_rate", "0.5")
        with self.assertRaisesRegex(check_release.CheckFailure, "cal_market_home"):
            check_release.check_forecast_tables(self.root)

    def test_non_finite_json_is_rejected(self):
        path = self.root / "home-advantage/output/results.json"
        path.write_text(path.read_text(encoding="utf-8").replace('"matches": 21589', '"matches": NaN'),
                        encoding="utf-8")
        with self.assertRaisesRegex(check_release.CheckFailure, "Non-finite"):
            check_release.check_home_tables(self.root)

    def test_missing_figure_is_caught(self):
        (self.root / "forecasting/output/figures/fig3_market_bias.png").unlink()
        with self.assertRaisesRegex(check_release.CheckFailure, "fig3_market_bias"):
            check_release.check_figures(self.root)

    def test_invalid_importance_row_is_caught(self):
        self.edit_csv("forecasting/output/tables/t4_permutation_importance.csv", 20, "std", "nan")
        with self.assertRaisesRegex(check_release.CheckFailure, "t4"):
            check_release.check_forecast_tables(self.root)

    def test_negative_contrast_standard_error_is_caught(self):
        path = self.root / "home-advantage/output/results.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        result["recovery"]["margin"]["contrast"]["se"] = -0.5
        path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(check_release.CheckFailure, "contrast: standard error"):
            check_release.check_home_tables(self.root)

    def test_truncated_figure_is_caught(self):
        path = self.root / "home-advantage/output/figures/fig2_seasons.png"
        path.write_bytes(path.read_bytes()[:64])
        with self.assertRaisesRegex(check_release.CheckFailure, "fig2_seasons"):
            check_release.check_figures(self.root)

    def test_manifest_url_must_match_file(self):
        path = self.root / "data/source_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files"][0]["url"] = "https://www.football-data.co.uk/mmz4281/1516/E0.csv"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(check_release.CheckFailure, "URL"):
            check_release.check_manifest(self.root)


class ReportedClaims(unittest.TestCase):
    """Headline statements in the READMEs agree with the saved results."""

    @classmethod
    def setUpClass(cls):
        def prose(path):  # Markdown with bold markers removed and line wraps joined
            return " ".join((ROOT / path).read_text(encoding="utf-8").replace("**", "").split())
        cls.forecast = check_release.load_json(ROOT / "forecasting/output/results.json")
        cls.home = check_release.load_json(ROOT / "home-advantage/output/results.json")
        cls.readme = prose("README.md")
        cls.forecast_readme = prose("forecasting/README.md")
        cls.home_readme = prose("home-advantage/README.md")

    def test_market_beats_every_standalone_model_pooled(self):
        pooled = self.forecast["pooled"]
        for model in STANDALONE_MODELS:
            self.assertLess(pooled["market"]["rps"], pooled[model]["rps"], model)

    def test_market_beats_every_standalone_model_each_season(self):
        by_season = self.forecast["rps_by_season"]
        seasons = by_season["market"]
        self.assertEqual(len(seasons), 9)
        for season, market_rps in seasons.items():
            for model in STANDALONE_MODELS:
                self.assertLess(market_rps, by_season[model][season], f"{model} {season}")

    def test_blend_is_compared_on_its_own_sample(self):
        blend, matched = self.forecast["pooled"]["blend"], self.forecast["market_on_blend_sample"]
        self.assertEqual(blend["n"], matched["n"])
        self.assertLess(blend["n"], self.forecast["pooled"]["market"]["n"])

    def test_every_betting_configuration_loses(self):
        grid = [entry for model in ("logit", "gbm", "dpois", "blend")
                for key, entry in self.forecast["backtest"][model].items()
                if key.startswith("theta_")]
        self.assertEqual(len(grid), 16)
        self.assertTrue(all(entry["roi"] < 0 for entry in grid))

    def test_closed_stadium_effects_exclude_zero(self):
        for outcome in ("margin", "home_win", "yellow_diff"):
            ghost = self.home["main"][outcome]["coefs"]["ghost"]
            self.assertLess(ghost["ci_hi"], 0, outcome)

    def test_readme_forecast_numbers_match_results(self):
        pooled = self.forecast["pooled"]
        for text in (self.readme, self.forecast_readme):
            for model in ("market", "logit", "gbm", "blend"):
                self.assertIn(f"{pooled[model]['rps']:.5f}", text, model)
            self.assertIn(f"{self.forecast['market_on_blend_sample']['rps']:.5f}", text)
            self.assertIn(f"{pooled['market']['n']:,.0f}", text)
            self.assertIn(f"{pooled['blend']['n']:,.0f}", text)
        rois = [entry["roi"] for model in ("logit", "gbm", "dpois", "blend")
                for key, entry in self.forecast["backtest"][model].items() if key.startswith("theta_")]
        roi_range = f"{MINUS}{-100 * min(rois):.1f}% to {MINUS}{-100 * max(rois):.1f}%"
        self.assertIn(roi_range, self.readme)
        self.assertIn(roi_range, self.forecast_readme)
        bias = {row["season"]: row for row in self.forecast["market_bias_by_season"]}
        for season in ("2020-21", "2024-25"):
            self.assertIn(f"{MINUS}{-100 * bias[season]['bias']:.2f} pp", self.forecast_readme, season)
            self.assertLess(bias[season]["ci_hi"], 0, season)
        clv = self.forecast["backtest"]["gbm"]["clv"]
        self.assertIn(f"+{100 * clv['mean_clv']:.2f}% on {clv['n']:,} selected bets", self.forecast_readme)

    def test_readme_forecast_inference_claims_match_results(self):
        """Model-dependent statements in the forecasting README follow the saved tests."""
        f, text = self.forecast, self.forecast_readme
        pooled = f["pooled"]
        standalone = {m: pooled[m]["rps"] for m in STANDALONE_MODELS}
        self.assertEqual(min(standalone, key=standalone.get), "logit",
                         "README calls the full-feature logit the best standalone model")
        self.assertIn("Best standalone model: full-feature logit", text)
        self.assertGreater(pooled["gbm"]["rps"], pooled["logit"]["rps"],
                           "README says gradient boosting is worse than the logit")
        logit = f["dm_logit_vs_market"]
        self.assertIn(f"gap +{logit['mean_gap']:.5f}, date-clustered t = {logit['t']:.1f}", text)
        blend = f["dm_blend_vs_market"]
        self.assertGreater(blend["mean_gap"], 0, "README says the blend is worse than the market")
        self.assertIn(f"gap +{blend['mean_gap']:.5f} (blend worse), unadjusted p = {blend['p']:.3f}", text)

        gaps = f["blend_gap_by_season"]
        favourable = sorted(s for s, g in gaps.items() if g["mean_gap"] < 0)
        self.assertTrue(all(gaps[s]["p"] >= 0.05 for s in favourable),
                        "README says no season shows a significant blend improvement")
        def join(items):  # "a", "a and b", "a, b and c"
            items = list(items)
            return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

        words = dict(enumerate(("no", "one", "two", "three", "four", "five", "six", "seven", "eight")))
        if len(favourable) == 1:
            self.assertIn("small favorable point estimate in one season", text)
        elif favourable:
            self.assertIn(f"small favorable point estimates in {words[len(favourable)]} seasons", text)
        else:
            self.assertNotIn("favorable point estimate", text)
        deficits = sorted(s for s, g in gaps.items() if g["mean_gap"] > 0 and g["p"] < 0.05)
        label = lambda s: s.replace("-", "–")  # README writes seasons with an en dash
        if deficits:
            seasons = join(label(s) for s in deficits)
            p_values = join(f"{gaps[s]['p']:.3f}" for s in deficits)
            noun, verb = ("deficits", "are") if len(deficits) > 1 else ("deficit", "is")
            self.assertIn(f"its {noun} in {seasons} {verb} nominally significant "
                          f"(unadjusted p = {p_values})", text)
        else:
            self.assertNotIn("nominally significant", text)

        cells = [entry for model in ("logit", "gbm", "dpois", "blend")
                 for key, entry in f["backtest"][model].items() if key.startswith("theta_")]
        self.assertTrue(any(c["ci_hi"] > 0 for c in cells),
                        "README says some ROI confidence intervals include zero")

        bias = {row["season"]: row for row in f["market_bias_by_season"]}
        basic = {round(100 * bias[s]["bias_basic"], 1) for s in ("2020-21", "2024-25")}
        self.assertEqual(len(basic), 1, "README gives one proportional-normalization value for both seasons")
        self.assertIn(f"about {MINUS}{-basic.pop():.1f} pp in both seasons with proportional "
                      "normalization", text)
        shift = sum(r["implied"] - r["implied_basic"] for r in bias.values()) / len(bias)
        self.assertIn(f"about {100 * shift:.1f} percentage points more probability", text)
        pre = [-100 * bias[s]["bias"] for s in ("2017-18", "2018-19", "2019-20")]
        self.assertIn(f"Shin biases of {MINUS}{min(pre):.1f} to {MINUS}{max(pre):.1f} pp", text)

    def test_readme_home_advantage_numbers_match_results(self):
        main = self.home["main"]
        ghost = main["margin"]["coefs"]["ghost"]
        expected = [
            f"{MINUS}{-ghost['coef']:.3f} goals",
            f"{MINUS}{-100 * main['home_win']['coefs']['ghost']['coef']:.1f} percentage points",
            f"+{main['margin']['coefs']['home_adv_const']['coef']:.3f} goals",
            f"{MINUS}{-main['yellow_diff']['coefs']['ghost']['coef']:.3f} cards",
            f"95% CI {MINUS}{-ghost['ci_lo']:.3f} to {MINUS}{-ghost['ci_hi']:.3f}",
            f"{self.home['counts']['matches']:,} matches",
        ]
        for claim in expected:
            self.assertIn(claim, self.home_readme, claim)
        self.assertIn(expected[0], self.readme)
        self.assertIn(expected[1], self.readme)
        recovery = self.home["recovery"]["margin"]["coefs"]
        self.assertIn(f"{MINUS}{-recovery['post_2425']['coef']:.3f} goals", self.home_readme)
        self.assertIn(f"{MINUS}{-recovery['post_2526']['coef']:.3f} goals", self.home_readme)
        share = -ghost["coef"] / main["margin"]["coefs"]["home_adv_const"]["coef"]
        self.assertIn(f"approximately {round(100 * share / 5) * 5}%", self.home_readme)


class PipelineRunner(unittest.TestCase):
    """The runner points at existing scripts and refuses incomplete inputs."""

    def setUp(self):
        self.original_root = run_pipeline.ROOT
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        manifest = (ROOT / "data/source_manifest.json").read_text(encoding="utf-8")
        (self.root / "data/source_manifest.json").write_text(manifest, encoding="utf-8")
        self.files = json.loads(manifest)["files"]
        run_pipeline.ROOT = self.root

    def tearDown(self):
        run_pipeline.ROOT = self.original_root
        self.tmp.cleanup()

    def write_inputs(self, content: bytes) -> Path:
        raw = self.root / "home-advantage/data/raw"
        raw.mkdir(parents=True)
        for item in self.files:
            (raw / item["file"]).write_bytes(content)
        return raw

    def test_every_step_exists(self):
        for project, scripts in run_pipeline.STEPS.values():
            for script in scripts:
                self.assertTrue((ROOT / project / "src" / script).is_file(), script)

    def test_missing_inputs_stop_before_estimation(self):
        with self.assertRaises(SystemExit) as caught:
            run_pipeline.check_inputs(require_snapshot=False)
        self.assertIn(f"Missing {len(self.files)} source files", str(caught.exception))

    def test_empty_input_counts_as_missing(self):
        raw = self.write_inputs(b"placeholder")
        (raw / self.files[0]["file"]).write_bytes(b"")
        with self.assertRaises(SystemExit) as caught:
            run_pipeline.check_inputs(require_snapshot=False)
        self.assertIn("Missing 1 source files", str(caught.exception))

    def test_revised_vintage_runs_only_without_snapshot_check(self):
        self.write_inputs(b"revised vintage")
        run_pipeline.check_inputs(require_snapshot=False)
        with self.assertRaises(SystemExit) as caught:
            run_pipeline.check_inputs(require_snapshot=True)
        self.assertIn("differs from the recorded research inputs", str(caught.exception))

    def test_recorded_snapshot_is_accepted(self):
        content = b"recorded snapshot"
        digest = hashlib.sha256(content).hexdigest()
        for item in self.files:
            item["sha256"] = digest
        (self.root / "data/source_manifest.json").write_text(json.dumps({"files": self.files}),
                                                             encoding="utf-8")
        self.write_inputs(content)
        run_pipeline.check_inputs(require_snapshot=True)


if __name__ == "__main__":
    unittest.main()
