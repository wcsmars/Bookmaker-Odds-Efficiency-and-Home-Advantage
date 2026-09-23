"""Offline tests for the saved results, release checks and pipeline runner (stdlib only).

Run from the repository root with: python -m unittest discover -s tests
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
STANDALONE_MODELS = ("elo", "logit", "gbm", "dpois")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_release = load_script("check_release")
run_pipeline = load_script("run_pipeline")


class ReleaseChecks(unittest.TestCase):
    """Each offline release check passes on the included outputs."""

    def assert_passes(self, check) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as output:
            check()
        self.assertTrue(output.getvalue().startswith("PASS"), output.getvalue())

    def test_forecasting_tables_match_results(self):
        self.assert_passes(check_release.check_forecast)

    def test_home_advantage_tables_match_results(self):
        self.assert_passes(check_release.check_home)

    def test_source_manifest_is_complete(self):
        self.assert_passes(check_release.check_manifest)

    def test_documentation_links_resolve(self):
        self.assert_passes(check_release.check_documentation)


class ReportedClaims(unittest.TestCase):
    """Headline statements in the READMEs agree with the saved results."""

    @classmethod
    def setUpClass(cls):
        cls.forecast = check_release.load_json(ROOT / "forecasting/output/results.json")
        cls.home = check_release.load_json(ROOT / "home-advantage/output/results.json")

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


class PipelineRunner(unittest.TestCase):
    """The runner points at existing scripts and refuses incomplete inputs."""

    def setUp(self):
        self.original_root = run_pipeline.ROOT
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        manifest = (ROOT / "data/source_manifest.json").read_text()
        (self.root / "data/source_manifest.json").write_text(manifest)
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
        (self.root / "data/source_manifest.json").write_text(json.dumps({"files": self.files}))
        self.write_inputs(content)
        run_pipeline.check_inputs(require_snapshot=True)


if __name__ == "__main__":
    unittest.main()
