# Release validation

The publication copy was checked on **September 18, 2026** using the original
local research inputs identified by [the source manifest](../data/source_manifest.json).
Raw data are not distributed in this repository.

## Executed checks

| Check | Result |
| --- | --- |
| Fresh Python 3.11 virtual environment | Installed every package in `requirements.txt`; `pip check` passed |
| Full pipeline from a relocated directory containing spaces | Passed from an unrelated working directory |
| Input snapshot verification | All 60 source files matched their recorded SHA-256 fingerprints |
| Home-advantage analysis | Cleaning, all regression specifications, 999-replication wild bootstraps and four figures completed |
| Forecasting analysis | Features, all nine model-fitting folds, evaluation, 2,000-replication betting bootstraps and five figures completed |
| Full run in the clean environment | Approximately **86 seconds** on the validation machine; other systems will differ |
| Missing input handling | Stops with the missing filenames before estimation or network activity |
| Failed acquisition handling | Simulated failure stops the downloader and leaves no partial source file |
| Extra files outside the study | An extra season CSV did not change the forecasting features |

The clean environment used Python 3.11.7 on macOS, NumPy 1.26.4, pandas 2.3.0,
SciPy 1.17.1, scikit-learn 1.8.0, statsmodels 0.14.6 and matplotlib 3.10.9.
Numerical libraries were limited to one worker by the runner where thread-count
variables were not already configured.

## Numerical comparison

The home-advantage result JSON reproduced the original saved values within
`1e-9`. Forecast metrics and betting results reproduced within `1e-7`. The
release corrects two output inconsistencies:

- `table4_s1920.csv` then switched from one-way to two-way home/away team-season
  standard errors, and both specifications were kept in the result JSON. The
  October 2026 revision below replaces both.
- Permutation importance now uses the gradient-boosting model's native missing
  feature handling, matching the main model pipeline. The Elo-difference
  importance is 0.09770; its leading rank and the forecast/betting conclusions
  are unchanged.

The rebuilt outputs were also compared between the existing research environment
and the freshly installed environment. Rounded reported scores and estimates
were unchanged. Numerical-library builds can still differ on other platforms.

## Re-run on September 23, 2026

The full pipeline was rerun with `--verify-snapshot` from a copy of the current
repository layout in a directory containing spaces, using the same pinned
library versions. All 60 inputs matched their recorded fingerprints. Every
numeric value in both result JSON files, all 19 CSV tables and all nine figures
were identical to the included outputs. The run took about 126 seconds on a
busier machine than the original validation.

## Revision, October 2026

An audit of the release found problems that the earlier checks could not detect.
The code was corrected as follows.

- **Within-2019–20 table and league-level estimates.** Clustering on home and away
  team-season groups, which coincide with the absorbed team-season strengths, made
  the clustered variance largely cancel. The published Ligue 1 closed-period
  goal-margin standard error was 0.00005 (p = 0), and in simulations with this
  design those standard errors were 20–45% of the true sampling spread. Both
  specifications now cluster by league matchday (league × calendar date), and
  `s1920_twoway` is no longer written.
- **Forecasting features.** The six matches without a complete Bet365 price now
  update Elo ratings and rolling form; they remain outside the evaluation sample.
  A club returning after at least one season away restarts with cleared rolling
  form, as its Elo rating already did.
- **Outputs.** `t3_rps_by_period.csv` adds the market restricted to the blend's
  matches, `t5_market_bias_by_season.csv` adds proportional de-margining next to
  Shin's method, the result JSON never contains NaN, closing odds are matched to
  predictions by date, league, result and prices, and the figure-5 overround line
  is computed from the test matches. Figure titles describe the period
  differences without interpreting them.
- **Checks.** The release check, unit tests, a synthetic smoke test and CI were
  extended, as described in the next section.

The pooled home-advantage specifications (main, recovery, sensitivity, placebo and
wild bootstrap) are numerically unchanged by these corrections.

`python tests/simulate_coverage.py` (300 replications, seed 7, 95 seconds)
measures how often each analytic 95% interval contains a true zero effect on the
study's design, with independent errors or with club-specific home advantage
(standard deviation 0.25 goals per team-season):

| Specification | Independent errors | Club home advantage |
| --- | --- | --- |
| Pooled, league-season clusters: closed-stadium coefficient | 0.92 | 0.92 |
| Pooled, league-season clusters: single post-pandemic seasons | 0.85–0.95 | 0.88–0.91 |
| Within 2019–20, league-matchday clusters | 0.93 | 0.97 |
| Single league, league-matchday clusters | 0.94–0.97 | 0.88–0.94 |

REGENERATION PENDING: rerun `python scripts/run_pipeline.py --verify-snapshot`
with the recorded inputs, replace this paragraph with the old → new values of
every changed number (forecasting headline metrics, ROI range, closing-line
comparison and `table4_s1920.csv`), the runtime and the snapshot result, and
update the README numbers that the unit tests flag.

## Checks available without source data

```sh
python scripts/check_release.py
python -m unittest discover -s tests
```

This standard-library check compares the 19 CSV tables with their result JSON:
every value the JSON stores, including coefficients, standard errors and
significance stars. For the parts the JSON does not store (the calibration bins
and the permutation-importance rows below the top ten) it checks internal
consistency instead: bin sizes and ordering, the recomputed ECE, the full feature
list and finite values. It requires all nine figures as complete PNG files;
rejects nulls (other than the blend's first-season score), non-finite values,
non-positive standard errors and out-of-range p-values anywhere in the result
files; and checks the test-sample counts, the market benchmark, the blend's
matched comparison, the input manifest (file names, URLs and unique digests) and
local document links. In a git checkout, links count only when their target is
tracked, so a link to a private or ignored file fails before publication. The
unit tests rerun those checks, confirm that the headline numbers and the
model-dependent statements in the READMEs match the saved results, show that
corrupted outputs are rejected, and exercise the pipeline runner's handling of
missing, empty and revised input files.

`python tests/smoke_pipeline.py` needs the research dependencies. It runs both
studies end to end on synthetic Football-Data-style inputs in a temporary copy
of the files git would publish, and applies the same consistency checks to the
regenerated outputs. `python tests/simulate_coverage.py` reproduces the interval
coverage reported above. A GitHub Actions workflow runs the offline checks, a
Python syntax check and the smoke test on pushes and pull requests. CI does not
download the source data.

## Scope

This validation checks implementation and reproducibility on an existing
historical snapshot. It does not establish prospective forecast performance,
executable betting profitability or an isolated causal crowd effect. Source
acquisition was not rerun against the live provider. See the study READMEs for
the substantive limitations.
