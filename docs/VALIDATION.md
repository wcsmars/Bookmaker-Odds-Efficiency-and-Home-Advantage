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

- `table4_s1920.csv` now reports the two-way team-season standard errors used in
  the reported within-2019–20 results, instead of the alternative one-way specification. Both
  specifications remain in the result JSON.
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

## Checks available without source data

```sh
python scripts/check_release.py
python -m unittest discover -s tests
```

This standard-library check compares every one of the 19 CSV tables with its
result JSON (coefficients, standard errors and significance stars included),
requires all nine figures, rejects non-finite values and non-positive standard
errors, and checks test-sample counts, the blend's matched market comparison,
the full ROI range, the input manifest (file names, URLs and unique digests) and
local document links. In a git checkout, links count only when their target is
tracked, so a link to a private or ignored file fails before publication. The
unit tests rerun those checks, confirm that the headline numbers in the READMEs
match the saved results, show that corrupted tables are rejected, and exercise
the pipeline runner's handling of missing, empty and revised input files.

`python tests/smoke_pipeline.py` needs the research dependencies. It runs both
studies end to end on synthetic Football-Data-style inputs in a temporary copy
and applies the same consistency checks to the regenerated outputs. A GitHub
Actions workflow runs the offline checks, a Python syntax check and the smoke
test on pushes and pull requests. CI does not download the source data.

## Scope

This validation checks implementation and reproducibility on an existing
historical snapshot. It does not establish prospective forecast performance,
executable betting profitability or an isolated causal crowd effect. Source
acquisition was not rerun against the live provider. See the study READMEs for
the substantive limitations.
