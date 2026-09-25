# European Football Forecasting and Home Advantage

Two linked Python research studies examine probability forecasting, bookmaker
benchmarks and changes in home advantage across Europe's five largest domestic
leagues. The work combines chronological model evaluation with panel regression
and clustered inference.

**Main findings:** the tested forecasting models underperform the bookmaker
benchmark overall. Home advantage is lower during the closed-stadium period,
with further variation after restrictions end. These results support disciplined
benchmarking and cautious interpretation of observational evidence.

## Research results

| Study | Sample | Main result |
| --- | --- | --- |
| [Forecasting and market comparison](forecasting/README.md) | 21,583 matches with complete Bet365 odds; 16,106 out-of-sample matches across nine seasons | Market RPS **0.19511**, versus **0.20007** for multinomial logistic regression and **0.20200** for gradient boosting. Lower is better. |
| [Home advantage and closed stadiums](home-advantage/README.md) | 21,589 matches, 2014–15 through 2025–26 | Adjusted closed-period home goal-margin difference **−0.198 goals** and home-win probability difference **−5.1 percentage points**. |

The forecasting study compares four public-match-data models, a market benchmark
and a model–market blend. The blend covers 14,280 matches and has RPS 0.19593,
compared with 0.19578 for the market on the same sample. Across the 16 tested
model/threshold combinations, hypothetical betting ROI ranges from **−13.8% to
−4.1%**. These experiments do not demonstrate profitable execution.

![Forecast performance by season](forecasting/output/figures/fig1_rps_by_season.png)

The home-advantage regressions absorb team-by-season strengths and use clustered
standard errors. They estimate period differences conditional on this model.
The pandemic also changed scheduling and other conditions, and disciplinary
statistics reflect both players and referees. The coefficients do not isolate
the causal effect of crowd noise or establish referee bias.

![Adjusted period differences in goals and yellow cards](home-advantage/output/figures/fig4_coefplot.png)

## Forecasting pipeline at a glance

| Stage | What the code does | Where |
| --- | --- | --- |
| Hypothesis | Public match records might add information that bookmaker prices miss, especially when home advantage shifts. | [forecasting README](forecasting/README.md) |
| Data | 60 league-season files from Football-Data: results, match statistics and Bet365 odds for the Big-5 leagues, 2014–15 to 2025–26. | [DATA_SOURCES.md](DATA_SOURCES.md) |
| Signal | Sequential Elo, rolling form, shots on target, corners, venue form and rest days, each computed only from earlier matches. Odds never enter the standalone models. | [`01_features.py`](forecasting/src/01_features.py) |
| Models | Elo-only logit, full-feature logit, gradient boosting and a double-Poisson goal model, refit each season on earlier seasons only. The market benchmark is Bet365 odds de-margined with Shin's method; the blend stacks gradient boosting with the market. | [`02_models.py`](forecasting/src/02_models.py) |
| Bet selection | Flat one-unit stake on any outcome where model probability × decimal odds exceeds 1 + θ, for θ ∈ {0, 0.02, 0.05, 0.10}. | [`04_backtest.py`](forecasting/src/04_backtest.py) |
| Execution assumptions | Bets settle at the stored Bet365 pre-closing price. There are no limits, slippage or fees, and no quote timestamps to confirm the price was available. | [`04_backtest.py`](forecasting/src/04_backtest.py) |
| Risk controls | Walk-forward refits, date-clustered bootstrap intervals for ROI, a closing-line-value check at closing odds, and results split by crowd period. | [`03_evaluate.py`](forecasting/src/03_evaluate.py), [`04_backtest.py`](forecasting/src/04_backtest.py) |
| Results | Ranked probability score, log-loss, Brier score, calibration, clustered score-difference tests and the ROI grid. | [`results.json`](forecasting/output/results.json), [tables](forecasting/output/tables/) |

## Start here

- Read the linked study READMEs for methods, limitations and result files.
- Browse the [forecasting code](forecasting/src) or
  [regression code](home-advantage/src).
- See [validation](docs/VALIDATION.md) for what was actually rerun.

## Setup and offline checks

Use Python 3.11. From this repository's root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_release.py
python -m unittest discover -s tests
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
The release check and the unit tests use only Python's standard library, so they
run before the research dependencies are installed and without the source data.
They check that every saved CSV table agrees with its result JSON, that all nine
figures exist, that the headline numbers in the READMEs match the saved results,
the reported sample sizes, the source manifest, local documentation links and the
pipeline runner's input handling. They do not refit models or replace
statistical review.

With the research dependencies installed, a smoke test runs both studies end to
end on synthetic Football-Data-style inputs in a temporary copy, leaving the
included outputs untouched (about two minutes):

```sh
python tests/smoke_pipeline.py
```

## Reproduce the studies

Raw third-party CSVs and per-match derived data are not distributed. Arrange
access to the source data as described in [DATA_SOURCES.md](DATA_SOURCES.md),
then place the 60 required files in `home-advantage/data/raw/`.
The [manifest](data/source_manifest.json) records the URLs and fingerprints of
the research inputs used for release verification. Both studies share them.

```sh
# Run both studies with the exact recorded source snapshot
python scripts/run_pipeline.py --verify-snapshot

# Or run one study, allowing a revised source vintage
python scripts/run_pipeline.py --study home-advantage
python scripts/run_pipeline.py --study forecasting
```

The runner never downloads inputs automatically. It runs:

1. Home advantage: cleaning → regressions and bootstrap inference → figures.
2. Forecasting: features → seasonal model fitting → evaluation → hypothetical
   betting returns → figures.

The forecasting study can run independently once the shared raw files exist.
All paths are resolved relative to script locations. The runner works from a
different working directory and after moving the checkout, including to a path
with spaces. Runs overwrite the corresponding generated results and figures;
use a separate copy to compare a new run with the included release outputs.
The full pipeline includes repeated model fitting and bootstrap calculations.
See the measured validation runtime in [VALIDATION.md](docs/VALIDATION.md).

Dependency pins describe the environment used for this release verification.
Random seeds are specified in the code, but library builds, platforms and revised
input data can change numerical results.

## Interpretation and limits

- Walk-forward fitting uses earlier seasons. Rolling features use preceding
  matches, and the blend trains on earlier out-of-sample model predictions.
  Model specifications were examined retrospectively; no preregistration or
  independent prospective holdout is claimed.
- The blend has a shorter evaluation sample. Compare its score with the market
  on the same observations, rather than directly with the nine-season pooled
  market score.
- Pre-closing bookmaker odds may precede the latest model features. Hypothetical
  returns omit executable quotes, limits, slippage and operational costs.
- Closed-stadium exposure is coded from policy windows rather than measured
  match attendance. Pandemic confounding remains possible.
- A nonsignificant coefficient does not prove equality or complete recovery.
  Season comparisons and feature-importance results are exploratory.
- Conclusions concern these leagues, features, specifications and historical
  seasons. They do not establish that all public information is priced in or
  that other approaches cannot perform better.

## Repository layout

```text
forecasting/               Model pipeline, aggregate tables and figures
home-advantage/            Panel analysis, aggregate tables and figures
data/source_manifest.json  Input URLs and historical file fingerprints
scripts/                   Pipeline runner and offline release checks
tests/                     Unit tests, synthetic-input generator and pipeline smoke test
docs/                      Release validation record
requirements.txt           Research dependencies for Python 3.11
```

## Attribution and license

Author: **Chung Shing Mars Wong**. Match data are from
[Football-Data](https://www.football-data.co.uk/data.php). Original code and
accompanying original text are released under the [MIT License](LICENSE).
Third-party data retain their own terms; see [DATA_SOURCES.md](DATA_SOURCES.md).
