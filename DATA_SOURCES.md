# Data sources and reproduction inputs

The research uses match results, match statistics and bookmaker odds from
[Football-Data](https://www.football-data.co.uk/data.php). The source covers
England (`E0`), Germany (`D1`), Spain (`SP1`), Italy (`I1`) and France (`F1`),
for twelve seasons from 2014–15 through 2025–26.

## What is distributed

This repository includes original analysis code, aggregate result tables
and figures. Raw source CSVs, processed match records and
per-match model predictions are not redistributed. The code license does not
grant rights to the provider's data.

The provider currently describes the free data as intended for private
individuals and places restrictions on commercial and automated data-training
products. Consult its [current conditions](https://www.football-data.co.uk/data.php)
and obtain appropriate access for your use. These source conditions were checked
on September 18, 2026. The repository does not automatically fetch data.

## Input layout

Place the 60 league-season CSV files in
`home-advantage/data/raw/`, preserving names such as `E0_1415.csv`
and `F1_2526.csv`. Both studies read this directory. The
[source manifest](data/source_manifest.json) records each required filename,
source URL, byte length and SHA-256 digest of the local research snapshot.
The original retrieval dates were not recorded; file modification times are
not used as substitutes.

After arranging access under the provider's conditions, files can be obtained
from its country pages using the URLs in the manifest. The historical helper
`bash home-advantage/src/01_download.sh` is an explicit opt-in convenience
for users whose access permits it. It skips existing nonempty files and reports
failed downloads. It is not part of the default pipeline.

## Snapshot verification

With the original research inputs present, run:

```sh
python scripts/run_pipeline.py --verify-snapshot
```

Omit `--verify-snapshot` to analyze a changed source vintage with the same
60 expected filenames. Historical source files can be revised. A changed
vintage or numerical-library build may yield different estimates; inspect
the new output rather than describing it as an exact reproduction.

## Odds and sample definitions

The main price benchmark and hypothetical betting returns use `B365H`,
`B365D` and `B365A`. The provider distinguishes these pre-closing observations
from closing observations with `C` in their column names from 2019–20 onward.
`B365CH`, `B365CD` and `B365CA` are used only for the closing-price comparisons
where available. See the provider's
[column notes](https://www.football-data.co.uk/notes.txt).

The home-advantage sample has 21,589 matches. Requiring complete Bet365 odds
leaves 21,583 forecasting feature rows; the six matches without a complete
price still update Elo ratings and rolling form. The initial three seasons are used for
training, leaving 16,106 out-of-sample matches across nine test seasons. The
blend starts a season later and covers 14,280 matches. Scores for the blend
must be compared with the market on that same sample.

Pre-closing prices are periodic snapshots, not guaranteed executable quotes at
the model's information cutoff. Latest prior-match features may become known
after a stored price snapshot. Betting returns therefore describe hypothetical
historical comparisons, without establishing a tradable strategy.
