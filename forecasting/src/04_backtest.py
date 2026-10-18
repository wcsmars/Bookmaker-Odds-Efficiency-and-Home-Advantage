"""Economic evaluation: value-betting backtests against Bet365 odds.

Flat 1-unit stakes on any outcome where p_model x decimal_odds > 1 + theta.
Bootstrap CIs cluster-resample match dates. A closing-line-value (CLV) check
settles the same bets at closing odds (available 2019-20 onward).
Writes backtest results into output/results.json (key 'backtest').
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pred = pd.read_csv(ROOT / "data" / "predictions_scored.csv", parse_dates=["Date"])
# Closing odds are keyed by the row position 02_models.py used as match_id, so
# confirm each predicted match still has the same date, league, result and prices.
MATCH_KEY = ["Date", "Div", "y", "B365H", "B365D", "B365A"]
feat = pd.read_csv(ROOT / "data" / "features.csv",
                   usecols=[*MATCH_KEY, "B365CH", "B365CD", "B365CA"], parse_dates=["Date"])
feat["match_id"] = np.arange(len(feat))
expected = pred.drop_duplicates("match_id").set_index("match_id")[MATCH_KEY]
if not feat.set_index("match_id")[MATCH_KEY].reindex(expected.index).equals(expected):
    raise ValueError("features.csv does not match predictions_scored.csv; rerun 02_models.py.")
pred = pred.merge(feat[["match_id", "B365CH", "B365CD", "B365CA"]], on="match_id", how="left")
rng = np.random.default_rng(42)


def without_nan(obj):
    """Replace non-finite floats with None so the saved JSON stays standard."""
    if isinstance(obj, dict):
        return {k: without_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [without_nan(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj

ODDS = {"H": "B365H", "D": "B365D", "A": "B365A"}
CODDS = {"H": "B365CH", "D": "B365CD", "A": "B365CA"}
YMAP = {"H": 0, "D": 1, "A": 2}


def bets_for(model, theta):
    g = pred[pred["model"] == model].copy()
    rows = []
    for o in ["H", "D", "A"]:
        p = g[f"p{o}"].to_numpy()
        odds = g[ODDS[o]].to_numpy()
        ev = p * odds - 1
        sel = ev > theta
        won = (g["y"].to_numpy() == YMAP[o])
        r = pd.DataFrame({
            "Date": g["Date"].to_numpy()[sel],
            "season": g["season"].to_numpy()[sel],
            "period": g["period"].to_numpy()[sel],
            "profit": np.where(won[sel], odds[sel] - 1, -1.0),
            "odds": odds[sel], "ev": ev[sel],
            "codds": g[CODDS[o]].to_numpy()[sel],
            "won": won[sel],
        })
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def boot_ci(bets, B=2000):
    if len(bets) == 0:
        return (np.nan, np.nan)
    dates = bets["Date"].unique()
    prof_by_date = bets.groupby("Date")["profit"].agg(["sum", "count"])
    sums, counts = prof_by_date["sum"].to_numpy(), prof_by_date["count"].to_numpy()
    rois = []
    for _ in range(B):
        idx = rng.integers(0, len(dates), len(dates))
        rois.append(sums[idx].sum() / counts[idx].sum())
    return (float(np.quantile(rois, 0.025)), float(np.quantile(rois, 0.975)))


results = json.loads((ROOT / "output" / "results.json").read_text())
bt = {}
print(f"{'model':7s} {'theta':>5s} {'n':>6s} {'ROI %':>7s} {'95% CI':>18s}")
for model in ["logit", "gbm", "dpois", "blend"]:
    bt[model] = {}
    for theta in [0.0, 0.02, 0.05, 0.10]:
        b = bets_for(model, theta)
        roi = b["profit"].mean() if len(b) else np.nan
        lo, hi = boot_ci(b)
        bt[model][f"theta_{theta}"] = {
            "n_bets": int(len(b)), "roi": float(roi),
            "ci_lo": lo, "ci_hi": hi,
            "mean_odds": float(b["odds"].mean()) if len(b) else np.nan,
        }
        print(f"{model:7s} {theta:5.2f} {len(b):6d} {100*roi:7.2f} [{100*lo:6.2f}, {100*hi:6.2f}]")

# By period at theta = 0.05 for the blend and gbm
for model in ["gbm", "blend"]:
    b = bets_for(model, 0.05)
    bt[model]["by_period"] = {}
    for per in ["pre", "ghost", "post"]:
        sub = b[b["period"] == per]
        lo, hi = boot_ci(sub)
        bt[model]["by_period"][per] = {"n_bets": int(len(sub)),
                                       "roi": float(sub["profit"].mean()) if len(sub) else np.nan,
                                       "ci_lo": lo, "ci_hi": hi}
    print(f"\n{model} theta=0.05 by period:",
          {p: (v["n_bets"], round(100 * v["roi"], 2)) for p, v in bt[model]["by_period"].items()})

# Closing-line value: for bets placed at theta=0.05 (2019-20+ where closing exists)
for model in ["gbm", "blend"]:
    b = bets_for(model, 0.05)
    b = b[b["codds"].notna()]
    if len(b):
        clv = (b["odds"] / b["codds"] - 1).mean()          # >0: beat the close
        roi_close = np.where(b["won"], b["codds"] - 1, -1.0).mean()
        bt[model]["clv"] = {"n": int(len(b)), "mean_clv": float(clv),
                            "roi_at_close": float(roi_close)}
        print(f"{model} CLV: n={len(b)}, mean odds vs close {100*clv:+.2f}%, "
              f"ROI if settled at closing odds {100*roi_close:+.2f}%")

# Ex-post diagnostic (NOT a tradable strategy): blind away-backing in the two
# seasons where the market overpriced home wins (2020-21, 2024-25)
diag = {}
mk = pred[pred["model"] == "market"]
for s in ["2020-21", "2024-25", "2025-26"]:
    g = mk[mk["season"] == s]
    prof = np.where(g["y"] == 2, g["B365A"] - 1, -1.0)
    lo, hi = boot_ci(pd.DataFrame({"Date": g["Date"], "profit": prof}))
    diag[s] = {"n": int(len(g)), "roi": float(prof.mean()), "ci_lo": lo, "ci_hi": hi}
    print(f"Blind-away {s}: ROI {100*prof.mean():+.2f}% [{100*lo:.2f}, {100*hi:.2f}]")
bt["blind_away_diagnostic"] = diag

# Mean Bet365 overround on the test matches: the reference level for the ROI
# figure (a bettor with no edge loses about the margin on average).
bt["mean_overround_test"] = float(mk["overround"].mean())
print(f"Mean overround on the {len(mk)} test matches: {100 * bt['mean_overround_test']:.2f}%")

results["backtest"] = bt
with open(ROOT / "output" / "results.json", "w") as f:
    json.dump(without_nan(results), f, indent=1, allow_nan=False)
print("\nSaved backtest results")
