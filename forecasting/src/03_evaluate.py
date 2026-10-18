"""Forecast evaluation: proper scoring rules, model-vs-market tests,
calibration, regime breakdowns, and GBM feature importance.

Writes output/results.json and output/tables/*.csv.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_T = ROOT / "output" / "tables"
OUT_T.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(ROOT / "data" / "predictions.csv", parse_dates=["Date"])
if pred.duplicated(["match_id", "model"]).any():
    raise ValueError("Duplicate model/match predictions would distort evaluation.")


def without_nan(obj):
    """Replace non-finite floats with None so the saved JSON stays standard."""
    if isinstance(obj, dict):
        return {k: without_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [without_nan(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


MATCH_KEY = ["Date", "Div", "y", "B365H", "B365D", "B365A"]


def read_feature_columns(columns):
    """Per-match columns from features.csv, keyed by the match_id 02_models.py assigned.

    The key is the row position, so each predicted match must still have the
    same date, league, result and prices at that position.
    """
    feat = pd.read_csv(ROOT / "data" / "features.csv", usecols=[*MATCH_KEY, *columns],
                       parse_dates=["Date"])
    feat["match_id"] = np.arange(len(feat))
    expected = pred.drop_duplicates("match_id").set_index("match_id")[MATCH_KEY]
    aligned = feat.set_index("match_id")[MATCH_KEY].reindex(expected.index)
    if not aligned.equals(expected):
        raise ValueError("features.csv does not match predictions.csv; rerun 02_models.py.")
    return feat[["match_id", *columns]]

# ---------------- Crowd-period coding (same rule as the home-advantage study) ----------------
GHOST_START = {"I1": "2020-03-04", "F1": "2020-03-04", "SP1": "2020-03-10",
               "D1": "2020-03-11", "E0": "2020-03-13"}
gs = pred["Div"].map(GHOST_START).astype("datetime64[ns]")
pred["period"] = np.select(
    [pred["Date"] < gs, pred["Date"] <= pd.Timestamp("2021-06-30")],
    ["pre", "ghost"], default="post")

# ---------------- Scoring rules ----------------
P = pred[["pH", "pD", "pA"]].to_numpy()
if not np.isfinite(P).all() or (P < 0).any() or not np.allclose(P.sum(axis=1), 1):
    raise ValueError("Prediction probabilities must be finite, nonnegative, and sum to one.")
Yoh = np.eye(3)[pred["y"].to_numpy()]
eps = 1e-12
pred["logloss"] = -np.log(np.clip(P[np.arange(len(P)), pred["y"]], eps, 1))
pred["brier"] = ((P - Yoh) ** 2).sum(axis=1)
cdf_p = np.cumsum(P, axis=1)[:, :2]
cdf_y = np.cumsum(Yoh, axis=1)[:, :2]
pred["rps"] = 0.5 * ((cdf_p - cdf_y) ** 2).sum(axis=1)
pred["hit"] = (P.argmax(axis=1) == pred["y"]).astype(float)

results = {}

# ---------------- Pooled and per-season metrics ----------------
def metrics(g):
    return pd.Series({"rps": g["rps"].mean(), "logloss": g["logloss"].mean(),
                      "brier": g["brier"].mean(), "acc": g["hit"].mean(),
                      "n": len(g)})

pooled = pred.groupby("model").apply(metrics, include_groups=False).round(5)
pooled.to_csv(OUT_T / "t1_pooled_metrics.csv")
print("Pooled metrics (all test seasons):\n", pooled.to_string(), "\n")
results["pooled"] = json.loads(pooled.to_json(orient="index"))

by_season = (pred.groupby(["model", "season"])["rps"].mean().unstack().round(5))
by_season.to_csv(OUT_T / "t2_rps_by_season.csv")
results["rps_by_season"] = json.loads(by_season.to_json(orient="index"))

by_period = (pred.groupby(["model", "period"])["rps"].mean()
             .unstack()[["pre", "ghost", "post"]])
# The blend starts a season later, so its cells cover a shorter sample. The
# market restricted to the blend's matches gives the like-for-like row.
bl_ids = set(pred.loc[pred["model"] == "blend", "match_id"])
mk_sub = pred[(pred["model"] == "market") & (pred["match_id"].isin(bl_ids))]
by_period.loc["market_blend_sample"] = mk_sub.groupby("period")["rps"].mean()
by_period = by_period.round(5)
by_period.to_csv(OUT_T / "t3_rps_by_period.csv")
results["rps_by_period"] = json.loads(by_period.to_json(orient="index"))
print("RPS by period:\n", by_period.to_string(), "\n")

# ---------------- Clustered DM-style tests vs the market ----------------
def dm_test(model, metric="rps", subset=None):
    a = pred[pred["model"] == model]
    b = pred[pred["model"] == "market"]
    m = a.merge(b, on="match_id", suffixes=("_m", "_b"))
    if subset is not None:
        m = m[subset(m)]
    d = (m[f"{metric}_m"] - m[f"{metric}_b"]).to_numpy()
    dates = m["Date_m"].to_numpy()
    dbar = d.mean()
    resid = d - dbar
    groups = pd.Series(resid).groupby(pd.Series(dates)).sum().to_numpy()
    se = np.sqrt((groups ** 2).sum()) / len(d)
    t = dbar / se
    from scipy.stats import norm
    return {"mean_gap": float(dbar), "se": float(se), "t": float(t),
            "p": float(2 * norm.sf(abs(t))), "n": int(len(d))}

for mdl in ["elo", "logit", "gbm", "dpois", "blend"]:
    results[f"dm_{mdl}_vs_market"] = dm_test(mdl)
    results[f"dm_{mdl}_vs_market_ghost"] = dm_test(mdl, subset=lambda m: m["period_m"] == "ghost")
    results[f"dm_{mdl}_vs_market_2425"] = dm_test(mdl, subset=lambda m: m["season_m"] == "2024-25")
print("DM vs market (pooled RPS gap, + = worse than market):")
for mdl in ["elo", "logit", "gbm", "dpois", "blend"]:
    r = results[f"dm_{mdl}_vs_market"]
    print(f"  {mdl:6s} gap {r['mean_gap']:+.5f} (t={r['t']:+.2f}, p={r['p']:.4f})")

# Blend vs market by season (the regime-shift question)
blend_gap_by_season = {}
for s in pred["season"].unique():
    if s == "2017-18":
        continue
    blend_gap_by_season[s] = dm_test("blend", subset=lambda m, s=s: m["season_m"] == s)
results["blend_gap_by_season"] = blend_gap_by_season
print("\nBlend - market RPS gap by season:")
for s, r in sorted(blend_gap_by_season.items()):
    star = "*" if r["p"] < 0.05 else ""
    print(f"  {s}: {r['mean_gap']:+.5f} (p={r['p']:.3f}){star}")

# ---------------- Calibration ----------------
def reliability(model, col, ycls, bins=10):
    g = pred[pred["model"] == model]
    p = g[col].to_numpy()
    o = (g["y"] == ycls).astype(float).to_numpy()
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() > 0:
            rows.append({"bin": b, "p_mean": p[mask].mean(),
                         "o_rate": o[mask].mean(), "n": int(mask.sum())})
    t = pd.DataFrame(rows)
    ece = (t["n"] * (t["p_mean"] - t["o_rate"]).abs()).sum() / t["n"].sum()
    return t, float(ece)

cal_tables = {}
for mdl in ["market", "gbm", "blend"]:
    for col, ycls, lab in [("pH", 0, "home"), ("pA", 2, "away")]:
        t, ece = reliability(mdl, col, ycls)
        t.to_csv(OUT_T / f"cal_{mdl}_{lab}.csv", index=False)
        cal_tables[f"{mdl}_{lab}_ece"] = ece
results["ece"] = cal_tables
print("\nECE (home-win prob):", {k: round(v, 4) for k, v in cal_tables.items() if "home" in k})

# Market calibration during ghost period specifically (regime mispricing)
g = pred[(pred["model"] == "market")]
for per in ["pre", "ghost", "post"]:
    sub = g[g["period"] == per]
    results[f"market_home_bias_{per}"] = {
        "implied": float(sub["pH"].mean()),
        "realized": float((sub["y"] == 0).mean()),
        "n": int(len(sub)),
    }
print("Market implied vs realized home-win rate by period:",
      {p: (round(results[f'market_home_bias_{p}']['implied'], 4),
           round(results[f'market_home_bias_{p}']['realized'], 4)) for p in ["pre", "ghost", "post"]})

# Per-season market home bias with date-clustered bootstrap CI (ties to the
# home-advantage study: did bookmakers track the 2024-25 fall and the rebound?)
# The benchmark uses Shin probabilities (pH); proportional normalization
# (mkt_H, the convention of the home-advantage study) is reported alongside
# because Shin assigns slightly more probability to the usually favoured home
# side, which shifts the measured bias by a few tenths of a point.
rng = np.random.default_rng(42)

def cluster_boot_mean(df_sub, cols, B=2000):
    """Bootstrap 95% interval of each column's mean, resampling match dates."""
    by_date = df_sub.groupby("Date")
    s, c = by_date[cols].sum().to_numpy(), by_date.size().to_numpy()
    draws = np.empty((B, len(cols)))
    for b in range(B):
        idx = rng.integers(0, len(s), len(s))
        draws[b] = s[idx].sum(axis=0) / c[idx].sum()
    return np.quantile(draws, [0.025, 0.975], axis=0)

bias_rows = []
for s in sorted(g["season"].unique()):
    sub = g[g["season"] == s].copy()
    sub["diff"] = (sub["y"] == 0).astype(float) - sub["pH"]
    sub["diff_basic"] = (sub["y"] == 0).astype(float) - sub["mkt_H"]
    (lo, lo_b), (hi, hi_b) = cluster_boot_mean(sub, ["diff", "diff_basic"])
    bias_rows.append({"season": s, "implied": sub["pH"].mean(),
                      "realized": (sub["y"] == 0).mean(),
                      "bias": sub["diff"].mean(), "ci_lo": lo, "ci_hi": hi,
                      "implied_basic": sub["mkt_H"].mean(),
                      "bias_basic": sub["diff_basic"].mean(),
                      "ci_lo_basic": lo_b, "ci_hi_basic": hi_b,
                      "n": len(sub)})
bias_t = pd.DataFrame(bias_rows).round(4)
bias_t.to_csv(OUT_T / "t5_market_bias_by_season.csv", index=False)
results["market_bias_by_season"] = bias_t.to_dict(orient="records")
print("\nMarket home-win bias by season (realized - implied, 95% CI):")
print(bias_t.to_string(index=False))

# Like-for-like: market metrics restricted to the blend's evaluation sample
results["market_on_blend_sample"] = json.loads(metrics(mk_sub).round(5).to_json())
print("\nMarket on blend sample:", results["market_on_blend_sample"])

# Robustness: de-margining method (Shin vs basic normalization), and
# closing vs pre-match odds on the subsample where closing odds exist
def rps_of(pcols, sub):
    Pm = sub[pcols].to_numpy()
    Yo = np.eye(3)[sub["y"].to_numpy()]
    cp, cy = np.cumsum(Pm, axis=1)[:, :2], np.cumsum(Yo, axis=1)[:, :2]
    return float((0.5 * ((cp - cy) ** 2).sum(axis=1)).mean())

mk = pred[pred["model"] == "market"]
results["demargin_robustness"] = {
    "shin_rps": rps_of(["pH", "pD", "pA"], mk),
    "basic_rps": rps_of(["mkt_H", "mkt_D", "mkt_A"], mk),
}
# Closing comparison: Shin de-margining applied to BOTH legs (apples to apples)
feat_odds = read_feature_columns(["B365CH", "B365CD", "B365CA"])
mk2 = mk.merge(feat_odds, on="match_id", how="left")
mc = mk2[mk2["B365CH"].notna()].copy()
from scipy.optimize import brentq
def shin_row(inv):
    s = inv.sum()
    def f(z):
        return np.sum((np.sqrt(z**2 + 4*(1-z)*inv**2/s) - z) / (2*(1-z))) - 1
    try:
        z = brentq(f, 0.0, 0.2, xtol=1e-10)
        return (np.sqrt(z**2 + 4*(1-z)*inv**2/s) - z) / (2*(1-z))
    except ValueError:
        return inv / s
invC = 1.0 / mc[["B365CH", "B365CD", "B365CA"]].to_numpy()
mc[["shinC_H", "shinC_D", "shinC_A"]] = np.vstack([shin_row(r) for r in invC])
results["closing_vs_prematch"] = {
    "n": int(len(mc)),
    "prematch_shin_rps": rps_of(["pH", "pD", "pA"], mc),
    "closing_shin_rps": rps_of(["shinC_H", "shinC_D", "shinC_A"], mc),
}
print("De-margin robustness:", {k: round(v, 5) for k, v in results["demargin_robustness"].items()})
print("Closing vs pre-match, both Shin (2019-20+):",
      {k: (round(v, 5) if isinstance(v, float) else v) for k, v in results["closing_vs_prematch"].items()})

# Models' own home-win bias in the two collapse seasons (are models more
# anchored to historical home advantage than the market?)
results["model_home_bias"] = {}
for mdl in ["gbm", "logit", "market"]:
    gm = pred[pred["model"] == mdl]
    for s in ["2020-21", "2024-25"]:
        sub = gm[gm["season"] == s]
        results["model_home_bias"][f"{mdl}_{s}"] = {
            "implied": float(sub["pH"].mean()),
            "realized": float((sub["y"] == 0).mean()),
            "bias": float((sub["y"] == 0).mean() - sub["pH"].mean()),
        }
print("Model home bias (realized - implied):",
      {k: round(v["bias"], 4) for k, v in results["model_home_bias"].items()})

# Pairwise model DM tests (is the logit's edge over Elo-only / GBM real?)
def dm_pair(model_a, model_b, metric="rps"):
    a = pred[pred["model"] == model_a]
    b = pred[pred["model"] == model_b]
    m = a.merge(b, on="match_id", suffixes=("_m", "_b"))
    d = (m[f"{metric}_m"] - m[f"{metric}_b"]).to_numpy()
    dbar = d.mean()
    groups = pd.Series(d - dbar).groupby(pd.Series(m["Date_m"].to_numpy())).sum().to_numpy()
    se = np.sqrt((groups ** 2).sum()) / len(d)
    from scipy.stats import norm
    return {"mean_gap": float(dbar), "t": float(dbar / se),
            "p": float(2 * norm.sf(abs(dbar / se)))}
results["dm_logit_vs_elo"] = dm_pair("logit", "elo")
results["dm_logit_vs_gbm"] = dm_pair("logit", "gbm")
print("logit vs elo:", {k: round(v, 5) for k, v in results["dm_logit_vs_elo"].items()})
print("logit vs gbm:", {k: round(v, 5) for k, v in results["dm_logit_vs_gbm"].items()})

# ---------------- GBM permutation importance (last fold) ----------------
print("\nPermutation importance (refit on pre-2025-26, scored on 2025-26)...")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

feat = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["Date"])
DIFF_FEATS = ["d_ppg5", "d_ppg10", "d_gf5", "d_ga5", "d_sotf5", "d_sota5",
              "d_cor5", "d_vppg", "d_vgd", "d_rest", "d_played"]
RAW_FEATS = ["elo_home", "elo_away", "elo_diff", "h_ppg5", "a_ppg5",
             "h_vppg", "a_vppg", "h_hist_n", "a_hist_n"]
lg = pd.get_dummies(feat["Div"], prefix="lg").astype(float)
X = pd.concat([feat[DIFF_FEATS + RAW_FEATS], lg], axis=1)
tr = (feat["season"] < "2025-26").to_numpy()
te = (feat["season"] == "2025-26").to_numpy()
gbm = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600,
                                     max_leaf_nodes=31, l2_regularization=1.0,
                                     early_stopping=True, validation_fraction=0.15,
                                     n_iter_no_change=30, random_state=42)
gbm.fit(X[tr], feat.loc[tr, "y"])
# Use the same native missing-value handling as the evaluated GBM predictions.
imp = permutation_importance(gbm, X[te], feat.loc[te, "y"],
                             scoring="neg_log_loss", n_repeats=5, random_state=42)
imp_t = (pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean,
                       "std": imp.importances_std})
         .sort_values("importance", ascending=False))
imp_t.to_csv(OUT_T / "t4_permutation_importance.csv", index=False)
results["top_features"] = imp_t.head(10).to_dict(orient="records")
print(imp_t.head(10).to_string(index=False))

pred.to_csv(ROOT / "data" / "predictions_scored.csv", index=False)
with open(ROOT / "output" / "results.json", "w") as f:
    json.dump(without_nan(results), f, indent=1, allow_nan=False)
print("\nSaved output/results.json")
