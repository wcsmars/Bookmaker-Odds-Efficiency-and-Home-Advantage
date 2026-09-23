"""Walk-forward model estimation and out-of-sample prediction.

For each test season s in 2017-18..2025-26, every model is fit on all matches
strictly before s and predicts every match of s. Models:

  elo     multinomial logit on the Elo difference alone
  logit   multinomial logit on the full public-data feature set
  gbm     histogram gradient boosting on the same features
  dpois   time-decayed double-Poisson goal model (Maher/Dixon-Coles family)
  market  Shin-de-margined Bet365 probabilities (benchmark; no fitting)
  blend   multinomial logit stacking gbm + market log-probabilities, trained
          only on *previous* test seasons' out-of-sample predictions

Output: data/predictions.csv (long format, one row per match x model).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

RNG = 42
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["Date"])
df["match_id"] = np.arange(len(df))

TEST_SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
                "2022-23", "2023-24", "2024-25", "2025-26"]

DIFF_FEATS = ["d_ppg5", "d_ppg10", "d_gf5", "d_ga5", "d_sotf5", "d_sota5",
              "d_cor5", "d_vppg", "d_vgd", "d_rest", "d_played"]
RAW_FEATS = ["elo_home", "elo_away", "elo_diff", "h_ppg5", "a_ppg5",
             "h_vppg", "a_vppg", "h_hist_n", "a_hist_n"]
league_dum = pd.get_dummies(df["Div"], prefix="lg").astype(float)
X_ALL = pd.concat([df[DIFF_FEATS + RAW_FEATS], league_dum], axis=1)
FEATS = list(X_ALL.columns)
Y = df["y"].to_numpy()


def fit_logit(Xtr, ytr, Xte):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=RNG)),
    ])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)


def fit_gbm(Xtr, ytr, Xte):
    gbm = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=RNG,
    )
    gbm.fit(Xtr, ytr)
    return gbm.predict_proba(Xte)


# ---------------- Time-decayed double-Poisson goal model ----------------
XI = 0.0019  # per-day decay (Dixon & Coles 1997 order of magnitude)

def fit_dpois_league(tr, te):
    """Fit attack/defense/home-adv Poisson model on one league's training
    matches with exponential time decay; return outcome probs for test."""
    teams = sorted(set(tr.HomeTeam) | set(tr.AwayTeam))
    tix = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    n = len(tr)
    # two rows per match: home goals, away goals
    Xr = np.zeros((2 * n, 2 + 2 * T))
    yr = np.empty(2 * n)
    wr = np.empty(2 * n)
    days = (tr.Date.max() - tr.Date).dt.days.to_numpy()
    w = np.exp(-XI * days)
    for k, r in enumerate(tr.itertuples(index=False)):
        i, j = tix[r.HomeTeam], tix[r.AwayTeam]
        Xr[2 * k, 0] = 1; Xr[2 * k, 1] = 1                     # const + home
        Xr[2 * k, 2 + i] = 1; Xr[2 * k, 2 + T + j] = -1        # att_home - def_away
        yr[2 * k] = r.FTHG; wr[2 * k] = w[k]
        Xr[2 * k + 1, 0] = 1
        Xr[2 * k + 1, 2 + j] = 1; Xr[2 * k + 1, 2 + T + i] = -1
        yr[2 * k + 1] = r.FTAG; wr[2 * k + 1] = w[k]
    # identification: drop reference team's att and def columns
    keep = [0, 1] + [2 + i for i in range(1, T)] + [2 + T + i for i in range(1, T)]
    res = sm.GLM(yr, Xr[:, keep], family=sm.families.Poisson(), var_weights=wr).fit()
    beta = np.zeros(2 + 2 * T)
    beta[keep] = res.params
    att, dfn = beta[2:2 + T], beta[2 + T:]
    # unseen (promoted) teams: average of the 3 weakest training teams
    strength = att + dfn
    weak = np.argsort(strength)[:3]
    att_new, dfn_new = att[weak].mean(), dfn[weak].mean()

    def team_par(t):
        return (att[tix[t]], dfn[tix[t]]) if t in tix else (att_new, dfn_new)

    G = np.arange(0, 11)
    probs = np.empty((len(te), 3))
    from scipy.stats import poisson
    for k, r in enumerate(te.itertuples(index=False)):
        ah, dh = team_par(r.HomeTeam)
        aa, da = team_par(r.AwayTeam)
        mu_h = np.exp(beta[0] + beta[1] + ah - da)
        mu_a = np.exp(beta[0] + aa - dh)
        ph = poisson.pmf(G, min(mu_h, 8.0))
        pa = poisson.pmf(G, min(mu_a, 8.0))
        M = np.outer(ph, pa)
        M /= M.sum()
        probs[k] = [np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()]
    return probs


# ---------------- Walk-forward loop ----------------
records = []
oos_blend_train = []  # accumulated prior OOS predictions for the blend

for s in TEST_SEASONS:
    tr_mask = (df["season"] < s).to_numpy()
    te_mask = (df["season"] == s).to_numpy()
    if not tr_mask.any() or not te_mask.any():
        raise ValueError(f"Missing training or test matches for {s}; check raw season coverage.")
    if df.loc[tr_mask, "Date"].max() >= df.loc[te_mask, "Date"].min():
        raise ValueError(f"Training dates overlap test season {s}; season labels alone are insufficient.")
    Xtr, Xte = X_ALL[tr_mask], X_ALL[te_mask]
    ytr = Y[tr_mask]
    te_rows = df[te_mask]

    preds = {}
    preds["elo"] = fit_logit(df.loc[tr_mask, ["elo_diff"]], ytr, df.loc[te_mask, ["elo_diff"]])
    preds["logit"] = fit_logit(Xtr, ytr, Xte)
    preds["gbm"] = fit_gbm(Xtr, ytr, Xte)
    dp = np.empty((te_mask.sum(), 3))
    for div in df["Div"].unique():
        tr_l = df[tr_mask & (df["Div"] == div).to_numpy()]
        te_l_mask = (te_rows["Div"] == div).to_numpy()
        dp[te_l_mask] = fit_dpois_league(tr_l, te_rows[te_l_mask])
    preds["dpois"] = dp
    preds["market"] = te_rows[["shin_H", "shin_D", "shin_A"]].to_numpy()

    # Blend: trained on prior test seasons' OOS gbm + market predictions
    if oos_blend_train:
        bt = pd.concat(oos_blend_train, ignore_index=True)
        Zb = np.log(np.clip(bt[["gH", "gD", "gA", "mH", "mD", "mA"]].to_numpy(), 1e-6, 1))
        lr = LogisticRegression(max_iter=2000, C=10.0, random_state=RNG)
        lr.fit(Zb, bt["y"].to_numpy())
        Zt = np.log(np.clip(np.hstack([preds["gbm"], preds["market"]]), 1e-6, 1))
        preds["blend"] = lr.predict_proba(Zt)
    oos_blend_train.append(pd.DataFrame({
        "y": Y[te_mask],
        "gH": preds["gbm"][:, 0], "gD": preds["gbm"][:, 1], "gA": preds["gbm"][:, 2],
        "mH": preds["market"][:, 0], "mD": preds["market"][:, 1], "mA": preds["market"][:, 2],
    }))

    for model, P in preds.items():
        if not np.isfinite(P).all() or (P < 0).any() or not np.allclose(P.sum(axis=1), 1):
            raise ValueError(f"Invalid predicted probabilities for {model} in {s}.")
        rec = te_rows[["match_id", "Date", "season", "Div", "y",
                       "B365H", "B365D", "B365A", "mkt_H", "mkt_D", "mkt_A",
                       "mktC_H", "mktC_D", "mktC_A", "overround"]].copy()
        rec["model"] = model
        rec[["pH", "pD", "pA"]] = P
        records.append(rec)
    print(f"{s}: trained on {tr_mask.sum():5d}, predicted {te_mask.sum():4d} "
          f"({'blend ok' if 'blend' in preds else 'no blend yet'})")

pred = pd.concat(records, ignore_index=True)
pred.to_csv(ROOT / "data" / "predictions.csv", index=False)
print(f"\nSaved {len(pred)} prediction rows "
      f"({pred['match_id'].nunique()} matches x {pred['model'].nunique()} models)")
