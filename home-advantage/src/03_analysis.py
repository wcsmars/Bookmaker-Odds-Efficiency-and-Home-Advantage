"""Estimate closed-stadium period differences in home advantage and discipline.

Model (paired-comparison home-advantage regression, cf. Clarke & Norman 1995):

    y_m = d_m' beta + s_{home(m)} - s_{away(m)} + eps_m

where s are team-season strength fixed effects entering +1 for the home side
and -1 for the away side, and d_m holds an intercept (= the pre-COVID home
advantage) plus crowd-period dummies. Because the strengths are sign-coded,
period dummies remain identified even in seasons played entirely without
crowds. Estimation by Frisch-Waugh-Lovell: the FE block is partialled out of
(y, d) with sparse least squares, then OLS on residualized variables with
league-season cluster-robust standard errors. Single-season and single-league
specifications cluster by league matchday instead (see fe_reg).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
OUT_T = ROOT / "output" / "tables"
OUT_T.mkdir(parents=True, exist_ok=True)

m = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", parse_dates=["Date"])
m["season_short"] = m["season"].str.replace("20", "", 1)
# League matchday (league x calendar date): the clustering unit when league-season
# clusters are too few. Home/away team-season clusters coincide with the absorbed
# strength effects and understate uncertainty in this paired design.
m["matchday"] = m["Div"] + "_" + m["Date"].dt.strftime("%Y-%m-%d")

# Post-period season dummies for the recovery analysis
for s in ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]:
    m[f"post_{s[2:4]}{s[5:]}"] = ((m["season"] == s)).astype(int)
m["ghost_loose"] = ((m["ghost"] == 1) & (m["ghost_strict"] == 0)).astype(int)

OUTCOMES = {
    "margin": "Goal margin (H-A)",
    "points_diff": "Points margin (H-A)",
    "home_win": "Home win (LPM)",
    "yellow_diff": "Yellow cards (A-H)",
    "foul_diff": "Fouls (A-H)",
    "red_diff": "Red cards (A-H)",
    "shots_diff": "Shots (H-A)",
    "sot_diff": "Shots on target (H-A)",
    "corner_diff": "Corners (H-A)",
}


def strength_block(df):
    """Sparse +1/-1 team-season paired-comparison design."""
    cats = pd.Index(sorted(set(df["home_ts"]) | set(df["away_ts"])))
    hi = cats.get_indexer(df["home_ts"])
    ai = cats.get_indexer(df["away_ts"])
    n = len(df)
    rows = np.concatenate([np.arange(n), np.arange(n)])
    cols = np.concatenate([hi, ai])
    vals = np.concatenate([np.ones(n), -np.ones(n)])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, len(cats)))


def fe_reg(df, y, dvars, cluster="league_season", contrast=None):
    """FWL regression of y on [const + dvars] absorbing strength FE.

    SEs are cluster-robust with a degrees-of-freedom correction that counts
    every absorbed FE parameter (deliberately conservative relative to
    reghdfe's default, which exempts FEs nested within clusters); inference
    uses t(G-1) where G = #clusters. If the requested cluster variable has
    fewer than 10 groups (e.g. the single-season spec), clustering falls back
    to the league matchday. Team-season clusters are avoided: with sign-coded
    team-season strengths, their score sums largely cancel and understate the
    sampling variance.
    `contrast=(a, b)` additionally tests H0: beta_a = beta_b.
    """
    from scipy import stats as st

    sub = df.dropna(subset=[y]).copy()
    if sub[cluster].nunique() < 10:
        cluster = "matchday"
    names = ["home_adv_const"] + dvars
    y_t, D_t = _fwl(sub, y, dvars)

    res = sm.OLS(y_t, D_t).fit(
        cov_type="cluster", cov_kwds={"groups": sub[cluster]}, use_t=True
    )
    n, k_d = len(sub), D_t.shape[1]
    # Rank of the FE block: one strength parameter per team-season, less one
    # normalization per league-season (each league-season is connected).
    n_team_seasons = len(set(sub["home_ts"]) | set(sub["away_ts"]))
    k_fe = n_team_seasons - sub["league_season"].nunique()
    dof_factor = np.sqrt((n - k_d) / (n - k_d - k_fe))
    G = sub[cluster].nunique()
    tcrit = st.t.ppf(0.975, G - 1)
    out = {}
    for i, nm in enumerate(names):
        se = float(res.bse[i]) * dof_factor
        coef = float(res.params[i])
        tval = coef / se
        out[nm] = {
            "coef": coef,
            "se": se,
            "p": float(2 * st.t.sf(abs(tval), G - 1)),
            "ci_lo": coef - tcrit * se,
            "ci_hi": coef + tcrit * se,
        }
    result = {"n": int(len(sub)), "n_clusters": int(G), "coefs": out}
    if contrast is not None:
        a, b = (names.index(v) for v in contrast)
        cov = np.asarray(res.cov_params()) * dof_factor**2
        diff = float(res.params[a] - res.params[b])
        se_d = float(np.sqrt(cov[a, a] + cov[b, b] - 2 * cov[a, b]))
        result["contrast"] = {
            "test": f"{contrast[0]} = {contrast[1]}",
            "diff": diff,
            "se": se_d,
            "p": float(2 * st.t.sf(abs(diff / se_d), G - 1)),
        }
    return result


def _fwl(sub, y, dvars):
    """Residualize [y, const, dvars] on the strength FE block; return (y_t, D_t)."""
    F = strength_block(sub)
    D = np.column_stack([np.ones(len(sub))] + [sub[v].values.astype(float) for v in dvars])
    M = np.column_stack([sub[y].values.astype(float), D])
    R = np.empty_like(M)
    for j in range(M.shape[1]):
        sol = lsqr(F, M[:, j], atol=1e-12, btol=1e-12, iter_lim=50000)[0]
        R[:, j] = M[:, j] - F @ sol
    return R[:, 0], R[:, 1:]


def wild_cluster_p(df, y, dvars, target, cluster="league_season", B=999, seed=20260612):
    """Wild cluster bootstrap p-value (Webb weights, null imposed) for one
    coefficient. Used for pooled specifications with few treated clusters.
    FE are nested within the bootstrap clusters, so the FWL-residualized
    system can be bootstrapped directly (Roodman et al. 2019 convention)."""
    sub = df.dropna(subset=[y]).copy()
    y_t, D_t = _fwl(sub, y, dvars)
    names = ["home_adv_const"] + dvars
    k = names.index(target)
    groups = sub[cluster].values
    gids = pd.factorize(groups)[0]
    G = gids.max() + 1

    def cluster_t(yv):
        beta, *_ = np.linalg.lstsq(D_t, yv, rcond=None)
        u = yv - D_t @ beta
        XtXi = np.linalg.inv(D_t.T @ D_t)
        meat = np.zeros((D_t.shape[1], D_t.shape[1]))
        for g in range(G):
            sg = D_t[gids == g].T @ u[gids == g]
            meat += np.outer(sg, sg)
        V = XtXi @ meat @ XtXi
        return beta[k] / np.sqrt(V[k, k])

    t_obs = cluster_t(y_t)
    # Restricted fit (beta_target = 0)
    Dr = np.delete(D_t, k, axis=1)
    br, *_ = np.linalg.lstsq(Dr, y_t, rcond=None)
    fit_r, u_r = Dr @ br, y_t - Dr @ br
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(B):
        w = webb[rng.integers(0, 6, G)][gids]
        if abs(cluster_t(fit_r + u_r * w)) >= abs(t_obs):
            hits += 1
    return float((hits + 1) / (B + 1))


def run_spec(df, dvars, outcomes, label, cluster="league_season"):
    res = {}
    for y in outcomes:
        res[y] = fe_reg(df, y, dvars, cluster=cluster)
    print(f"  spec '{label}' done ({len(outcomes)} outcomes)")
    return res


results = {}

# ---------- Descriptive statistics (Table 1) ----------
def descriptives(df):
    g = df.groupby("period")
    t = pd.DataFrame({
        "Matches": g.size(),
        "Home win %": g["home_win"].mean() * 100,
        "Draw %": g.apply(lambda x: (x["margin"] == 0).mean() * 100, include_groups=False),
        "Away win %": g["away_win"].mean() * 100,
        "Home goals": g["FTHG"].mean(),
        "Away goals": g["FTAG"].mean(),
        "Goal margin": g["margin"].mean(),
        "Points margin": g["points_diff"].mean(),
        "Home points": g["home_pts"].mean(),
        "Yellows A-H": g["yellow_diff"].mean(),
        "Fouls A-H": g["foul_diff"].mean(),
        "Implied home prob": g["implied_home_prob"].mean(),
    }).loc[["pre", "ghost", "post"]]
    return t

t1 = descriptives(m)
t1.round(3).to_csv(OUT_T / "table1_descriptives.csv")
print("Table 1:\n", t1.round(3).to_string(), "\n")
results["descriptives"] = json.loads(t1.to_json())

# Per-league descriptive home advantage by period
tA = (m.groupby(["league", "period"])["margin"].mean().unstack()[["pre", "ghost", "post"]])
tA.round(3).to_csv(OUT_T / "tableA1_league_margin.csv")
results["league_margin_by_period"] = json.loads(tA.to_json())

# Season-level home stats (for figures and the recovery narrative)
season_stats = m.groupby("season").agg(
    home_win=("home_win", "mean"), margin=("margin", "mean"),
    yellow_diff=("yellow_diff", "mean"), implied=("implied_home_prob", "mean"),
)
season_stats.round(4).to_csv(OUT_T / "season_stats.csv")
results["season_stats"] = json.loads(season_stats.to_json())

# ---------- Main specs ----------
print("Main spec: ghost + post")
results["main"] = run_spec(m, ["ghost", "post"], list(OUTCOMES), "main")

print("Recovery spec: ghost + post x season (with ghost = 2024-25 Wald test)")
POST_DUMMIES = ["post_2122", "post_2223", "post_2324", "post_2425", "post_2526"]
results["recovery"] = {
    y: fe_reg(m, y, ["ghost"] + POST_DUMMIES, contrast=("ghost", "post_2425"))
    for y in ["margin", "points_diff", "home_win", "yellow_diff", "foul_diff"]
}
# Second contrast: is the 2025-26 level distinguishable from the ghost trough?
results["recovery_2526_contrast"] = fe_reg(
    m, "margin", ["ghost"] + POST_DUMMIES, contrast=("ghost", "post_2526")
)["contrast"]
print("  spec 'recovery' done")

print("Recovery spec excluding Omicron-restricted post matches")
m_clean = m[m["omicron_restricted"] == 0]
results["recovery_clean"] = run_spec(
    m_clean, ["ghost"] + POST_DUMMIES, ["margin", "home_win", "yellow_diff"],
    "recovery_clean",
)

print("Strict window spec: ghost_strict + ghost_loose + post")
results["strict"] = run_spec(
    m, ["ghost_strict", "ghost_loose", "post"],
    ["margin", "points_diff", "home_win", "yellow_diff", "foul_diff"], "strict",
)

print("Within 2019-20 spec (cleanest within-season identification; matchday clusters)")
s1920 = m[m["season"] == "2019-20"]
results["s1920"] = run_spec(
    s1920, ["ghost"], ["margin", "points_diff", "home_win", "yellow_diff", "foul_diff"], "s1920",
    cluster="matchday",
)

print("Placebo: fake ghost window 2017-03-04..2018-06-30, pre-COVID data only")
pre = m[m["period"] == "pre"].copy()
pre["fake_ghost"] = pre["Date"].between("2017-03-04", "2018-06-30").astype(int)
results["placebo"] = run_spec(
    pre, ["fake_ghost"], ["margin", "points_diff", "home_win", "yellow_diff", "foul_diff"], "placebo",
)

print("League-by-league main spec (league-matchday clustering: within a single")
print("league the ghost period spans only ~2 league-seasons, so league-season")
print("clustering has too few treated clusters for reliable inference)")
results["by_league"] = {}
for div, sub in m.groupby("league"):
    results["by_league"][div] = {
        y: fe_reg(sub, y, ["ghost", "post"], cluster="matchday")
        for y in ["margin", "home_win", "yellow_diff"]
    }
    g = results["by_league"][div]["margin"]["coefs"]["ghost"]
    yw = results["by_league"][div]["yellow_diff"]["coefs"]["ghost"]
    print(f"  {div}: margin {g['coef']:+.3f} (p={g['p']:.3f}), yellow {yw['coef']:+.3f} (p={yw['p']:.3f})")

print("Pooled wild-cluster bootstrap (Webb, null imposed, league-season clusters)")
results["pooled_wild"] = {
    "ghost_margin": wild_cluster_p(m, "margin", ["ghost", "post"], "ghost"),
    "ghost_yellow": wild_cluster_p(m, "yellow_diff", ["ghost", "post"], "ghost"),
    "post2425_margin": wild_cluster_p(
        m, "margin", ["ghost"] + POST_DUMMIES, "post_2425"
    ),
    "post2122_margin": wild_cluster_p(
        m, "margin", ["ghost"] + POST_DUMMIES, "post_2122"
    ),
    "post2526_margin": wild_cluster_p(
        m, "margin", ["ghost"] + POST_DUMMIES, "post_2526"
    ),
}
print(f"  {results['pooled_wild']}")

# Bookmaker anticipation: implied home prob with FE (market's view of home edge)
print("Bookmaker-implied probability spec")
results["bookmaker"] = run_spec(
    m, ["ghost", "post"], ["implied_home_prob"], "bookmaker",
)

# ---------- Formatted coefficient tables ----------
def fmt_table(spec_res, dvars, outcomes, fname):
    rows = []
    for y in outcomes:
        r = spec_res[y]
        row = {"Outcome": OUTCOMES.get(y, y), "N": r["n"]}
        for v in ["home_adv_const"] + dvars:
            c = r["coefs"][v]
            stars = "***" if c["p"] < 0.01 else "**" if c["p"] < 0.05 else "*" if c["p"] < 0.1 else ""
            row[v] = f"{c['coef']:.3f}{stars} ({c['se']:.3f})"
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT_T / f"{fname}.csv", index=False)
    print(t.to_string(index=False), "\n")
    return t

print("\n=== Table 2: main results ===")
fmt_table(results["main"], ["ghost", "post"], list(OUTCOMES), "table2_main")
print("=== Table 3: recovery by season ===")
fmt_table(results["recovery"], ["ghost"] + POST_DUMMIES,
          ["margin", "home_win", "yellow_diff"], "table3_recovery")
print("=== Table 4: robustness ===")
fmt_table(results["strict"], ["ghost_strict", "ghost_loose", "post"],
          ["margin", "home_win", "yellow_diff"], "table4_strict")
fmt_table(results["s1920"], ["ghost"], ["margin", "home_win", "yellow_diff"], "table4_s1920")
fmt_table(results["placebo"], ["fake_ghost"], ["margin", "home_win", "yellow_diff"], "table4_placebo")

# Traceable sample counts and the pre-pandemic linear trend in home advantage
results["counts"] = {
    "matches": int(len(m)),
    "ghost": int(m["ghost"].sum()),
    "ghost_strict": int(m["ghost_strict"].sum()),
    "ghost_1920": int(m[(m["season"] == "2019-20") & (m["ghost"] == 1)].shape[0]),
    "omicron_restricted": int(m["omicron_restricted"].sum()),
}
pre_means = (m[m["season"] <= "2018-19"].groupby("season")["margin"].mean())
slope = float(np.polyfit(np.arange(len(pre_means)), pre_means.values, 1)[0])
results["pre_trend_slope_per_season"] = slope
# Season-to-season volatility of the pooled home margin, pre vs post pandemic
post_means = (m[m["season"] >= "2021-22"].groupby("season")["margin"].mean())
results["margin_volatility"] = {
    "pre_sd": float(pre_means.std(ddof=1)),
    "post_sd": float(post_means.std(ddof=1)),
    "pre_seasons": list(pre_means.index),
    "post_seasons": list(post_means.index),
}
print(f"Counts: {results['counts']}")
print(f"Pre-pandemic trend in season-mean margin: {slope:+.4f} goals/season")
rec = results["recovery"]["margin"]["contrast"]
print(f"Wald ghost=post_2425 (margin): diff={rec['diff']:+.3f}, p={rec['p']:.3f}")
rc = results["recovery_clean"]["margin"]["coefs"]
print(f"Omicron-clean post_2122 margin: {rc['post_2122']['coef']:+.3f} (se {rc['post_2122']['se']:.3f})")

with open(ROOT / "output" / "results.json", "w") as f:
    json.dump(results, f, indent=1)
print("Saved output/results.json")
