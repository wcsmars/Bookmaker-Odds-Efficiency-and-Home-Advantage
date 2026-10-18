"""Result figures for the forecasting study."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
results = json.loads((ROOT / "output" / "results.json").read_text())

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
C = {"market": "black", "blend": "#4C72B0", "gbm": "#DD8452",
     "logit": "#55A868", "dpois": "#8172B3", "elo": "#937860"}
SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
           "2022-23", "2023-24", "2024-25", "2025-26"]
GHOST_X = (2.6, 3.45)  # brackets the 2020-21 marker on the season index axis;
                       # the 2019-20 marker also covers closed-stadium matches
# Earliest league closure (Serie A and Ligue 1) to the end of the policy window.
GHOST_DATES = (pd.Timestamp("2020-03-04"), pd.Timestamp("2021-06-30"))
LABELS = {"dpois": "double-Poisson", "blend": "blend (2018-19 on)"}


def shade(ax):
    ax.axvspan(*GHOST_X, color="#C44E52", alpha=0.10, lw=0)


# ---- Figure 1: RPS by season, and gap to market ----
rps = results["rps_by_season"]
fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1.4]})
x = np.arange(len(SEASONS))
ax = axes[0]
for mdl in ["market", "logit", "gbm", "dpois", "blend"]:
    v = [rps[mdl].get(s) for s in SEASONS]
    ax.plot(x, v, marker="o", ms=4, lw=1.6 if mdl in ("market", "blend") else 1.1,
            color=C[mdl], label=LABELS.get(mdl, mdl),
            ls="-" if mdl != "market" else "--")
shade(ax)
ax.set_ylabel("Mean RPS (lower = better)")
ax.set_title("(a) Forecast accuracy by season")
ax.legend(ncol=5, fontsize=8.5, frameon=False, loc="upper left")
ax = axes[1]
for mdl in ["logit", "gbm", "dpois", "blend"]:
    v = [None if rps[mdl].get(s) is None or rps["market"].get(s) is None
         else rps[mdl][s] - rps["market"][s] for s in SEASONS]
    ax.plot(x, v, marker="o", ms=4, lw=1.6 if mdl == "blend" else 1.1, color=C[mdl])
ax.axhline(0, color="black", lw=1.0, ls="--")
shade(ax)
ax.set_ylabel("RPS gap to market")
ax.set_title("(b) Gap to the bookmaker benchmark (below 0 = beats the market)")
ax.set_xticks(x, [s.replace("20", "", 1) for s in SEASONS])
ax.annotate("ghost\ngames", xy=(3.02, ax.get_ylim()[1] * 0.75), color="#C44E52",
            fontsize=8, ha="center")
fig.tight_layout()
fig.savefig(FIG / "fig1_rps_by_season.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: reliability curves (home win) ----
fig, ax = plt.subplots(figsize=(5.6, 5.0))
for mdl in ["market", "gbm", "blend"]:
    t = pd.read_csv(ROOT / "output" / "tables" / f"cal_{mdl}_home.csv")
    ece = results["ece"][f"{mdl}_home_ece"]
    ax.plot(t["p_mean"], t["o_rate"], marker="o", ms=4.5, lw=1.4, color=C[mdl],
            label=f"{mdl} (ECE {ece:.3f})")
ax.plot([0, 0.9], [0, 0.9], color="grey", lw=1.0, ls=":")
ax.set_xlabel("Forecast home-win probability (bin mean)")
ax.set_ylabel("Realized home-win frequency")
ax.set_title("Home-win reliability (blend sample starts in 2018-19)")
ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig2_reliability.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 3: market home-win bias by season (bridge to the home-advantage study) ----
bias = pd.DataFrame(results["market_bias_by_season"])
fig, ax = plt.subplots(figsize=(8.0, 4.2))
xb = np.arange(len(bias))
colors = ["#C44E52" if s in ("2020-21", "2024-25") else "#4C72B0" for s in bias["season"]]
ax.errorbar(xb, 100 * bias["bias"],
            yerr=[100 * (bias["bias"] - bias["ci_lo"]), 100 * (bias["ci_hi"] - bias["bias"])],
            fmt="none", ecolor="#888888", capsize=3, lw=1)
ax.scatter(xb, 100 * bias["bias"], c=colors, s=45, zorder=3)
ax.axhline(0, color="black", lw=1.0)
ax.set_xticks(xb, [s.replace("20", "", 1) for s in bias["season"]])
ax.set_ylabel("Realized − implied home-win rate (pp)")
ax.set_title("Market home-win calibration by season (pointwise 95% intervals)")
for i, s in enumerate(bias["season"]):
    if s in ("2020-21", "2024-25"):
        ax.annotate({"2020-21": "ghost season", "2024-25": "HA collapse"}[s],
                    xy=(i, 100 * bias.loc[i, "ci_lo"] - 0.55), ha="center",
                    fontsize=8.5, color="#C44E52")
ax.set_ylim(bottom=min(-6.2, ax.get_ylim()[0]))
fig.tight_layout()
fig.savefig(FIG / "fig3_market_bias.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 4: permutation importance ----
imp = pd.read_csv(ROOT / "output" / "tables" / "t4_permutation_importance.csv").head(10)
fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.barh(imp["feature"][::-1], imp["importance"][::-1],
        xerr=imp["std"][::-1], color="#DD8452", alpha=0.85)
ax.set_xlabel("Permutation importance (Δ log-loss, held-out 2025-26; bars: SD over 5 shuffles)")
ax.set_title("Elo difference has the largest GBM permutation importance")
fig.tight_layout()
fig.savefig(FIG / "fig4_importance.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 5: economics — ROI by threshold and cumulative profit ----
bt = results["backtest"]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
ax = axes[0]
thetas = [0.0, 0.02, 0.05, 0.10]
off = {"logit": -0.006, "gbm": -0.002, "dpois": 0.002, "blend": 0.006}
for mdl in ["logit", "gbm", "dpois", "blend"]:
    # A configuration with no bets is saved as null; plot it as a gap.
    r = [{k: np.nan if v is None else v for k, v in bt[mdl][f"theta_{t}"].items()}
         for t in thetas]
    xv = np.array(thetas) + off[mdl]
    ax.errorbar(xv, [100 * q["roi"] for q in r],
                yerr=[[100 * (q["roi"] - q["ci_lo"]) for q in r],
                      [100 * (q["ci_hi"] - q["roi"]) for q in r]],
                fmt="o", ms=4, capsize=3, color=C[mdl],
                label=LABELS.get(mdl, mdl))
ax.axhline(0, color="black", lw=1.0, ls="--")
overround = 100 * bt["mean_overround_test"]
ax.axhline(-overround, color="grey", lw=0.9, ls=":",
           label=f"− mean overround ({overround:.2f}%)")
ax.set_xlabel("Value threshold θ")
ax.set_ylabel("ROI, % (flat stakes)")
ax.set_title("(a) Backtest ROI at Bet365 odds, 95% CI")
ax.legend(fontsize=8, frameon=False, loc="lower left")

ax = axes[1]
pred = pd.read_csv(ROOT / "data" / "predictions_scored.csv", parse_dates=["Date"])
g = pred[pred["model"] == "gbm"].sort_values("Date")
prof = []
for o, ycls in [("H", 0), ("D", 1), ("A", 2)]:
    p = g[f"p{o}"].to_numpy() * g[f"B365{o}"].to_numpy() - 1
    sel = p > 0.05
    pr = np.where(g["y"].to_numpy()[sel] == ycls, g[f"B365{o}"].to_numpy()[sel] - 1, -1.0)
    prof.append(pd.DataFrame({"Date": g["Date"].to_numpy()[sel], "profit": pr}))
prof = pd.concat(prof).sort_values("Date")
cum = prof.groupby("Date")["profit"].sum().cumsum()
ax.plot(cum.index, cum.values, color="#DD8452", lw=1.4)
ax.axhline(0, color="black", lw=1.0, ls="--")
ax.axvspan(*GHOST_DATES, color="#C44E52", alpha=0.10, lw=0)
ax.set_ylabel("Cumulative units (GBM, θ=0.05)")
ax.set_title("(b) Cumulative realized profit")
fig.tight_layout()
fig.savefig(FIG / "fig5_economics.png", bbox_inches="tight")
plt.close(fig)

print("Saved:", sorted(p.name for p in FIG.glob("*.png")))
