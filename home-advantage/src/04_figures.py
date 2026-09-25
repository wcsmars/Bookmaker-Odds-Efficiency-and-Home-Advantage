"""Result figures. Reads data/processed/matches.csv and output/results.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

m = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", parse_dates=["Date"])
results = json.loads((ROOT / "output" / "results.json").read_text())

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
# Earliest league closure (Serie A and Ligue 1) to the end of the policy window.
GHOST_SPAN = (pd.Timestamp("2020-03-04"), pd.Timestamp("2021-06-30"))
C_PRE, C_GHOST, C_POST = "#4C72B0", "#C44E52", "#55A868"
LEAGUE_COLORS = {
    "Premier League": "#4C72B0", "La Liga": "#DD8452", "Serie A": "#55A868",
    "Bundesliga": "#C44E52", "Ligue 1": "#8172B3",
}


def shade_ghost(ax):
    ax.axvspan(*GHOST_SPAN, color="#C44E52", alpha=0.12, lw=0)


# ---- Figure 1: rolling home advantage over time ----
fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
d = m.sort_values("Date").set_index("Date")
roll_margin = d["margin"].rolling(500, center=True, min_periods=300).mean()
roll_yellow = d["yellow_diff"].rolling(500, center=True, min_periods=300).mean()

ax = axes[0]
ax.plot(roll_margin.index, roll_margin.values, color=C_PRE, lw=1.6)
ax.axhline(0, color="black", lw=0.8)
shade_ghost(ax)
ax.set_ylabel("Goal margin, home − away")
ax.set_title("(a) Home goal-margin advantage, 500-match rolling mean")

ax = axes[1]
ax.plot(roll_yellow.index, roll_yellow.values, color="#937860", lw=1.6)
ax.axhline(0, color="black", lw=0.8)
shade_ghost(ax)
ax.set_ylabel("Yellow cards, away − home")
ax.set_title("(b) Home yellow-card advantage, 500-match rolling mean")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.annotate("ghost games", xy=(pd.Timestamp("2020-09-15"), ax.get_ylim()[1] * 0.9),
            color="#C44E52", fontsize=9, ha="center")
fig.suptitle("Home advantage fell during the closed-stadium period", y=0.995, fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fig1_rolling.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: season-level home margin by league ----
fig, ax = plt.subplots(figsize=(8.5, 4.6))
seasons = sorted(m["season"].unique())
x = np.arange(len(seasons))
for lg, sub in m.groupby("league"):
    s = sub.groupby("season")["margin"].mean().reindex(seasons)
    ax.plot(x, s.values, marker="o", ms=3.5, lw=1.2, label=lg,
            color=LEAGUE_COLORS[lg], alpha=0.75)
pooled = m.groupby("season")["margin"].mean().reindex(seasons)
ax.plot(x, pooled.values, marker="o", ms=5, lw=2.6, color="black", label="Big-5 pooled")
ax.axhline(0, color="black", lw=0.8)
ghost_i = seasons.index("2020-21")
ax.axvspan(ghost_i - 0.4, ghost_i + 0.45, color="#C44E52", alpha=0.12, lw=0)
ax.text(ghost_i, 0.97, "closed\nstadiums", color="#C44E52", ha="center", va="top",
        fontsize=8, transform=ax.get_xaxis_transform())
ax.set_xticks(x, [s.replace("20", "", 1) for s in seasons], rotation=0)
ax.set_ylabel("Mean goal margin, home − away")
ax.set_title(f"Season-level home advantage by league, {seasons[0]} to {seasons[-1]}")
top = max(0.55, float(m.groupby(["league", "season"])["margin"].mean().max()) + 0.03)
ax.set_ylim(top=top)
ax.legend(ncol=2, fontsize=8, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "fig2_seasons.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 4: coefficient plot, recovery spec ----
rec = results["recovery"]
labels = ["Ghost\n(2020-03–2021-06)", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
keys = ["ghost", "post_2122", "post_2223", "post_2324", "post_2425", "post_2526"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
for ax, (y, ttl, col) in zip(axes, [
    ("margin", "(a) Goal margin (home − away)", C_PRE),
    ("yellow_diff", "(b) Yellow cards (away − home)", "#937860"),
]):
    co = rec[y]["coefs"]
    est = [co[k]["coef"] for k in keys]
    lo = [co[k]["ci_lo"] for k in keys]
    hi = [co[k]["ci_hi"] for k in keys]
    ypos = np.arange(len(keys))[::-1]
    ax.errorbar(est, ypos, xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
                fmt="o", color=col, ecolor=col, capsize=3, ms=5)
    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_yticks(ypos, labels)
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("Difference from pre-COVID level")
fig.suptitle("Period differences relative to pre-COVID baseline (95% CI, team-season FE)",
             fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig4_coefplot.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 3: referee outcomes by period ----
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
periods = ["pre", "ghost", "post"]
pal = [C_PRE, C_GHOST, C_POST]
for ax, (var, ttl) in zip(axes, [
    ("yellow_diff", "(a) Yellow cards, away − home"),
    ("foul_diff", "(b) Fouls, away − home"),
]):
    by = m.groupby(["league", "period"])[var].mean().unstack()[periods]
    xb = np.arange(len(by.index))
    w = 0.26
    for i, p in enumerate(periods):
        ax.bar(xb + (i - 1) * w, by[p], width=w, color=pal[i],
               label={"pre": "Pre-COVID", "ghost": "Ghost", "post": "Post"}[p])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xb, [l.replace(" ", "\n") for l in by.index], fontsize=8)
    ax.set_title(ttl, fontsize=10)
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("Away-minus-home card and foul differentials fell during the closed-stadium "
             "period (unadjusted league means)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig3_referee.png", bbox_inches="tight")
plt.close(fig)

print("Saved:", sorted(p.name for p in FIG.glob("*.png")))
