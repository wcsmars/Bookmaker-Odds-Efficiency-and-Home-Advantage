"""Build the match-level feature table for the forecasting study.

Leakage discipline: every feature for match m is computed strictly from
matches completed before m's date — Elo ratings are updated sequentially
after each match, rolling form uses only prior fixtures, and no odds enter
any model feature (odds are carried separately for the market benchmark,
the blend, and the backtest).

Input: ../home-advantage/data/raw/*.csv (12 seasons x 5 leagues)
Output: data/features.csv (one row per match, 2014-15..2025-26)
"""
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "home-advantage" / "data" / "raw"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

# ---------------- Load & clean ----------------
frames = []
raw_files = sorted(
    RAW / f"{league}_{year:02d}{year + 1:02d}.csv"
    for league in ("E0", "D1", "SP1", "I1", "F1")
    for year in range(14, 26)
)
missing_files = [f.name for f in raw_files if not f.is_file() or not f.stat().st_size]
if missing_files:
    raise FileNotFoundError(
        "Incomplete raw sample; see DATA_SOURCES.md. Missing: " + ", ".join(missing_files)
    )
for f in raw_files:
    div, season = f.stem.split("_")
    df = pd.read_csv(f, encoding="latin-1")
    df = df.rename(columns=lambda c: c.replace("﻿", "").replace("ï»¿", "").strip())
    df = df.dropna(how="all")
    df["season"] = "20" + season[:2] + "-" + season[2:]
    # Unify market-average odds columns (BbAv* pre-2019, Avg* after)
    for tgt, old, new in [("avgH", "BbAvH", "AvgH"), ("avgD", "BbAvD", "AvgD"), ("avgA", "BbAvA", "AvgA")]:
        df[tgt] = df[new] if new in df.columns else (df[old] if old in df.columns else np.nan)
    keep = ["Div", "season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "HS", "AS", "HST", "AST", "HC", "AC",
            "B365H", "B365D", "B365A", "avgH", "avgD", "avgA"]
    for c in ["B365CH", "B365CD", "B365CA"]:  # closing odds, 2019-20 onward
        df[c] = df[c] if c in df.columns else np.nan
    frames.append(df[keep + ["B365CH", "B365CD", "B365CA"]])

m = pd.concat(frames, ignore_index=True)
m["Date"] = pd.to_datetime(m["Date"], dayfirst=True, format="mixed")
# Every completed match updates the sequential Elo and form state below. Rows
# without a complete Bet365 price are excluded only from the written feature
# table, so the market benchmark and the models share one evaluation sample.
m = m.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
m = m.sort_values(["Date", "Div", "HomeTeam"]).reset_index(drop=True)
m["y"] = m["FTR"].map({"H": 0, "D": 1, "A": 2})
if m["y"].isna().any():
    raise ValueError("Unrecognized full-time result; expected H, D, or A.")
m["y"] = m["y"].astype(int)
has_odds = m[["B365H", "B365D", "B365A"]].notna().all(axis=1)
if (m.loc[has_odds, ["B365H", "B365D", "B365A"]] <= 1).any().any():
    raise ValueError("Pre-match decimal odds must be greater than one.")
# With date-only ordering, sequential updates require at most one fixture per
# team per date. Refuse ambiguous inputs rather than use same-day outcomes.
team_dates = pd.concat([
    m[["Date", "Div", "HomeTeam"]].rename(columns={"HomeTeam": "team"}),
    m[["Date", "Div", "AwayTeam"]].rename(columns={"AwayTeam": "team"}),
])
if team_dates.duplicated(["Date", "Div", "team"]).any():
    raise ValueError("A team appears twice on one date; resolve duplicate/ambiguous fixtures first.")

# ---------------- Implied market probabilities ----------------
def shin_probs(inv):
    """Shin (1993) de-margined probabilities from inverse odds (basic
    normalization fallback when the solver cannot bracket a root)."""
    s = inv.sum()
    def f(z):
        return np.sum((np.sqrt(z**2 + 4 * (1 - z) * inv**2 / s) - z) / (2 * (1 - z))) - 1
    try:
        z = brentq(f, 0.0, 0.2, xtol=1e-10)
        return (np.sqrt(z**2 + 4 * (1 - z) * inv**2 / s) - z) / (2 * (1 - z))
    except ValueError:
        return inv / s

inv = 1.0 / m[["B365H", "B365D", "B365A"]].to_numpy()
with np.errstate(invalid="ignore"):
    m[["mkt_H", "mkt_D", "mkt_A"]] = inv / inv.sum(axis=1, keepdims=True)
shin = np.full_like(inv, np.nan)
shin[has_odds.to_numpy()] = np.vstack([shin_probs(row) for row in inv[has_odds.to_numpy()]])
m[["shin_H", "shin_D", "shin_A"]] = shin
m["overround"] = inv.sum(axis=1) - 1

inv_c = 1.0 / m[["B365CH", "B365CD", "B365CA"]].to_numpy()
with np.errstate(invalid="ignore"):
    m[["mktC_H", "mktC_D", "mktC_A"]] = inv_c / inv_c.sum(axis=1, keepdims=True)

# ---------------- Sequential Elo (per league) ----------------
ELO_ENTRY, ELO_HA, ELO_K = 1400.0, 60.0, 20.0

elo = {}
last_season_seen = {}
elo_h_col, elo_a_col = np.empty(len(m)), np.empty(len(m))
# Row index -> clubs whose rating is reset at that match. The rolling-form
# pass below clears their form history at the same match, so a club returning
# after an absence is treated like a newly observed club rather than carrying
# form from a stint at least one season earlier.
resets = defaultdict(list)

def get_elo(div, team, season, row):
    key = (div, team)
    if key not in elo:
        elo[key] = ELO_ENTRY
    elif last_season_seen.get(key) is not None:
        prev, cur = last_season_seen[key], season
        if prev != cur and int(cur[:4]) - int(prev[:4]) > 1:
            elo[key] = ELO_ENTRY  # returned after an absence: reset
            resets[row].append(key)
    last_season_seen[key] = season
    return elo[key]

for i, r in enumerate(m.itertuples(index=False)):
    ra = get_elo(r.Div, r.HomeTeam, r.season, i)
    rb = get_elo(r.Div, r.AwayTeam, r.season, i)
    elo_h_col[i], elo_a_col[i] = ra, rb
    exp_home = 1.0 / (1.0 + 10 ** (-((ra + ELO_HA) - rb) / 400.0))
    res = 1.0 if r.FTHG > r.FTAG else (0.5 if r.FTHG == r.FTAG else 0.0)
    d = abs(int(r.FTHG) - int(r.FTAG))
    g = 1.0 if d <= 1 else (1.5 if d == 2 else (11 + d) / 8.0)
    delta = ELO_K * g * (res - exp_home)
    elo[(r.Div, r.HomeTeam)] = ra + delta
    elo[(r.Div, r.AwayTeam)] = rb - delta

m["elo_home"], m["elo_away"] = elo_h_col, elo_a_col
m["elo_diff"] = m["elo_home"] + ELO_HA - m["elo_away"]

# ---------------- Rolling form (strictly past matches) ----------------
hist = defaultdict(lambda: deque(maxlen=10))        # all venues
hist_venue = defaultdict(lambda: deque(maxlen=5))   # venue-specific
last_date = {}
season_played = defaultdict(int)
seen_season = {}

def roll_stats(key, n):
    rows = list(hist[key])[-n:]
    if not rows:
        return dict(ppg=np.nan, gf=np.nan, ga=np.nan, sotf=np.nan, sota=np.nan, cor=np.nan, n=0)
    a = np.array(rows, dtype=float)  # cols: pts, gf, ga, sot_f, sot_a, cor
    return dict(ppg=a[:, 0].mean(), gf=a[:, 1].mean(), ga=a[:, 2].mean(),
                sotf=a[:, 3].mean(), sota=a[:, 4].mean(), cor=a[:, 5].mean(), n=len(a))

def venue_stats(key):
    rows = list(hist_venue[key])
    if not rows:
        return dict(vppg=np.nan, vgd=np.nan)
    a = np.array(rows, dtype=float)  # cols: pts, gd
    return dict(vppg=a[:, 0].mean(), vgd=a[:, 1].mean())

rows_out = []
for i, r in enumerate(m.itertuples(index=False)):
    hk, ak = (r.Div, r.HomeTeam), (r.Div, r.AwayTeam)
    for key in resets.get(i, ()):  # rating reset for a returning club
        hist.pop(key, None)
        hist_venue.pop((key, "H"), None)
        hist_venue.pop((key, "A"), None)
        last_date.pop(key, None)
    # season match counters
    for k, team in [(hk, r.HomeTeam), (ak, r.AwayTeam)]:
        if seen_season.get(k) != r.season:
            seen_season[k] = r.season
            season_played[k] = 0
    f = {}
    for side, k in [("h", hk), ("a", ak)]:
        s5, s10 = roll_stats(k, 5), roll_stats(k, 10)
        v = venue_stats((k, "H" if side == "h" else "A"))
        f[f"{side}_ppg5"], f[f"{side}_ppg10"] = s5["ppg"], s10["ppg"]
        f[f"{side}_gf5"], f[f"{side}_ga5"] = s5["gf"], s5["ga"]
        f[f"{side}_sotf5"], f[f"{side}_sota5"] = s5["sotf"], s5["sota"]
        f[f"{side}_cor5"] = s5["cor"]
        f[f"{side}_hist_n"] = s10["n"]
        f[f"{side}_vppg"], f[f"{side}_vgd"] = v["vppg"], v["vgd"]
        ld = last_date.get(k)
        f[f"{side}_rest"] = min((r.Date - ld).days, 30) if ld is not None else np.nan
        f[f"{side}_played"] = season_played[k]
    rows_out.append(f)
    # ---- update state AFTER computing features ----
    hp = 3 if r.FTHG > r.FTAG else (1 if r.FTHG == r.FTAG else 0)
    ap = 3 if r.FTAG > r.FTHG else (1 if r.FTHG == r.FTAG else 0)
    hst = r.HST if not pd.isna(r.HST) else 0
    ast = r.AST if not pd.isna(r.AST) else 0
    hc = r.HC if not pd.isna(r.HC) else 0
    ac = r.AC if not pd.isna(r.AC) else 0
    hist[hk].append((hp, r.FTHG, r.FTAG, hst, ast, hc))
    hist[ak].append((ap, r.FTAG, r.FTHG, ast, hst, ac))
    hist_venue[(hk, "H")].append((hp, r.FTHG - r.FTAG))
    hist_venue[(ak, "A")].append((ap, r.FTAG - r.FTHG))
    last_date[hk] = r.Date
    last_date[ak] = r.Date
    season_played[hk] += 1
    season_played[ak] += 1

feat = pd.DataFrame(rows_out)
out = pd.concat([m.reset_index(drop=True), feat], axis=1)

# Convenience differentials
for c in ["ppg5", "ppg10", "gf5", "ga5", "sotf5", "sota5", "cor5", "vppg", "vgd", "rest", "played"]:
    out[f"d_{c}"] = out[f"h_{c}"] - out[f"a_{c}"]

# The evaluation sample requires a complete pre-match Bet365 price.
dropped = int((~has_odds).sum())
out = out[has_odds.to_numpy()].reset_index(drop=True)
out.to_csv(OUT / "features.csv", index=False)
print(f"Matches: {len(out)}   Date range: {out['Date'].min().date()} .. {out['Date'].max().date()}")
print(f"Completed matches without a complete Bet365 price (used for Elo/form only): {dropped}")
print(f"Seasons: {out['season'].nunique()}   Leagues: {out['Div'].nunique()}")
print(f"Closing odds available: {out['mktC_H'].notna().sum()} matches (from {out.loc[out['mktC_H'].notna(), 'season'].min()})")
print(f"Mean overround: {out['overround'].mean():.4f}")
print("Elo sanity (top-5 end ratings):")
top = sorted(elo.items(), key=lambda kv: -kv[1])[:5]
for (dv, t), v in top:
    print(f"  {t} ({dv}): {v:.0f}")
print("NaN share in key features:")
print(out[["d_ppg5", "d_vppg", "d_rest", "elo_diff"]].isna().mean().round(4).to_string())
