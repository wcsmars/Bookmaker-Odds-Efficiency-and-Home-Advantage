"""Build the analysis dataset from raw football-data.co.uk files.

Output: data/processed/matches.csv, one row per match, with outcome variables
and period flags (pre / ghost / post, plus a narrower-window sensitivity).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CORE = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
STATS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
ODDS = ["B365H", "B365D", "B365A"]
LEAGUE_NAMES = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

SEASONS = [f"{year:02d}{year + 1:02d}" for year in range(14, 26)]
EXPECTED_FILES = [RAW / f"{div}_{season}.csv" for season in SEASONS for div in LEAGUE_NAMES]
missing_files = [f.name for f in EXPECTED_FILES if not f.is_file() or f.stat().st_size == 0]
if missing_files:
    raise FileNotFoundError(
        "Incomplete raw sample; see DATA_SOURCES.md for permitted acquisition. Missing: "
        + ", ".join(missing_files)
    )

# Operational period boundaries (policy windows), not match-level attendance.
# The broad ghost period can contain matches with partial or unrestricted crowds.
# Post begins 2021-07-01 although some later capacity restrictions remained.
GHOST_START_BY_DIV = {
    "I1": pd.Timestamp("2020-03-04"),
    "F1": pd.Timestamp("2020-03-04"),
    "SP1": pd.Timestamp("2020-03-10"),
    "D1": pd.Timestamp("2020-03-11"),
    "E0": pd.Timestamp("2020-03-13"),
}
GHOST_END = pd.Timestamp("2021-06-30")

# Historical name retained for output compatibility: this narrower policy window
# drops several partial-reopening months, but does not verify zero attendance.
# In particular, Ligue 1's late-August 2020 restart admitted partial crowds.
STRICT_WINDOWS = [
    ("2020-03-01", "2020-08-31"),
    ("2020-11-05", "2020-11-30"),
    ("2021-01-01", "2021-04-30"),
]

frames = []
for f in sorted(EXPECTED_FILES):
    div, season = f.stem.split("_")
    df = pd.read_csv(f, encoding="latin-1")
    # Newer files carry a UTF-8 BOM which corrupts the first column name
    df = df.rename(columns=lambda c: c.replace("﻿", "").replace("ï»¿", "").strip())
    missing_columns = set(CORE + STATS + ODDS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"{f.name}: missing required columns {sorted(missing_columns)}")
    df = df.dropna(how="all")
    keep = [c for c in CORE + STATS + ODDS + ["Referee"] if c in df.columns]
    df = df[keep].copy()
    df["season"] = "20" + season[:2] + "-" + season[2:]
    frames.append(df)

m = pd.concat(frames, ignore_index=True)
m["Date"] = pd.to_datetime(m["Date"], dayfirst=True, format="mixed")
m = m.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
m["league"] = m["Div"].map(LEAGUE_NAMES)

# Outcomes
m["margin"] = m["FTHG"] - m["FTAG"]
m["home_win"] = (m["FTHG"] > m["FTAG"]).astype(int)
m["away_win"] = (m["FTHG"] < m["FTAG"]).astype(int)
m["home_pts"] = np.select([m["margin"] > 0, m["margin"] == 0], [3, 1], default=0)
m["away_pts"] = np.select([m["margin"] < 0, m["margin"] == 0], [3, 1], default=0)
for h, a, name in [
    ("HY", "AY", "yellow_diff"),   # away minus home: >0 means more away sanctions
    ("HF", "AF", "foul_diff"),
    ("HR", "AR", "red_diff"),
]:
    m[name] = m[a] - m[h]
# Bet365 B365H/D/A snapshot probabilities, not the separate closing-odds fields.
# Require all three prices to avoid normalizing an incomplete market.
inv = 1 / m[ODDS]
m["implied_home_prob"] = inv["B365H"] / inv.sum(axis=1, min_count=3)
m["points_diff"] = m["home_pts"] - m["away_pts"]
m["shots_diff"] = m["HS"] - m["AS"]          # home minus away: >0 home dominance
m["sot_diff"] = m["HST"] - m["AST"]
m["corner_diff"] = m["HC"] - m["AC"]

# Crowd periods
ghost_start = m["Div"].map(GHOST_START_BY_DIV)
m["period"] = np.select(
    [m["Date"] < ghost_start, m["Date"] <= GHOST_END],
    ["pre", "ghost"],
    default="post",
)
m["ghost"] = (m["period"] == "ghost").astype(int)
m["post"] = (m["period"] == "post").astype(int)
strict = pd.Series(False, index=m.index)
for s, e in STRICT_WINDOWS:
    strict |= m["Date"].between(pd.Timestamp(s), pd.Timestamp(e))
# Exclude the nine early-March French matches in this sensitivity definition.
strict &= ~((m["Div"] == "F1") & (m["Date"] < pd.Timestamp("2020-03-09")))
m["ghost_strict"] = (strict & (m["period"] == "ghost")).astype(int)
# Omicron-era re-restrictions inside the post period:
# Germany returned to ghost games / caps, France, Spain and Italy were capped
OMICRON_WINDOWS = {
    "D1": ("2021-12-28", "2022-03-20"),
    "F1": ("2022-01-03", "2022-02-02"),
    "SP1": ("2022-01-01", "2022-03-04"),
    "I1": ("2022-01-01", "2022-04-01"),
}
m["omicron_restricted"] = 0
for div, (s, e) in OMICRON_WINDOWS.items():
    mask = (m["Div"] == div) & m["Date"].between(pd.Timestamp(s), pd.Timestamp(e))
    m.loc[mask, "omicron_restricted"] = 1
# Mature post period: first fully unrestricted season onward
m["post_mature"] = (m["Date"] >= pd.Timestamp("2022-07-01")).astype(int)

# Team-season identifiers for fixed effects
m["home_ts"] = m["HomeTeam"] + "_" + m["season"] + "_" + m["Div"]
m["away_ts"] = m["AwayTeam"] + "_" + m["season"] + "_" + m["Div"]
m["league_season"] = m["Div"] + "_" + m["season"]

m = m.sort_values(["Date", "Div"]).reset_index(drop=True)
m.to_csv(OUT / "matches.csv", index=False)

# --- Summary for sanity checking ---
print(f"Matches: {len(m)}")
print(f"Date range: {m['Date'].min().date()} .. {m['Date'].max().date()}")
print("\nMatches by league:")
print(m.groupby("league").size().to_string())
print("\nMatches by period:")
print(m.groupby("period").size().to_string())
print(f"\nNarrower-window matches (ghost_strict): {m['ghost_strict'].sum()}")
print(f"Omicron-restricted post matches: {m['omicron_restricted'].sum()}")
print("\nMissing values in match stats:")
print(m[STATS].isna().sum().to_string())
print("\nMatches by league x period:")
print(m.pivot_table(index="league", columns="period", values="margin", aggfunc="size").to_string())
print("\nSeason x period counts (ghost rows):")
print(m[m["ghost"] == 1].groupby(["season", "league"]).size().to_string())
