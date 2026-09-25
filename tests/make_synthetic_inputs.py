"""Write 60 synthetic Football-Data-style league-season CSVs for pipeline tests.

    python tests/make_synthetic_inputs.py OUT_DIR [--seed N] [--truth TRUTH_CSV]

The files mimic the provider's layout well enough to exercise every code path
of both studies: the column set of each era (BbAv* market averages before
2019-20, Avg*/B365C* closing prices from 2019-20, a Time column, a UTF-8 BOM
in the newest files), two-digit and four-digit years, trailing blank rows, a
few rows with missing odds or statistics, promotion and relegation, Ligue 1
shrinking to 18 clubs in 2023-24, the 2019-20 stoppage (Ligue 1 abandoned
after 28 rounds, the other leagues resuming behind closed doors) and the late
start of 2020-21. Match outcomes follow a known data-generating process:

* goals are Poisson with log-rate base + attack - defence, plus a home boost of
  0.20 before the pandemic, 0.05 while stadiums were closed, 0.15 afterwards
  (0.08 in 2024-25);
* away yellow cards exceed home cards by 0.26 on average, except 0.02 while
  stadiums were closed;
* Bet365 prices are Shin-margined (z = 0.025) quotes on lightly perturbed true
  probabilities, so the market is a strong benchmark, as in the real data.

`--truth` also writes each match's true outcome probabilities and expected goal
margins for parameter-recovery checks. Nothing here resembles real fixtures,
clubs or results, and the files are not a substitute for the provider's data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

LEAGUES = {"E0": 20, "D1": 18, "SP1": 20, "I1": 20, "F1": 20}
SEASONS = [f"{y:02d}{y + 1:02d}" for y in range(14, 26)]
GHOST_START = {"I1": "2020-03-04", "F1": "2020-03-04", "SP1": "2020-03-10",
               "D1": "2020-03-11", "E0": "2020-03-13"}
RESTART = {"E0": "2020-06-17", "D1": "2020-05-16", "SP1": "2020-06-11", "I1": "2020-06-20"}
GHOST_END = pd.Timestamp("2021-06-30")
HOME_BOOST = {"pre": 0.20, "ghost": 0.05, "post": 0.15}
YELLOW_SHIFT = {"pre": 0.26, "ghost": 0.02, "post": 0.26}
SHIN_Z = 0.025


def round_robin(teams, rng):
    """Double round robin by the circle method: a list of rounds of (home, away)."""
    order = list(teams)
    rng.shuffle(order)
    n = len(order)
    rounds = []
    for r in range(n - 1):
        pairs = []
        for i in range(n // 2):
            a, b = order[i], order[n - 1 - i]
            pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
        rounds.append(pairs)
        order = [order[0], order[-1]] + order[1:-1]
    return rounds + [[(b, a) for a, b in rnd] for rnd in rounds]


def round_dates(div, season, n_rounds):
    """One weekend date per round, honouring the 2019-20 stoppage and later breaks."""
    start = {"2021": "2020-09-12", "2223": "2022-08-06"}.get(season, f"20{season[:2]}-08-09")
    dates, day = [], pd.Timestamp(start)
    for _ in range(n_rounds):
        dates.append(day)
        day += pd.Timedelta(days=7)
        if season == "2223" and day == pd.Timestamp("2022-11-12"):
            day = pd.Timestamp("2022-12-24")  # World Cup break
    if season != "1920":
        return dates
    if div == "F1":
        # Weekly rounds with a winter break; round 28 on 2020-03-07, then abandoned.
        dates = [pd.Timestamp("2019-08-10") + pd.Timedelta(days=7 * r) if r < 20
                 else pd.Timestamp("2020-01-18") + pd.Timedelta(days=7 * (r - 20))
                 for r in range(n_rounds)]
        return [d if d <= pd.Timestamp("2020-03-07") else None for d in dates]
    # Rounds after the stoppage are played every four days from the restart date.
    cutoff = pd.Timestamp(GHOST_START[div]) - pd.Timedelta(days=5)
    restart = pd.Timestamp(RESTART[div])
    resumed = [d for d in dates if d > cutoff]
    return [d for d in dates if d <= cutoff] + [
        restart + pd.Timedelta(days=4 * k) for k in range(len(resumed))]


def period_of(div, date):
    if date < pd.Timestamp(GHOST_START[div]):
        return "pre"
    return "ghost" if date <= GHOST_END else "post"


def outcome_probs(mu_h, mu_a, gmax=12):
    g = np.arange(gmax + 1)
    grid = np.outer(poisson.pmf(g, mu_h), poisson.pmf(g, mu_a))
    grid /= grid.sum()
    return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])


def shin_prices(p, z):
    """Decimal odds implied by Shin's model with insider share z."""
    root = np.sqrt((1 - z) * p ** 2 + z * p)
    return 1.0 / (root.sum() * root)


def write_inputs(out: Path, seed: int = 7, truth_path: Path | None = None) -> int:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    truth = []
    for div, n_teams in LEAGUES.items():
        pool = [f"{div} Club {i:02d}" for i in range(40)]
        att = {t: rng.normal(0, 0.25) for t in pool}
        dfn = {t: rng.normal(0, 0.20) for t in pool}
        current, reserve = pool[:n_teams], pool[n_teams:]
        for season in SEASONS:
            size = 18 if (div == "F1" and season >= "2324") else n_teams
            if len(current) > size:
                current = sorted(current, key=lambda t: att[t] - dfn[t], reverse=True)[:size]
            for t in pool:  # slow drift in club strength
                att[t] = 0.85 * att[t] + rng.normal(0, 0.08)
                dfn[t] = 0.85 * dfn[t] + rng.normal(0, 0.07)
            rounds = round_robin(current, rng)
            rows = []
            for i_round, (rnd, weekend) in enumerate(zip(rounds, round_dates(div, season, len(rounds)))):
                if weekend is None:
                    continue
                for i, (h, a) in enumerate(rnd):
                    if div == "F1" and season == "1920" and i_round == 27 and i >= 9:
                        continue  # the abandoned round was only half played
                    date = weekend + pd.Timedelta(days=i % 2)  # Saturday / Sunday split
                    per = period_of(div, date)
                    boost = 0.08 if (per == "post" and season == "2425") else HOME_BOOST[per]
                    base = np.log(1.30)
                    lam_h = np.exp(base + boost + att[h] - dfn[a])
                    lam_a = np.exp(base + att[a] - dfn[h])
                    lam_h_ref = np.exp(base + HOME_BOOST["pre"] + att[h] - dfn[a])
                    fthg, ftag = rng.poisson(lam_h), rng.poisson(lam_a)
                    p_true = outcome_probs(lam_h, lam_a)
                    noisy = p_true * np.exp(rng.normal(0, 0.04, 3))
                    p_mkt = noisy / noisy.sum()
                    odds = np.round(shin_prices(p_mkt, SHIN_Z), 2)
                    closing = p_true * np.exp(rng.normal(0, 0.025, 3))
                    codds = np.round(shin_prices(closing / closing.sum(), SHIN_Z * 0.9), 2)
                    avg = np.round(shin_prices(p_mkt, SHIN_Z * 1.1), 2)
                    hs, as_ = rng.poisson(11 + 6 * lam_h), rng.poisson(9 + 6 * lam_a)
                    hst, ast = max(rng.binomial(hs, 0.33), fthg), max(rng.binomial(as_, 0.33), ftag)
                    hs, as_ = max(hs, hst), max(as_, ast)
                    hy, ay = rng.poisson(1.85), rng.poisson(1.85 + YELLOW_SHIFT[per])
                    hf, af = rng.poisson(11.5), rng.poisson(11.4 if per == "ghost" else 11.9)
                    hthg, htag = rng.binomial(fthg, 0.45), rng.binomial(ftag, 0.45)
                    row = {
                        "Div": div,
                        "Date": date.strftime("%d/%m/%Y" if season >= "1718" else "%d/%m/%y"),
                        "HomeTeam": h, "AwayTeam": a, "FTHG": fthg, "FTAG": ftag,
                        "FTR": "H" if fthg > ftag else ("D" if fthg == ftag else "A"),
                        "HTHG": hthg, "HTAG": htag,
                        "HTR": "H" if hthg > htag else ("D" if hthg == htag else "A"),
                        "HS": hs, "AS": as_, "HST": hst, "AST": ast, "HF": hf, "AF": af,
                        "HC": rng.poisson(5.6), "AC": rng.poisson(4.5), "HY": hy, "AY": ay,
                        "HR": rng.binomial(1, 0.05), "AR": rng.binomial(1, 0.07),
                        "B365H": odds[0], "B365D": odds[1], "B365A": odds[2],
                    }
                    if div == "E0":
                        row["Referee"] = f"Ref {rng.integers(0, 25)}"
                    if season >= "1920":
                        row["Time"] = "15:00" if i % 2 == 0 else "17:30"
                        row["AvgH"], row["AvgD"], row["AvgA"] = avg
                        row["B365CH"], row["B365CD"], row["B365CA"] = codds
                    else:
                        row["BbAvH"], row["BbAvD"], row["BbAvA"] = avg
                    rows.append(row)
                    truth.append({
                        "div": div, "season": season, "date": date.strftime("%Y-%m-%d"),
                        "home": h, "away": a, "period": per,
                        "pH": p_true[0], "pD": p_true[1], "pA": p_true[2],
                        "pH_ref": outcome_probs(lam_h_ref, lam_a)[0],
                        "exp_margin": lam_h - lam_a, "exp_margin_ref": lam_h_ref - lam_a,
                        "mkt_pH": p_mkt[0], "mkt_pD": p_mkt[1], "mkt_pA": p_mkt[2],
                    })
            df = pd.DataFrame(rows)
            # Defects that the real files contain
            if season == "2021" and div in ("SP1", "I1"):
                df.loc[rng.choice(len(df), 2, replace=False), ["B365H", "B365D", "B365A"]] = np.nan
            if season == "2122" and div == "F1":
                df.loc[5, ["B365H", "B365D", "B365A"]] = np.nan
            if season == "1617" and div == "D1":
                df.loc[3, "B365A"] = np.nan
            if season == "1718" and div == "SP1":
                df.loc[10, ["HY", "AY", "HF", "AF", "HS", "AS", "HC", "AC", "HR", "AR"]] = np.nan
            if season == "1516" and div == "I1":
                df.loc[20, ["HY", "AY"]] = np.nan
            text = df.to_csv(index=False)
            if season in ("1415", "1516"):
                text += ",,,,,,,,\n,,,,,,,,\n"  # trailing blank rows
            data = text.encode("latin-1")
            if season >= "2324":
                data = b"\xef\xbb\xbf" + data  # UTF-8 byte-order mark
            (out / f"{div}_{season}.csv").write_bytes(data)
            # Promotion and relegation: the three weakest go down, three come up
            down = sorted(current, key=lambda t: att[t] - dfn[t])[:3]
            up = [reserve.pop(0) for _ in range(3)]
            reserve.extend(down)
            current = [t for t in current if t not in down] + up
    if truth_path is not None:
        pd.DataFrame(truth).to_csv(truth_path, index=False)
    return len(truth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--truth", type=Path, default=None)
    args = parser.parse_args()
    n = write_inputs(args.out, args.seed, args.truth)
    print(f"Wrote 60 synthetic league-season files with {n} matches to {args.out}")


if __name__ == "__main__":
    main()
