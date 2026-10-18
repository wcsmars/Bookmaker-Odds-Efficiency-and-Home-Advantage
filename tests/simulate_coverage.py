"""Monte Carlo coverage of the home-advantage study's analytic 95% intervals.

    python tests/simulate_coverage.py [--reps 300] [--seed 7]

Needs the research dependencies. The synthetic schedule from
tests/make_synthetic_inputs.py is cleaned by home-advantage/src/02_clean.py in a
temporary copy, which keeps the real design: 60 league-seasons, team-season
strengths and the policy-window period indicators. Outcomes are then simulated
with every period effect equal to zero,

    y = 0.36 + h_home + s_home - s_away + e,   s ~ N(0, 0.5),   e ~ N(0, 1.75),

where h is zero ("independent errors") or a club-specific home advantage
h ~ N(0, 0.25) per team-season ("club home advantage"). Each replication is
estimated with `fe_reg` from home-advantage/src/03_analysis.py, and the table
reports how often the 95% interval contains zero:

* pooled recovery specification with league-season clusters, t(59);
* within-2019-20 and single-league specifications with league-matchday clusters.

The functions are taken from 03_analysis.py without running the analysis.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POST = ["post_2122", "post_2223", "post_2324", "post_2425", "post_2526"]


def load_estimator():
    """Compile strength_block, _fwl and fe_reg from 03_analysis.py on their own."""
    source = (ROOT / "home-advantage/src/03_analysis.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    keep = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in {"strength_block", "_fwl", "fe_reg"}]
    namespace = {}
    exec("import numpy as np\nimport pandas as pd\nimport scipy.sparse as sp\n"
         "from scipy.sparse.linalg import lsqr\nimport statsmodels.api as sm\n", namespace)
    exec(compile(ast.Module(body=keep, type_ignores=[]), "03_analysis.py", "exec"), namespace)
    return namespace["fe_reg"]


def synthetic_matches(seed: int) -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("make_synthetic_inputs",
                                                  ROOT / "tests/make_synthetic_inputs.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    tmp = Path(tempfile.mkdtemp(prefix="coverage "))
    try:
        project = tmp / "home-advantage"
        (project / "src").mkdir(parents=True)
        shutil.copy(ROOT / "home-advantage/src/02_clean.py", project / "src")
        generator.write_inputs(project / "data/raw", seed)
        subprocess.run([sys.executable, str(project / "src/02_clean.py")], check=True,
                       stdout=subprocess.DEVNULL)
        m = pd.read_csv(project / "data/processed/matches.csv", parse_dates=["Date"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    m["matchday"] = m["Div"] + "_" + m["Date"].dt.strftime("%Y-%m-%d")
    for s in POST:
        m[s] = (m["season"] == f"20{s[5:7]}-{s[7:]}").astype(int)
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    fe_reg = load_estimator()
    m = synthetic_matches(args.seed)
    rng = np.random.default_rng(args.seed)
    team_seasons = pd.Index(sorted(set(m["home_ts"]) | set(m["away_ts"])))
    specs = [("pooled recovery, league-season clusters", m, ["ghost", *POST], "league_season")]
    specs.append(("within 2019-20, matchday clusters", m[m["season"] == "2019-20"].copy(),
                  ["ghost"], "matchday"))
    for league, sub in m.groupby("league"):
        specs.append((f"{league}, matchday clusters", sub.copy(), ["ghost", "post"], "matchday"))

    rows = []
    for errors, home_sd in (("independent errors", 0.0), ("club home advantage", 0.25)):
        covered = {label: [] for label, *_ in specs}
        for _ in range(args.reps):
            s = pd.Series(rng.normal(0, 0.5, len(team_seasons)), index=team_seasons)
            h = pd.Series(rng.normal(0, home_sd, len(team_seasons)), index=team_seasons)
            e = pd.Series(rng.normal(0, 1.75, len(m)), index=m.index)
            y = 0.36 + h[m["home_ts"]].to_numpy() + s[m["home_ts"]].to_numpy() \
                - s[m["away_ts"]].to_numpy() + e.to_numpy()
            m["ysim"] = y
            for label, sub, dvars, cluster in specs:
                sub["ysim"] = m.loc[sub.index, "ysim"]
                coefs = fe_reg(sub, "ysim", dvars, cluster=cluster)["coefs"]
                covered[label].append([coefs[v]["ci_lo"] <= 0 <= coefs[v]["ci_hi"] for v in dvars])
        for label, sub, dvars, cluster in specs:
            rate = np.mean(covered[label], axis=0)
            rows.append({"errors": errors, "specification": label,
                         **{v: round(float(r), 3) for v, r in zip(dvars, rate)}})
    table = pd.DataFrame(rows)
    print(f"95% interval coverage of a zero effect, {args.reps} replications, seed {args.seed}")
    print(table.to_string(index=False, na_rep=""))


if __name__ == "__main__":
    main()
