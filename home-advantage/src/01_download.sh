#!/bin/bash
# Download Big-5 league match data from football-data.co.uk
# Leagues: E0=Premier League, D1=Bundesliga, SP1=La Liga, I1=Serie A, F1=Ligue 1
# Seasons: 2014-15 .. 2025-26 (60 league-season files)
# Optional helper: review the provider's current terms before running.
set -euo pipefail
RAW_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$RAW_DIR"

SEASONS="1415 1516 1617 1718 1819 1920 2021 2122 2223 2324 2425 2526"
LEAGUES="E0 D1 SP1 I1 F1"

for s in $SEASONS; do
  for l in $LEAGUES; do
    out="$RAW_DIR/${l}_${s}.csv"
    if [ -s "$out" ]; then
      echo "skip $out (exists)"
      continue
    fi
    url="https://www.football-data.co.uk/mmz4281/${s}/${l}.csv"
    echo "GET $url"
    temporary="${out}.part"
    if ! curl -fsSL --retry 3 --connect-timeout 15 --max-time 120 "$url" -o "$temporary"; then
      rm -f "$temporary"
      echo "FAILED: $url; stopping to avoid analyzing an incomplete sample." >&2
      exit 1
    fi
    if [ ! -s "$temporary" ]; then
      rm -f "$temporary"
      echo "EMPTY RESPONSE: $url" >&2
      exit 1
    fi
    mv "$temporary" "$out"
    sleep 0.5
  done
done

echo "All 60 league-season files are available in $RAW_DIR."
