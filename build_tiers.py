#!/usr/bin/env python3
"""
Build calibration.json: for each line (1+ to 5+ shots) and each range of model
percentages ("tier"), how often players in that range really reached the line.

    python build_tiers.py data/season_2025 data/games --out calibration.json

Give it one or more game-cache folders (each is replayed as its own season, using
only earlier games, exactly like backtest.py), and it pools all the predictions.
The site reads calibration.json to show each player's "tier hit rate".

Tiers are made from 5-point slices of the model's percentage, merged with their
neighbours until each tier holds at least --min-n predictions, so thin areas (the
very top of the range) become one wider tier instead of a noisy one.

Rebuild it when the model changes. Standard library only.
"""
import argparse
import datetime as dt
import json

import backtest as BT

LINES = BT.LINES
SLICE = 5           # width of the building-block slices, in percentage points


def tiers_for_line(rows, k, min_n):
    n = [0] * (100 // SLICE)
    hit = [0] * (100 // SLICE)
    said = [0.0] * (100 // SLICE)
    for r in rows:
        i = min(int(r[f"p{k}"] * 100 // SLICE), len(n) - 1)
        n[i] += 1
        hit[i] += int(r["shots"] >= k)
        said[i] += r[f"p{k}"]
    bins, start, an, ah, asd = [], 0, 0, 0, 0.0
    for i in range(len(n)):
        an, ah, asd = an + n[i], ah + hit[i], asd + said[i]
        if an >= min_n:
            bins.append(dict(lo=start * SLICE, hi=(i + 1) * SLICE, n=an, hits=ah, said=asd / an))
            start, an, ah, asd = i + 1, 0, 0, 0.0
    if an:                                    # leftover at the top joins the last tier
        if bins:
            b = bins[-1]
            tot = b["n"] + an
            b.update(hi=100, said=(b["said"] * b["n"] + asd) / tot, n=tot, hits=b["hits"] + ah)
        else:
            bins.append(dict(lo=0, hi=100, n=an, hits=ah, said=asd / an))
    bins[-1]["hi"] = 100
    # A higher model percentage should never have a lower hit rate; where noise flips two
    # neighbouring tiers, merge them (pool-adjacent-violators).
    merged = []
    for b in bins:
        merged.append(dict(b))
        while len(merged) > 1 and merged[-2]["hits"] / merged[-2]["n"] > merged[-1]["hits"] / merged[-1]["n"]:
            b2, b1 = merged.pop(), merged[-1]
            tot = b1["n"] + b2["n"]
            b1.update(hi=b2["hi"], said=(b1["said"] * b1["n"] + b2["said"] * b2["n"]) / tot,
                      n=tot, hits=b1["hits"] + b2["hits"])
    bins = merged
    return [dict(lo=b["lo"], hi=b["hi"], n=b["n"], said=round(b["said"], 4),
                 actual=round(b["hits"] / b["n"], 4)) for b in bins]


def build_tiers(rows, min_n=500):
    return {str(k): tiers_for_line(rows, k, min_n) for k in LINES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+", help="game-cache folders, one per season")
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--min-n", type=int, default=500, help="minimum predictions per tier")
    args = ap.parse_args()

    rows, seasons = [], []
    for c in args.caches:
        summaries = BT.load_summaries(c)
        if not summaries:
            print(f"Skipping {c}: no games found")
            continue
        r, _ = BT.replay(summaries)
        print(f"{c}: {len(summaries)} games, {len(r)} predictions "
              f"({summaries[0]['date']} to {summaries[-1]['date']})")
        rows += r
        seasons.append(dict(start=summaries[0]["date"], end=summaries[-1]["date"], n=len(r)))
    if not rows:
        raise SystemExit("No predictions to build tiers from.")

    out = dict(generated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               n=len(rows), seasons=seasons, tiers=build_tiers(rows, args.min_n))
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {args.out}: {len(rows)} predictions pooled")
    for k in (2, 3):
        print(f" {k}+ shots tiers:")
        for b in out["tiers"][str(k)]:
            print(f"   model {b['lo']:>3}-{b['hi']:<3}%  players {b['n']:>6}  happened {b['actual'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
