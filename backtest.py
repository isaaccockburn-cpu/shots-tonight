#!/usr/bin/env python3
"""
Backtest for the Shots Tonight model.

    python backtest.py                     # replay this season from the cached games
    python backtest.py --history history   # score the daily predictions the site saved

Replay mode: for every game date, rebuild the model using ONLY games played before
that date, predict that day's games exactly as the live job would, then compare the
predictions with the shots each player actually took. Nothing from the future is
used, so the score is an honest estimate of how the live predictions would have done.

Needs the per-game cache that build_predictions.py creates (data/games). Run
build_predictions.py first to refresh it.

Differences from the live job worth knowing
  * Only players who actually played that night can be scored. The report shows how
    many projected players did not play (scratched, injured, traded).
  * The model projects even-strength and power-play shots; results count all shots
    on goal, including the ~2% taken shorthanded, so predictions run very slightly low.
  * The model's settings (blend weights, shrinkage) were chosen by judgment, not fitted
    to this data, so this replay is a fair test. If we tune them against it, later
    scores should come from the saved history instead.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import build_predictions as B
import shots_model as M

LINES = M.LINES
MIN_TEAM_GP = 8     # both teams need this many earlier games before we score them


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def load_summaries(cache_dir):
    out = []
    for path in glob.glob(os.path.join(cache_dir, "*.json")):
        with open(path) as f:
            out.append(json.load(f))
    return sorted(out, key=lambda s: (s["date"], s["id"]))


def home_away(game):
    """Summaries store the home team first (dict order survives the JSON cache)."""
    home, away = list(game["teams"])
    return away, home


def shots_of(rec):
    return rec["ev"] + rec["pp"] + rec["sh"]


# ------------------------------------------------------------------ replay
def replay(summaries):
    """Predict every game from earlier games only. Returns (rows, pool_stats)."""
    rows = []
    pool_stats = dict(projected=0, played=0, unprojected=0)
    for d in sorted({s["date"] for s in summaries}):
        prior = [s for s in summaries if s["date"] < d]
        today = [s for s in summaries if s["date"] == d]
        if not prior:
            continue
        M.LG.update(B.league_baselines(prior))
        pool = B.build_players(prior)
        teams = {}
        for g in today:
            away, home = home_away(g)
            for ab in (away, home):
                if ab not in teams:
                    teams[ab] = B.build_team(prior, ab)
            ta, th = teams[away], teams[home]
            if not (ta and th) or min(ta.gp, th.gp) < MIN_TEAM_GP:
                continue

            players = pool[away] + pool[home]
            res = M.predict_game(ta, th, players)
            spg = {(p.name, p.team): p.spg_season for p in players}
            actual = {(r["name"], r["team"]): r for r in g["players"].values()}

            projected = set(spg)
            pool_stats["projected"] += len(projected)
            pool_stats["played"] += len(projected & set(actual))
            pool_stats["unprojected"] += len(set(actual) - projected)

            for p in res["players"]:
                a = actual.get((p["player"], p["team"]))
                if a is None or p["lam"] < B.MIN_LAM:
                    continue
                naive = max(spg[(p["player"], p["team"])], 0.05)   # player's own average only
                row = dict(date=d, game=f"{away}@{home}", player=p["player"], team=p["team"],
                           lam=p["lam"], naive_lam=round(naive, 2), shots=shots_of(a))
                for k in LINES:
                    row[f"p{k}"] = p["probs"][f"{k}+"]
                    row[f"n{k}"] = round(M.prob_at_least(k, naive), 3)
                rows.append(row)
    return rows, pool_stats


# ----------------------------------------------------------- saved history
def score_history(history_dir, summaries):
    """Score the daily files saved by the job against results in the cache."""
    results = defaultdict(dict)
    for s in summaries:
        for r in s["players"].values():
            results[s["date"]][(r["name"], r["team"])] = shots_of(r)
    rows, missing = [], 0
    for path in sorted(glob.glob(os.path.join(history_dir, "*.json"))):
        d = os.path.basename(path)[:-5]
        if d not in results:
            continue
        with open(path) as f:
            day = json.load(f)
        for g in day["games"]:
            for p in g["players"]:
                shots = results[d].get((p["player"], p["team"]))
                if shots is None:
                    missing += 1
                    continue
                row = dict(date=d, game=f'{g["away"]}@{g["home"]}', player=p["player"],
                           team=p["team"], lam=p["lam"], shots=shots)
                for k in LINES:
                    row[f"p{k}"] = p["probs"][f"{k}+"]
                rows.append(row)
    return rows, missing


# ------------------------------------------------------------------ report
def brier(pairs):
    return mean((p - y) ** 2 for p, y in pairs)


def calibration(rows, k):
    buckets = defaultdict(list)
    for r in rows:
        buckets[min(int(r[f"p{k}"] * 10), 9)].append((r[f"p{k}"], int(r["shots"] >= k)))
    print(f"Calibration for {k}+ shots: when the model says X%, how often did it happen?")
    print(f'{"Model said":<14}{"Players":>9}{"Avg said":>10}{"Actual":>9}')
    for b in sorted(buckets):
        ps = buckets[b]
        label = f"{b * 10}-{b * 10 + 10}%"
        print(f"{label:<14}{len(ps):>9}{mean(p for p, _ in ps) * 100:>9.0f}%"
              f"{mean(y for _, y in ps) * 100:>8.0f}%")
    print()


def report(rows):
    if not rows:
        print("No scoreable predictions yet.")
        return
    dates = sorted({r["date"] for r in rows})
    has_naive = "n1" in rows[0]
    print(f"{len(rows)} player predictions over {len(dates)} days ({dates[0]} to {dates[-1]})")
    print(f"Average projected shots {mean(r['lam'] for r in rows):.2f}, "
          f"actual {mean(r['shots'] for r in rows):.2f}")
    spread = mean((r["shots"] - r["lam"]) ** 2 / r["lam"] for r in rows)
    print(f"Spread check: {spread:.2f} (1.00 means shots vary exactly as much as the model "
          f"assumes; higher means streakier shooting or noisy estimates)\n")

    print("Brier score by line (lower is better)")
    head = f'{"Line":<6}{"Players":>9}{"Hit rate":>10}{"Avg said":>10}{"Model":>9}'
    if has_naive:
        head += f'{"Own avg":>9}'
    head += f'{"League":>9}'
    print(head)
    for k in LINES:
        ys = [int(r["shots"] >= k) for r in rows]
        base = mean(ys)
        line = (f'{k}+'.ljust(6) + f"{len(rows):>9}{base * 100:>9.0f}%"
                f"{mean(r[f'p{k}'] for r in rows) * 100:>9.0f}%"
                f"{brier((r[f'p{k}'], y) for r, y in zip(rows, ys)):>9.4f}")
        if has_naive:
            line += f"{brier((r[f'n{k}'], y) for r, y in zip(rows, ys)):>9.4f}"
        line += f"{brier((base, y) for y in ys):>9.4f}"
        print(line)
    print('  Model = this model. Own avg = each player\'s season average alone, no team, '
          'opponent or power-play adjustments.\n  League = same chance for everyone.\n')
    for k in (2, 3, 4):
        calibration(rows, k)


def summarize(rows):
    """Compact summary of scored predictions, for the site's Track record page."""
    if not rows:
        return None
    dates = sorted({r["date"] for r in rows})
    has_naive = "n1" in rows[0]
    out = dict(n=len(rows), days=len(dates), start=dates[0], end=dates[-1],
               avg_pred=round(mean(r["lam"] for r in rows), 2),
               avg_actual=round(mean(r["shots"] for r in rows), 2),
               lines=[], calibration={})
    for k in LINES:
        ys = [int(r["shots"] >= k) for r in rows]
        base = mean(ys)
        line = dict(k=k, hit=round(base, 4), said=round(mean(r[f"p{k}"] for r in rows), 4),
                    brier=round(brier((r[f"p{k}"], y) for r, y in zip(rows, ys)), 5),
                    league=round(brier((base, y) for y in ys), 5), own=None)
        if has_naive:
            line["own"] = round(brier((r[f"n{k}"], y) for r, y in zip(rows, ys)), 5)
        out["lines"].append(line)
        buckets = defaultdict(list)
        for r, y in zip(rows, ys):
            buckets[min(int(r[f"p{k}"] * 10), 9)].append((r[f"p{k}"], y))
        out["calibration"][str(k)] = [
            dict(lo=b * 10, hi=b * 10 + 10, n=len(v), said=round(mean(p for p, _ in v), 4),
                 actual=round(mean(y for _, y in v), 4))
            for b, v in sorted(buckets.items())]
    return out


def write_track_json(path, summaries, history_dir):
    import datetime as dt
    rows, pool = replay(summaries)
    replay_sum = summarize(rows)
    if replay_sum and pool["projected"]:
        replay_sum["played_pct"] = round(pool["played"] / pool["projected"] * 100)
    live_sum, missing = None, 0
    if history_dir and os.path.isdir(history_dir):
        live_rows, missing = score_history(history_dir, summaries)
        live_sum = summarize(live_rows)
    with open(path, "w") as f:
        json.dump(dict(generated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       replay=replay_sum, live=live_sum, live_missing=missing), f)
    print(f"Wrote {path}: replay {replay_sum['n'] if replay_sum else 0} predictions, "
          f"live {live_sum['n'] if live_sum else 0}")


def write_csv(rows, path):
    if not rows or not path:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Every prediction with its result is saved in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/games")
    ap.add_argument("--history", help="folder of saved daily predictions to score")
    ap.add_argument("--out", default="backtest_rows.csv", help="CSV of every prediction (empty to skip)")
    ap.add_argument("--json", help="write the site's track record data to this file (replay of the "
                                   "season, plus saved history if --history is given), then stop")
    args = ap.parse_args()

    summaries = load_summaries(args.cache)
    if not summaries:
        raise SystemExit(f"No cached games in {args.cache}. Run build_predictions.py first.")
    print(f"{len(summaries)} completed games in the cache\n")

    if args.json:
        write_track_json(args.json, summaries, args.history)
        return

    if args.history:
        rows, missing = score_history(args.history, summaries)
        report(rows)
        print(f"{missing} saved predictions had no matching result (player scratched, or the "
              f"game isn't final in the cache yet; run build_predictions.py to refresh it).")
    else:
        rows, pool = replay(summaries)
        report(rows)
        if pool["projected"]:
            print(f"Of {pool['projected']} player slots the model projected, "
                  f"{pool['played'] / pool['projected'] * 100:.0f}% actually played that night. "
                  f"{pool['unprojected']} players played who weren't projected "
                  f"(fewer than {B.MIN_GP} games, or missed the previous game).")
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
