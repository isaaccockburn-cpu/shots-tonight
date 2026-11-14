#!/usr/bin/env python3
"""
Build predictions.json for the Shots Tonight site.

    python build_predictions.py                       # today's games (US Eastern)
    python build_predictions.py --date 2026-11-12
    python build_predictions.py --season-start 2026-10-01 --out predictions.json

How it works
  1. Lists every finished regular-season game this season (NHL schedule API).
  2. Downloads each game's play-by-play ONCE and boils it down to a small
     summary (cached in data/games/), so daily runs only fetch new games.
  3. Aggregates team shot rates by strength (even strength / power play),
     power-play minutes, and player shot shares; season and last 5 games.
  4. Feeds those into shots_model.predict_game() for tonight's games.

Standard library only. Needs network access to api-web.nhle.com.

Known simplifications
  * Time by strength is measured between consecutive play-by-play events using
    the situationCode, so it is accurate to within seconds per game.
  * Tonight's player pool = skaters who dressed in their team's most recent
    game (and have 3+ games). Scratches, injuries and trades are not known
    until lineups are confirmed, so treat early-day numbers as projections.
  * Shorthanded shots are excluded from the player model (about 2% of shots).
"""
import argparse
import datetime as dt
import json
import os
import time
import urllib.request
from collections import defaultdict

import shots_model as M

API = "https://api-web.nhle.com/v1"
MIN_GP = 3          # minimum games for a player to be projected
MIN_LAM = 0.8       # hide players projected below this many shots
SHARE_K = 3         # games' worth of league-average shot share blended into each player


# ------------------------------------------------------------------ fetching
def fetch_json(url, retries=3, pause=0.25):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "shots-tonight/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            time.sleep(pause)  # be polite to the API
            return data
        except Exception as e:  # network errors, 5xx, bad JSON
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last}")


def week_games(start_date: str):
    """(date, game) pairs for the 7-day window starting at start_date."""
    j = fetch_json(f"{API}/schedule/{start_date}")
    return [(day["date"], g) for day in j.get("gameWeek", []) for g in day.get("games", [])]


def completed_games(season_start: dt.date, before: dt.date) -> dict:
    """{game_id: date} for finished regular-season games before `before`."""
    found, d = {}, season_start
    while d < before:
        for gdate, g in week_games(d.isoformat()):
            if (g.get("gameType") == 2 and g.get("gameState") in ("OFF", "FINAL")
                    and gdate < before.isoformat()):
                found[g["id"]] = gdate
        d += dt.timedelta(days=7)
    return found


# ------------------------------------------------------------ play-by-play
def parse_clock(s: str) -> int:
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def strength(code: str, is_home: bool) -> str:
    """situationCode = awayGoalie, awaySkaters, homeSkaters, homeGoalie."""
    ag, ask, hsk, hg = (int(c) for c in code)
    own, opp = (hsk, ask) if is_home else (ask, hsk)
    own_g, opp_g = (hg, ag) if is_home else (ag, hg)
    if own > opp and own_g == 1:
        return "pp"
    if opp > own and opp_g == 1:
        return "sh"
    return "ev"


def summarize_pbp(pbp: dict) -> dict:
    """Reduce a play-by-play file to per-team and per-player shot counts."""
    home, away = pbp["homeTeam"], pbp["awayTeam"]
    ids = {home["id"]: (home["abbrev"], True), away["id"]: (away["abbrev"], False)}
    blank = lambda opp: dict(opp=opp, ev_sf=0, pp_sf=0, sh_sf=0,
                             ev_min=0.0, pp_min=0.0, sh_min=0.0)
    teams = {home["abbrev"]: blank(away["abbrev"]), away["abbrev"]: blank(home["abbrev"])}

    players = {}
    for r in pbp.get("rosterSpots", []):
        if r.get("positionCode") == "G" or r["teamId"] not in ids:
            continue
        players[r["playerId"]] = dict(
            team=ids[r["teamId"]][0], pos=r.get("positionCode"), ev=0, pp=0, sh=0,
            name=f'{r["firstName"]["default"]} {r["lastName"]["default"]}')

    last_code, prev = None, None  # prev = (period, seconds, code)
    for p in pbp.get("plays", []):
        pd = p.get("periodDescriptor", {})
        if pd.get("periodType") == "SO":          # skip shootouts
            continue
        code = p.get("situationCode") or last_code
        t, per = parse_clock(p["timeInPeriod"]), pd.get("number")

        if prev and prev[0] == per and prev[2]:   # time since previous event
            minutes = max(t - prev[1], 0) / 60
            for ab, is_home in ids.values():
                teams[ab][strength(prev[2], is_home) + "_min"] += minutes
        prev = (per, t, code)
        last_code = code or last_code

        kind = p.get("typeDescKey")
        if kind in ("shot-on-goal", "goal") and code:   # goals count as shots on goal
            d = p.get("details", {})
            if d.get("eventOwnerTeamId") not in ids:
                continue
            ab, is_home = ids[d["eventOwnerTeamId"]]
            s = strength(code, is_home)
            teams[ab][s + "_sf"] += 1
            pid = d.get("shootingPlayerId") if kind == "shot-on-goal" else d.get("scoringPlayerId")
            if pid in players:
                players[pid][s] += 1
    return dict(teams=teams, players={str(k): v for k, v in players.items()})


def get_game_summary(game_id: int, gdate: str, cache_dir: str) -> dict:
    path = os.path.join(cache_dir, f"{game_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    s = summarize_pbp(fetch_json(f"{API}/gamecenter/{game_id}/play-by-play"))
    s.update(id=game_id, date=gdate)
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(s, f)
    return s


# ------------------------------------------------------------- aggregation
def _sums(rows):
    """rows: iterable of (team_record, opponent_record)."""
    tot, n = defaultdict(float), 0
    for t, o in rows:
        n += 1
        for k in ("ev_sf", "pp_sf", "sh_sf", "ev_min", "pp_min", "sh_min"):
            tot[k] += t[k]
        tot["ev_sa"] += o["ev_sf"]
        tot["pk_sa"] += o["pp_sf"]   # shots against while we are shorthanded
    return tot, n


def _rates(tot, n, fallback):
    per60 = lambda x, mins, d: x / mins * 60 if mins > 0 else d
    return dict(
        sf_ev=per60(tot["ev_sf"], tot["ev_min"], fallback["sf_ev"]),
        sf_pp=per60(tot["pp_sf"], tot["pp_min"], fallback["sf_pp"]),
        sa_ev=per60(tot["ev_sa"], tot["ev_min"], fallback["sa_ev"]),
        sa_pk=per60(tot["pk_sa"], tot["sh_min"], fallback["sa_pk"]),
        pp_min=tot["pp_min"] / n if n else fallback["pp_min"],
        sh_min=tot["sh_min"] / n if n else fallback["sh_min"],
        sf_pg=(tot["ev_sf"] + tot["pp_sf"] + tot["sh_sf"]) / n if n else fallback["sf_pg"],
    )


def team_rates(games, team):
    rows = [(g["teams"][team], g["teams"][g["teams"][team]["opp"]]) for g in games]
    tot, n = _sums(rows)
    return _rates(tot, n, M.LG)


def league_baselines(summaries):
    rows = [(t, g["teams"][t["opp"]]) for g in summaries for t in g["teams"].values()]
    tot, n = _sums(rows)
    return _rates(tot, n, M.LG)


def build_team(summaries, team):
    games = [g for g in summaries if team in g["teams"]]          # oldest first
    if not games:
        return None
    return M.TeamStats(team, len(games), team_rates(games, team), team_rates(games[-5:], team))


def build_players(summaries):
    """{team: [PlayerStats]} for skaters who dressed in their team's latest game."""
    log = defaultdict(list)                       # pid -> [(date, player_rec, team_rec)]
    last_game = {}                                # team -> latest summary
    for g in summaries:
        for pid, r in g["players"].items():
            log[pid].append((r, g["teams"][r["team"]]))
        for ab in g["teams"]:
            last_game[ab] = g

    def shares(rows):
        ev = sum(r["ev"] for r, _ in rows)
        pp = sum(r["pp"] for r, _ in rows)
        t_ev = sum(t["ev_sf"] for _, t in rows)
        t_pp = sum(t["pp_sf"] for _, t in rows)
        spg = sum(r["ev"] + r["pp"] + r["sh"] for r, _ in rows) / len(rows)
        return (ev / t_ev if t_ev else 0.0, pp / t_pp if t_pp else 0.0, spg)

    out = defaultdict(list)
    posgroup = {}
    for ab, g in last_game.items():
        for pid, r in g["players"].items():
            if r["team"] != ab or len(log[pid]) < MIN_GP:
                continue
            rows = log[pid]
            ev, pp, spg = shares(rows)
            ev5, pp5, spg5 = shares(rows[-5:])
            out[ab].append(M.PlayerStats(r["name"], ab, len(rows), ev, pp, ev5, pp5, spg, spg5))
            posgroup[(r["name"], ab)] = "D" if r.get("pos") == "D" else "F"

    # Shrink each player's shot shares toward the average for his position group
    # (forward or defenseman). Small samples otherwise swing projections too far:
    # a player with 3 games and no power-play shots would be projected at zero.
    # Backtest on 25 days: better on 23 of them; SHARE_K=3 was the best of 2-7.
    groups = defaultdict(list)
    for ps in out.values():
        for p in ps:
            groups[posgroup[(p.name, p.team)]].append(p)
    for g, ps in groups.items():
        ev0 = sum(p.ev_share for p in ps) / len(ps)
        pp0 = sum(p.pp_share for p in ps) / len(ps)
        for p in ps:
            n = p.gp
            blend = lambda x, prior: (n * x + SHARE_K * prior) / (n + SHARE_K)
            p.ev_share, p.ev_share_l5 = blend(p.ev_share, ev0), blend(p.ev_share_l5, ev0)
            p.pp_share, p.pp_share_l5 = blend(p.pp_share, pp0), blend(p.pp_share_l5, pp0)
    return out


# --------------------------------------------------------------------- main
def today_eastern() -> dt.date:
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return dt.date.today()


def build(summaries, schedule_games, target: dt.date) -> dict:
    """Pure function: summaries + tonight's schedule -> predictions dict."""
    M.LG.update(league_baselines(summaries))
    teams = {}
    pool = build_players(summaries)
    games_out = []
    for g in schedule_games:
        a, h = g["awayTeam"]["abbrev"], g["homeTeam"]["abbrev"]
        for ab in (a, h):
            if ab not in teams:
                teams[ab] = build_team(summaries, ab)
        if not (teams[a] and teams[h]):
            print(f"Skipping {a} at {h}: no completed games yet")
            continue
        res = M.predict_game(teams[a], teams[h], pool[a] + pool[h])
        res["players"] = [p for p in res["players"] if p["lam"] >= MIN_LAM]
        games_out.append(res)
    return dict(date=f"{target:%b} {target.day}, {target.year}",
                generated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                games=games_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: today, US Eastern)")
    ap.add_argument("--season-start", default="2026-10-01")
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--cache", default="data/games")
    ap.add_argument("--history", default="history",
                    help="folder for a dated copy of each day's predictions (empty to skip)")
    args = ap.parse_args()

    target = dt.date.fromisoformat(args.date) if args.date else today_eastern()
    tonight = [g for d, g in week_games(target.isoformat())
               if d == target.isoformat() and g.get("gameType") == 2]
    print(f"{len(tonight)} games on {target}")

    done = completed_games(dt.date.fromisoformat(args.season_start), target)
    print(f"{len(done)} completed games ({args.cache} caches them)")
    summaries = sorted((get_game_summary(gid, d, args.cache) for gid, d in done.items()),
                       key=lambda s: (s["date"], s["id"]))

    result = build(summaries, tonight, target)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    if args.history:   # dated copy, used later to score the predictions against results
        os.makedirs(args.history, exist_ok=True)
        with open(os.path.join(args.history, f"{target.isoformat()}.json"), "w") as f:
            json.dump(result, f, indent=1)
    lg = M.LG
    print(f"League baselines: even-strength {lg['sf_ev']:.1f} shots/60, power play "
          f"{lg['sf_pp']:.1f}/60, PP minutes {lg['pp_min']:.1f}/game, "
          f"{lg['sf_pg']:.1f} shots/game")
    print(f"Wrote {args.out}: {len(result['games'])} games")


if __name__ == "__main__":
    main()
