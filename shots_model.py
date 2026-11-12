"""
NHL Shots-on-Goal likelihood model.

Pipeline:  team shot environment -> player share -> Poisson probabilities.
Includes: opponent shot suppression, expected power-play minutes,
rolling 5-game trends (blended with season), and home/away.

Data layer is separate: build TeamStats / PlayerStats from the NHL API
(api-web.nhle.com) and pass them to predict_game(). Demo at the bottom
uses SAMPLE numbers only.
"""
import json
import math
from dataclasses import dataclass

# League baselines (per 60 minutes / per game). Recompute from data daily.
LG = dict(sf_ev=29.0, sf_pp=50.0, sa_ev=29.0, sa_pk=50.0,
          pp_min=5.4, sh_min=5.4, sf_pg=29.0)
# Shrinkage strength: higher = trust the league average longer.
K = dict(sf_ev=10, sf_pp=25, sa_ev=10, sa_pk=25,
         pp_min=20, sh_min=20, sf_pg=10)
W_LAST5 = 0.30      # weight on last-5 vs season
HOME_BOOST = 1.02   # home teams shoot slightly more
TREND_BAND = 0.10   # +/-10% vs season counts as up/down
LINES = (1, 2, 3, 4, 5)


@dataclass
class TeamStats:
    name: str
    gp: int
    season: dict   # keys of LG: sf_ev, sf_pp, sa_ev, sa_pk, pp_min, sh_min, sf_pg
    last5: dict    # same keys, last 5 games
    # sf = shots for per 60, sa_pk = shots allowed per 60 while shorthanded,
    # pp_min = PP minutes/game earned, sh_min = minutes/game shorthanded


@dataclass
class PlayerStats:
    name: str
    team: str
    gp: int
    ev_share: float        # share of team's even-strength shots (season)
    pp_share: float        # share of team's power-play shots (season)
    ev_share_l5: float
    pp_share_l5: float
    spg_season: float      # shots per game, for the trend arrow
    spg_last5: float


def _shrink(value, n, prior, k):
    return (n * value + k * prior) / (n + k)


def blended(team: TeamStats) -> dict:
    """Season rate shrunk to league, then 70/30 blended with (shrunk) last 5."""
    out = {}
    for key, lg in LG.items():
        s = _shrink(team.season[key], team.gp, lg, K[key])
        l5 = _shrink(team.last5[key], min(5, team.gp), s, K[key])
        out[key] = (1 - W_LAST5) * s + W_LAST5 * l5
    return out


def trend(season: float, last5: float) -> str:
    if season <= 0:
        return "flat"
    r = last5 / season
    return "up" if r > 1 + TREND_BAND else "down" if r < 1 - TREND_BAND else "flat"


def project_team(team: TeamStats, opp: TeamStats, home: bool) -> dict:
    """Expected shots for `team` tonight, split into even-strength and PP."""
    t, o = blended(team), blended(opp)
    # Power-play minutes: team's tendency to draw x opponent's tendency to take
    pp_min = t["pp_min"] * o["sh_min"] / LG["pp_min"]
    sh_min = o["pp_min"] * t["sh_min"] / LG["sh_min"]
    ev_min = max(60 - pp_min - sh_min, 40)
    # Shot rates, adjusted for opponent suppression
    ev_rate = t["sf_ev"] * o["sa_ev"] / LG["sa_ev"]
    pp_rate = t["sf_pp"] * o["sa_pk"] / LG["sa_pk"]
    f = HOME_BOOST if home else 1 / HOME_BOOST
    ev = ev_rate * ev_min / 60 * f
    pp = pp_rate * pp_min / 60 * f
    return dict(ev=ev, pp=pp, total=ev + pp, pp_min=pp_min,
                trend=trend(team.season["sf_pg"], team.last5["sf_pg"]))


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def prob_at_least(k: int, lam: float) -> float:
    return 1 - sum(poisson_pmf(i, lam) for i in range(k))


def predict_player(p: PlayerStats, team_proj: dict) -> dict:
    ev_s = (1 - W_LAST5) * p.ev_share + W_LAST5 * p.ev_share_l5
    pp_s = (1 - W_LAST5) * p.pp_share + W_LAST5 * p.pp_share_l5
    lam = ev_s * team_proj["ev"] + pp_s * team_proj["pp"]
    return dict(
        player=p.name, team=p.team, lam=round(lam, 2),
        probs={f"{k}+": round(prob_at_least(k, lam), 3) for k in LINES},
        trend=trend(p.spg_season, p.spg_last5),
    )


def predict_game(away: TeamStats, home: TeamStats, players: list) -> dict:
    proj = {away.name: project_team(away, home, False),
            home.name: project_team(home, away, True)}
    rows = []
    for p in players:
        if p.team in proj:
            opp = home.name if p.team == away.name else away.name
            r = predict_player(p, proj[p.team])
            r["opp"] = opp
            r["home"] = p.team == home.name
            rows.append(r)
    return dict(away=away.name, home=home.name,
                team_proj={k: {m: round(v, 1) if isinstance(v, float) else v
                               for m, v in d.items()} for k, d in proj.items()},
                players=sorted(rows, key=lambda r: -r["lam"]))


# ---------------- DEMO (SAMPLE DATA ONLY) ----------------
if __name__ == "__main__":
    def mk(sf_ev, sf_pp, sa_ev, sa_pk, ppm, shm, sf_pg):
        return dict(sf_ev=sf_ev, sf_pp=sf_pp, sa_ev=sa_ev, sa_pk=sa_pk,
                    pp_min=ppm, sh_min=shm, sf_pg=sf_pg)

    away = TeamStats("AAA", 18, mk(31, 54, 27, 46, 5.0, 6.5, 31.5),
                     mk(33, 56, 27, 45, 5.2, 6.0, 34.0))
    home = TeamStats("BBB", 18, mk(27, 47, 32, 55, 6.2, 4.8, 27.0),
                     mk(26, 46, 33, 56, 6.0, 4.5, 25.0))
    players = [
        PlayerStats("Sample Sniper", "AAA", 18, .13, .22, .15, .24, 4.1, 4.8),
        PlayerStats("Sample Defenseman", "AAA", 18, .08, .15, .07, .14, 2.6, 2.2),
        PlayerStats("Sample Grinder", "BBB", 18, .06, .02, .06, .02, 1.7, 1.7),
        PlayerStats("Sample PP1 Winger", "BBB", 18, .09, .20, .08, .18, 2.9, 2.5),
    ]
    out = predict_game(away, home, players)
    print(json.dumps(out["team_proj"], indent=2))
    for r in out["players"]:
        print(f'{r["player"]:<20} {r["team"]} lam={r["lam"]:<5} '
              f'{r["probs"]}  trend={r["trend"]}')
    with open("predictions.json", "w") as f:
        json.dump(out, f, indent=2)
