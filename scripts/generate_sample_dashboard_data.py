"""Generate realistic sample JSON data for the NBA Analytics Dashboard preview.

Populates dashboard/data/ with synthetic but plausible prediction data
so the dashboard can be previewed without a live database.

Usage:
    python scripts/generate_sample_dashboard_data.py
"""

import json
import os
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DATA = PROJECT_ROOT / "dashboard" / "data"

# NBA teams with full names
TEAMS = [
    ("BOS", "Boston Celtics"),
    ("MIL", "Milwaukee Bucks"),
    ("PHI", "Philadelphia 76ers"),
    ("CLE", "Cleveland Cavaliers"),
    ("NYK", "New York Knicks"),
    ("BKN", "Brooklyn Nets"),
    ("MIA", "Miami Heat"),
    ("ATL", "Atlanta Hawks"),
    ("CHI", "Chicago Bulls"),
    ("TOR", "Toronto Raptors"),
    ("IND", "Indiana Pacers"),
    ("WAS", "Washington Wizards"),
    ("ORL", "Orlando Magic"),
    ("CHA", "Charlotte Hornets"),
    ("DET", "Detroit Pistons"),
    ("DEN", "Denver Nuggets"),
    ("OKC", "Oklahoma City Thunder"),
    ("MIN", "Minnesota Timberwolves"),
    ("LAL", "Los Angeles Lakers"),
    ("LAC", "Los Angeles Clippers"),
    ("SAC", "Sacramento Kings"),
    ("PHX", "Phoenix Suns"),
    ("DAL", "Dallas Mavericks"),
    ("GSW", "Golden State Warriors"),
    ("HOU", "Houston Rockets"),
    ("MEM", "Memphis Grizzlies"),
    ("NOP", "New Orleans Pelicans"),
    ("SAS", "San Antonio Spurs"),
    ("POR", "Portland Trail Blazers"),
    ("UTA", "Utah Jazz"),
]

TEAM_LOOKUP = {abbr: name for abbr, name in TEAMS}

SAMPLE_PLAYERS = [
    ("Jayson Tatum", "BOS", 27.1, 8.3, 4.8),
    ("Luka Doncic", "DAL", 33.4, 9.1, 9.5),
    ("Nikola Jokic", "DEN", 26.3, 12.4, 9.0),
    ("Giannis Antetokounmpo", "MIL", 31.2, 11.8, 5.7),
    ("Shai Gilgeous-Alexander", "OKC", 31.5, 5.5, 6.2),
    ("Joel Embiid", "PHI", 33.0, 10.2, 4.1),
    ("LeBron James", "LAL", 25.3, 7.1, 8.4),
    ("Stephen Curry", "GSW", 26.8, 4.5, 5.1),
    ("Kevin Durant", "PHX", 27.5, 6.7, 5.2),
    ("Anthony Edwards", "MIN", 26.1, 5.6, 5.3),
    ("Jaylen Brown", "BOS", 23.4, 5.5, 3.6),
    ("Donovan Mitchell", "CLE", 24.8, 4.1, 5.3),
    ("Tyrese Haliburton", "IND", 20.5, 3.9, 10.8),
    ("Trae Young", "ATL", 25.8, 2.8, 10.2),
    ("De'Aaron Fox", "SAC", 25.9, 4.5, 6.1),
    ("Jalen Brunson", "NYK", 28.2, 3.5, 6.8),
    ("Paolo Banchero", "ORL", 22.8, 6.9, 5.4),
    ("Ja Morant", "MEM", 25.1, 5.6, 8.2),
    ("Damian Lillard", "MIL", 24.6, 4.4, 7.0),
    ("Jimmy Butler", "MIA", 20.8, 5.9, 5.0),
]


def _write_json(filename: str, data) -> None:
    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)
    path = DASHBOARD_DATA / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Wrote {path}")


def _random_matchups(n: int, game_date: str) -> list:
    """Generate n random non-overlapping matchups."""
    available = list(TEAMS)
    random.shuffle(available)
    matchups = []
    for i in range(0, min(n * 2, len(available)) - 1, 2):
        home = available[i]
        away = available[i + 1]
        matchups.append((home, away, game_date))
    return matchups[:n]


def generate_todays_picks() -> None:
    today = date.today().isoformat()
    matchups = _random_matchups(7, today)
    picks = []
    for (home_abbr, home_name), (away_abbr, away_name), gd in matchups:
        home_prob = round(random.uniform(0.30, 0.78), 4)
        away_prob = round(1.0 - home_prob, 4)
        spread = round(random.uniform(-12.0, 12.0) * 2) / 2  # half-point spreads
        is_value = random.random() < 0.3
        edge = round(random.uniform(0.05, 0.15), 3) if is_value else None

        picks.append({
            "game_date": gd,
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_team_name": home_name,
            "away_team_name": away_name,
            "home_win_prob": home_prob,
            "away_win_prob": away_prob,
            "predicted_winner": home_abbr if home_prob >= 0.5 else away_abbr,
            "ats_pick": home_abbr if random.random() > 0.5 else away_abbr,
            "spread": spread,
            "value_bet": is_value,
            "edge_pct": edge,
            "model_name": "gradient_boosting_v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    _write_json("todays_picks.json", picks)


def generate_accuracy_history() -> None:
    history = []
    cumulative_correct = 0
    cumulative_total = 0
    base_date = date.today() - timedelta(days=60)

    for i in range(60):
        d = base_date + timedelta(days=i)
        if d.weekday() in (1, 4):  # skip some days
            continue
        games = random.randint(4, 14)
        # Model accuracy around 66-70% with some variance
        accuracy_rate = random.gauss(0.668, 0.08)
        accuracy_rate = max(0.35, min(0.95, accuracy_rate))
        correct = int(round(games * accuracy_rate))
        correct = max(0, min(games, correct))

        cumulative_correct += correct
        cumulative_total += games
        rolling = cumulative_correct / cumulative_total if cumulative_total > 0 else 0

        history.append({
            "date": d.isoformat(),
            "daily_accuracy": round(correct / games, 4) if games > 0 else 0,
            "rolling_accuracy": round(rolling, 4),
            "games": games,
            "correct": correct,
            "cumulative_games": cumulative_total,
        })

    _write_json("accuracy_history.json", history)


def generate_value_bets() -> None:
    bets = []
    for i in range(12):
        d = date.today() - timedelta(days=random.randint(0, 7))
        home_abbr, home_name = random.choice(TEAMS)
        away_abbr, away_name = random.choice(
            [t for t in TEAMS if t[0] != home_abbr]
        )
        model_prob = round(random.uniform(0.45, 0.75), 4)
        market_prob = round(model_prob - random.uniform(0.05, 0.14), 4)
        edge = round(model_prob - market_prob, 4)
        side = home_abbr if model_prob >= 0.5 else away_abbr

        bets.append({
            "game_date": d.isoformat(),
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_team_name": home_name,
            "away_team_name": away_name,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "edge_pct": edge,
            "recommended_side": side,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    bets.sort(key=lambda b: b["edge_pct"], reverse=True)
    _write_json("value_bets.json", bets)


def generate_player_predictions() -> None:
    players = []
    for name, team, avg_pts, avg_reb, avg_ast in SAMPLE_PLAYERS:
        opp_abbr = random.choice([t[0] for t in TEAMS if t[0] != team])
        pts = round(avg_pts + random.gauss(0, 3.0), 1)
        reb = round(avg_reb + random.gauss(0, 1.5), 1)
        ast = round(avg_ast + random.gauss(0, 1.5), 1)

        players.append({
            "player_name": name,
            "pts": max(0, pts),
            "reb": max(0, reb),
            "ast": max(0, ast),
            "team": team,
            "team_name": TEAM_LOOKUP.get(team, team),
            "opponent": opp_abbr,
            "opponent_name": TEAM_LOOKUP.get(opp_abbr, opp_abbr),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    _write_json("player_predictions.json", players)


def generate_meta() -> None:
    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "db_exists": False,
        "sample_data": True,
    }
    _write_json("meta.json", meta)


def main():
    random.seed(42)  # reproducible sample data
    print("Generating sample dashboard data...")
    generate_todays_picks()
    generate_accuracy_history()
    generate_value_bets()
    generate_player_predictions()
    generate_meta()
    print("Done. Dashboard data ready at dashboard/data/")


if __name__ == "__main__":
    main()
