import pandas as pd
import numpy as np
import pickle
import json
import random
from elo import compute_elo_ratings
from features import get_recent_form, get_h2h


def load_model():
    with open("models/xgb_model.pkl", "rb") as f:
        return pickle.load(f)


def get_match_probs(home, away, model, all_results, elo_ratings, today):
    home_elo = elo_ratings.get(home, 1500)
    away_elo = elo_ratings.get(away, 1500)
    home_form = get_recent_form(home, today, all_results)
    away_form = get_recent_form(away, today, all_results)
    h2h = get_h2h(home, away, today, all_results)

    X = pd.DataFrame([{
        "elo_diff": home_elo - away_elo, "home_elo": home_elo, "away_elo": away_elo,
        "home_form": home_form, "away_form": away_form, "h2h": h2h, "neutral": 1
    }])

    proba = model.predict_proba(X)[0]
    # redistribute draw probability proportionally between home/away (no draws in knockout)
    home_p = proba[0] + proba[1] * (proba[0] / (proba[0] + proba[2]))
    away_p = proba[2] + proba[1] * (proba[2] / (proba[0] + proba[2]))
    return home_p, away_p


def simulate_one_run(r32_matches, model, all_results, elo_ratings, today, prob_cache):
    """Simulate one full bracket run probabilistically, return the champion."""
    def get_probs(home, away):
        key = (home, away)
        if key not in prob_cache:
            prob_cache[key] = get_match_probs(home, away, model, all_results, elo_ratings, today)
        return prob_cache[key]

    def play(home, away):
        if home is None: return away
        if away is None: return home
        home_p, _ = get_probs(home, away)
        return home if random.random() < home_p else away

    round_winners = []
    for m in r32_matches:
        if m["decided"]:
            round_winners.append(m["winner"])
        else:
            round_winners.append(play(m["home"], m["away"]))

    while len(round_winners) > 1:
        next_round = []
        for i in range(0, len(round_winners), 2):
            winner = play(round_winners[i], round_winners[i+1])
            next_round.append(winner)
        round_winners = next_round

    return round_winners[0]


def run_simulation(n_simulations=1000):
    all_matches = pd.read_csv("data/wc2026_all.csv")
    r32_df = all_matches[all_matches["stage"] == "LAST_32"].reset_index(drop=True)

    results = pd.read_csv("data/results.csv")
    results["date"] = pd.to_datetime(results["date"])

    completed = pd.read_csv("data/wc2026_completed.csv")
    completed["date"] = pd.to_datetime(completed["date"], utc=True).dt.tz_localize(None)
    completed = completed.rename(columns={"home_goals": "home_score", "away_goals": "away_score"})
    completed["neutral"] = True
    completed = completed.dropna(subset=["home_score", "away_score"])

    all_results = pd.concat([results, completed], ignore_index=True).sort_values("date").reset_index(drop=True)
    elo_ratings = compute_elo_ratings(all_results)
    model = load_model()
    today = pd.Timestamp.now()

    r32_matches = []
    for _, row in r32_df.iterrows():
        if row["status"] == "FINISHED":
            winner = row["home_team"] if row["home_goals"] > row["away_goals"] else row["away_team"]
            r32_matches.append({"home": row["home_team"], "away": row["away_team"], "winner": winner, "decided": True})
        else:
            r32_matches.append({"home": row["home_team"], "away": row["away_team"], "winner": None, "decided": False})

    print(f"Running {n_simulations} bracket simulations...")
    prob_cache = {}
    champion_counts = {}

    for i in range(n_simulations):
        champion = simulate_one_run(r32_matches, model, all_results, elo_ratings, today, prob_cache)
        champion_counts[champion] = champion_counts.get(champion, 0) + 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_simulations} done...")

    odds = [{"team": team, "win_pct": round(count / n_simulations * 100, 1)}
            for team, count in champion_counts.items()]
    odds = sorted(odds, key=lambda x: x["win_pct"], reverse=True)

    with open("data/championship_odds.json", "w") as f:
        json.dump(odds, f, indent=2)

    print("\nTop 10 Championship Odds:")
    for o in odds[:10]:
        print(f"  {o['team']}: {o['win_pct']}%")

    print("\nSaved to data/championship_odds.json")
    return odds


if __name__ == "__main__":
    run_simulation(n_simulations=1000)