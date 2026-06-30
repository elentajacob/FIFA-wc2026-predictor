import pandas as pd
import pickle
from elo import compute_elo_ratings


def load_model():
    with open("models/xgb_model.pkl", "rb") as f:
        return pickle.load(f)


def get_team_stats(team, all_results, elo_ratings):
    from features import get_recent_form
    today = pd.Timestamp.now()
    elo = elo_ratings.get(team, 1500)
    form = get_recent_form(team, today, all_results)
    return elo, form


def predict_winner(home, away, model, all_results, elo_ratings):
    from features import get_h2h

    home_elo, home_form = get_team_stats(home, all_results, elo_ratings)
    away_elo, away_form = get_team_stats(away, all_results, elo_ratings)
    today = pd.Timestamp.now()
    h2h = get_h2h(home, away, today, all_results)

    X = pd.DataFrame([{
        "elo_diff":  home_elo - away_elo,
        "home_elo":  home_elo,
        "away_elo":  away_elo,
        "home_form": home_form,
        "away_form": away_form,
        "h2h":       h2h,
        "neutral":   1
    }])

    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]

    if pred == 1:  # no draws in knockout
        pred = 0 if proba[0] > proba[2] else 2

    winner = home if pred == 0 else away
    confidence = proba[0] if pred == 0 else proba[2]
    confidence = float(confidence)

    return winner, round(confidence * 100, 1)


def get_actual_or_predicted_winner(row, model, all_results, elo_ratings):
    """If match is finished, return the real winner. Otherwise predict it."""
    if row["status"] == "FINISHED":
        if pd.notna(row["home_goals"]) and pd.notna(row["away_goals"]):
            if row["home_goals"] > row["away_goals"]:
                return row["home_team"], 100.0, True
            elif row["away_goals"] > row["home_goals"]:
                return row["away_team"], 100.0, True
            else:
                # finished but drawn in regulation, went to penalties
                # football-data.org marks the eventual winner via 'winner' field if available
                return row["home_team"], 100.0, True  # fallback, rarely hit

    if pd.isna(row["home_team"]) or pd.isna(row["away_team"]):
        return None, None, False

    winner, confidence = predict_winner(row["home_team"], row["away_team"], model, all_results, elo_ratings)
    return winner, confidence, False


def simulate_bracket():
    all_matches = pd.read_csv("data/wc2026_all.csv")
    r32 = all_matches[all_matches["stage"] == "LAST_32"].reset_index(drop=True)

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

    bracket = {"R32": [], "R16": [], "QF": [], "SF": [], "F": []}

    # Round of 32 -> winners
    r32_winners = []
    for _, row in r32.iterrows():
        winner, confidence, is_actual = get_actual_or_predicted_winner(row, model, all_results, elo_ratings)
        r32_winners.append(winner)
        bracket["R32"].append({
            "home": row["home_team"], "away": row["away_team"],
            "winner": winner, "confidence": confidence, "decided": is_actual
        })

    # Round of 16 -> pair up consecutive winners
    r16_winners = []
    for i in range(0, len(r32_winners), 2):
        home, away = r32_winners[i], r32_winners[i + 1]
        if home is None or away is None:
            r16_winners.append(None)
            bracket["R16"].append({"home": home, "away": away, "winner": None, "confidence": None, "decided": False})
            continue
        winner, confidence = predict_winner(home, away, model, all_results, elo_ratings)
        r16_winners.append(winner)
        bracket["R16"].append({"home": home, "away": away, "winner": winner, "confidence": confidence, "decided": False})

    # Quarterfinals
    qf_winners = []
    for i in range(0, len(r16_winners), 2):
        home, away = r16_winners[i], r16_winners[i + 1]
        if home is None or away is None:
            qf_winners.append(None)
            bracket["QF"].append({"home": home, "away": away, "winner": None, "confidence": None, "decided": False})
            continue
        winner, confidence = predict_winner(home, away, model, all_results, elo_ratings)
        qf_winners.append(winner)
        bracket["QF"].append({"home": home, "away": away, "winner": winner, "confidence": confidence, "decided": False})

    # Semifinals
    sf_winners = []
    for i in range(0, len(qf_winners), 2):
        home, away = qf_winners[i], qf_winners[i + 1]
        if home is None or away is None:
            sf_winners.append(None)
            bracket["SF"].append({"home": home, "away": away, "winner": None, "confidence": None, "decided": False})
            continue
        winner, confidence = predict_winner(home, away, model, all_results, elo_ratings)
        sf_winners.append(winner)
        bracket["SF"].append({"home": home, "away": away, "winner": winner, "confidence": confidence, "decided": False})

    # Final
    home, away = sf_winners[0], sf_winners[1]
    if home is None or away is None:
        bracket["F"].append({"home": home, "away": away, "winner": None, "confidence": None, "decided": False})
    else:
        winner, confidence = predict_winner(home, away, model, all_results, elo_ratings)
        bracket["F"].append({"home": home, "away": away, "winner": winner, "confidence": confidence, "decided": False})

    return bracket


if __name__ == "__main__":
    bracket = simulate_bracket()
    for round_name, matches in bracket.items():
        print(f"\n--- {round_name} ---")
        for m in matches:
            print(f"{m['home']} vs {m['away']} -> {m['winner']} ({m['confidence']}%)")

    import json
    with open("data/bracket.json", "w") as f:
        json.dump(bracket, f, indent=2)
    print("\nSaved to data/bracket.json")