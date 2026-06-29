import pandas as pd
import numpy as np
import pickle
from elo import compute_elo_ratings


def load_model():
    with open("models/xgb_model.pkl", "rb") as f:
        return pickle.load(f)


def predict_matches():
    results = pd.read_csv("data/results.csv")
    results["date"] = pd.to_datetime(results["date"])
    results = results.dropna(subset=["home_score", "away_score"])

    completed = pd.read_csv("data/wc2026_completed.csv")
    completed["date"] = pd.to_datetime(completed["date"], utc=True).dt.tz_localize(None)
    completed = completed.rename(columns={
        "home_goals": "home_score",
        "away_goals": "away_score"
    })
    completed["neutral"] = True
    completed = completed.dropna(subset=["home_score", "away_score"])

    all_results = pd.concat([results, completed], ignore_index=True)
    all_results = all_results.sort_values("date").reset_index(drop=True)

    elo_ratings = compute_elo_ratings(all_results)

    remaining = pd.read_csv("data/wc2026_remaining.csv")
    remaining = remaining.dropna(subset=["home_team", "away_team"])
    model = load_model()

    predictions = []
    for _, row in remaining.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_elo = elo_ratings.get(home, 1500)
        away_elo = elo_ratings.get(away, 1500)

        from features import get_recent_form, get_h2h
        today = pd.Timestamp.now()
        home_form = get_recent_form(home, today, all_results)
        away_form = get_recent_form(away, today, all_results)
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

        # No draws in knockout stage
        if pred == 1:
            pred = 0 if proba[0] > proba[2] else 2

        outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}

        predictions.append({
            "date":          row["date"],
            "stage":         row["stage"],
            "home_team":     home,
            "away_team":     away,
            "prediction":    outcome_map[pred],
            "home_win_prob": round(proba[0] * 100, 1),
            "draw_prob":     round(proba[1] * 100, 1),
            "away_win_prob": round(proba[2] * 100, 1),
            "home_elo":      round(home_elo, 1),
            "away_elo":      round(away_elo, 1)
        })

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv("data/predictions.csv", index=False)
    print(predictions_df[["home_team", "away_team", "prediction", "home_win_prob", "draw_prob", "away_win_prob"]].to_string())
    return predictions_df


if __name__ == "__main__":
    predict_matches()