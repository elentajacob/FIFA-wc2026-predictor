import pandas as pd
import numpy as np


def get_recent_form(team, date, results_df, n=10):
    past = results_df[
        ((results_df["home_team"] == team) | (results_df["away_team"] == team)) &
        (results_df["date"] < date)
    ].tail(n)

    if len(past) == 0:
        return 0.5

    wins = 0
    for _, row in past.iterrows():
        if row["home_team"] == team and row["home_score"] > row["away_score"]:
            wins += 1
        elif row["away_team"] == team and row["away_score"] > row["home_score"]:
            wins += 1

    return wins / len(past)


def get_h2h(home_team, away_team, date, results_df, n=10):
    h2h = results_df[
        (
            ((results_df["home_team"] == home_team) & (results_df["away_team"] == away_team)) |
            ((results_df["home_team"] == away_team) & (results_df["away_team"] == home_team))
        ) &
        (results_df["date"] < date)
    ].tail(n)

    if len(h2h) == 0:
        return 0.5

    home_wins = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == home_team and row["home_score"] > row["away_score"]:
            home_wins += 1
        elif row["away_team"] == home_team and row["away_score"] > row["home_score"]:
            home_wins += 1

    return home_wins / len(h2h)


def build_features(results_df, elo_ratings):
    results_df = results_df.copy()
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df = results_df.sort_values("date").reset_index(drop=True)
    results_df = results_df.dropna(subset=["home_score", "away_score"])

    rows = []
    for idx, row in results_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        home_elo = elo_ratings.get(home, 1500)
        away_elo = elo_ratings.get(away, 1500)

        home_form = get_recent_form(home, date, results_df)
        away_form = get_recent_form(away, date, results_df)
        h2h = get_h2h(home, away, date, results_df)

        if row["home_score"] > row["away_score"]:
            outcome = 0  # home win
        elif row["home_score"] < row["away_score"]:
            outcome = 2  # away win
        else:
            outcome = 1  # draw

        rows.append({
            "home_team":   home,
            "away_team":   away,
            "date":        date,
            "elo_diff":    home_elo - away_elo,
            "home_elo":    home_elo,
            "away_elo":    away_elo,
            "home_form":   home_form,
            "away_form":   away_form,
            "h2h":         h2h,
            "neutral":     int(row["neutral"]),
            "outcome":     outcome
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = pd.read_csv("data/results.csv")
    elo_df = pd.read_csv("data/elo_ratings.csv")
    elo_ratings = dict(zip(elo_df["team"], elo_df["elo_rating"]))

    print("Building features, this will take a few minutes...")
    features_df = build_features(results, elo_ratings)
    features_df.to_csv("data/features.csv", index=False)
    print(f"Done! Feature matrix shape: {features_df.shape}")
    print(features_df.head())