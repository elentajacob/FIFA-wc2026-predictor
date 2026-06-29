import pandas as pd
import numpy as np

K = 32          # how fast ratings change
HOME_ADVANTAGE = 100


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_home, rating_away, home_score, away_score):
    expected_home = expected_score(rating_home + HOME_ADVANTAGE, rating_away)
    expected_away = 1 - expected_home

    if home_score > away_score:
        actual_home, actual_away = 1, 0
    elif home_score < away_score:
        actual_home, actual_away = 0, 1
    else:
        actual_home, actual_away = 0.5, 0.5

    new_home = rating_home + K * (actual_home - expected_home)
    new_away = rating_away + K * (actual_away - expected_away)
    return new_home, new_away


def compute_elo_ratings(results_df):
    ratings = {}

    for _, row in results_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in ratings:
            ratings[home] = 1500
        if away not in ratings:
            ratings[away] = 1500

        new_home, new_away = update_elo(
            ratings[home],
            ratings[away],
            row["home_score"],
            row["away_score"]
        )
        ratings[home] = new_home
        ratings[away] = new_away

    return ratings


if __name__ == "__main__":
    results = pd.read_csv("data/results.csv")
    results = results.dropna(subset=["home_score", "away_score"])
    results["date"] = pd.to_datetime(results["date"])
    results = results.sort_values("date")

    ratings = compute_elo_ratings(results)

    ratings_df = pd.DataFrame(
        ratings.items(), columns=["team", "elo_rating"]
    ).sort_values("elo_rating", ascending=False)

    ratings_df.to_csv("data/elo_ratings.csv", index=False)
    print(ratings_df.head(20))