import requests
import pandas as pd
import os

API_KEY = "17cf805581e949669770f83675301201"
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}


def fetch_wc2026():
    response = requests.get(f"{BASE_URL}/competitions/WC/matches", headers=HEADERS)
    data = response.json()

    matches = []
    for m in data["matches"]:
        matches.append({
            "fixture_id":   m["id"],
            "date":         m["utcDate"],
            "stage":        m["stage"],
            "group":        m["group"],
            "home_team":    m["homeTeam"]["name"],
            "away_team":    m["awayTeam"]["name"],
            "home_goals":   m["score"]["fullTime"]["home"],
            "away_goals":   m["score"]["fullTime"]["away"],
            "winner":       m["score"]["winner"],
            "status":       m["status"]
        })

    df = pd.DataFrame(matches)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/wc2026_all.csv", index=False)
    print(f"Saved {len(df)} matches")
    return df


def split_fixtures(df):
    completed = df[df["status"] == "FINISHED"].copy()
    remaining = df[df["status"] == "TIMED"].copy()
    completed.to_csv("data/wc2026_completed.csv", index=False)
    remaining.to_csv("data/wc2026_remaining.csv", index=False)
    print(f"Completed: {len(completed)} | Remaining: {len(remaining)}")
    return completed, remaining


if __name__ == "__main__":
    df = fetch_wc2026()
    split_fixtures(df)