import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


def train_model():
    df = pd.read_csv("data/features.csv")

    features = ["elo_diff", "home_elo", "away_elo", "home_form", "away_form", "h2h", "neutral"]
    X = df[features]
    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    class_counts = y_train.value_counts()
    total = len(y_train)
    class_weights = {cls: total / (len(class_counts) * count)
                     for cls, count in class_counts.items()}
    sample_weights = y_train.map(class_weights)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

    os.makedirs("models", exist_ok=True)
    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to models/xgb_model.pkl")

    return model


if __name__ == "__main__":
    train_model()