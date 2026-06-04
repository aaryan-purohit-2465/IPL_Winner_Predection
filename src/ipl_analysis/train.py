from __future__ import annotations

import argparse
import json

import joblib
import math
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from .data import load_matches, prepare_modeling_data


def train_model(matches_path: str) -> dict:
    matches = load_matches(matches_path)
    modeling = prepare_modeling_data(matches)

    X = modeling.drop(columns=["target"])
    y = modeling["target"]
    class_count = int(y.nunique())
    class_frequencies = y.value_counts()

    categorical_features = ["team1", "team2", "toss_winner", "toss_decision", "venue", "city"]
    numeric_features = ["season", "match_year", "dl_applied", "toss_winner_is_team1"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=300, max_depth=18, random_state=42)),
        ]
    )

    default_test_size = 0.2
    test_rows = math.ceil(len(X) * default_test_size)
    use_stratify = class_frequencies.min() >= 2 and test_rows >= class_count

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=default_test_size,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "stratified_split": use_stratify,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted")), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / "ipl_winner_model.joblib")
    modeling.to_csv(PROCESSED_DATA_DIR / "modeling_dataset.csv", index=False)

    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an IPL match winner prediction model.")
    parser.add_argument("--matches", required=True, help="Path to the matches.csv dataset")
    args = parser.parse_args()

    metrics = train_model(args.matches)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
