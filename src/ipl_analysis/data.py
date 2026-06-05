from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "season",
    "date",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "venue",
    "winner",
}

OPTIONAL_COLUMNS = {"city", "result", "dl_applied"}


def load_matches(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    matches = pd.read_csv(path)
    matches.columns = [column.strip().lower() for column in matches.columns]

    missing = REQUIRED_COLUMNS - set(matches.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in matches dataset: {missing_list}")

    for column in REQUIRED_COLUMNS | OPTIONAL_COLUMNS:
        if column in matches.columns:
            matches[column] = matches[column].astype("string").str.strip()

    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")

    if "season" in matches.columns:
        matches["season"] = pd.to_numeric(matches["season"], errors="coerce")

    return matches


def prepare_modeling_data(matches: pd.DataFrame) -> pd.DataFrame:
    frame = matches.copy()
    frame = frame.dropna(subset=["team1", "team2", "toss_winner", "toss_decision", "venue", "winner"])

    if "result" in frame.columns:
        # Skip matches without a regular winner label, such as abandoned games.
        frame = frame[frame["result"].fillna("").str.lower() != "no result"]

    frame["city"] = frame["city"] if "city" in frame.columns else "Unknown"
    frame["dl_applied"] = pd.to_numeric(frame["dl_applied"], errors="coerce").fillna(0) if "dl_applied" in frame.columns else 0
    frame["match_year"] = frame["date"].dt.year.fillna(frame["season"]) if "date" in frame.columns else frame["season"]
    frame["toss_winner_is_team1"] = (frame["toss_winner"] == frame["team1"]).astype(int)

    feature_columns = [
        "season",
        "match_year",
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "venue",
        "city",
        "dl_applied",
        "toss_winner_is_team1",
    ]

    modeling = frame[feature_columns + ["winner"]].copy()
    modeling = modeling.rename(columns={"winner": "target"})
    return modeling

