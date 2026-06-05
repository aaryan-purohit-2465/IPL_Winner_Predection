from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES_DIR, REPORTS_DIR
from .data import load_matches


def build_report(matches_path: str) -> dict:
    matches = load_matches(matches_path)

    total_matches = int(len(matches))
    total_seasons = int(matches["season"].nunique(dropna=True))
    winner_counts = matches["winner"].value_counts(dropna=True)
    toss_counts = matches["toss_winner"].value_counts(dropna=True)

    report = {
        "total_matches": total_matches,
        "total_seasons": total_seasons,
        "top_winning_teams": winner_counts.head(5).to_dict(),
        "top_toss_winners": toss_counts.head(5).to_dict(),
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    winner_counts.head(8).plot(kind="bar", color="#1f77b4")
    plt.title("Top IPL Teams by Match Wins")
    plt.xlabel("Team")
    plt.ylabel("Wins")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_winning_teams.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    toss_counts.head(8).plot(kind="bar", color="#ff7f0e")
    plt.title("Top IPL Teams by Toss Wins")
    plt.xlabel("Team")
    plt.ylabel("Toss Wins")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_toss_winners.png", dpi=200)
    plt.close()

    with open(REPORTS_DIR / "summary_report.json", "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IPL analysis charts and summary report.")
    parser.add_argument("--matches", required=True, help="Path to the matches.csv dataset")
    args = parser.parse_args()

    report = build_report(args.matches)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

