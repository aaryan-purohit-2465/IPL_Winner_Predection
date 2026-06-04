from __future__ import annotations

import base64
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from .config import MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR
from .data import load_matches
from .report import build_report
from .train import train_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web"
DEFAULT_DATASET = PROJECT_ROOT / "examples" / "sample_matches.csv"
UPLOADED_DATASET = RAW_DATA_DIR / "uploaded_matches.csv"
MODEL_PATH = MODELS_DIR / "ipl_winner_model.joblib"


app = Flask(__name__, template_folder=str(WEB_DIR / "templates"), static_folder=str(WEB_DIR / "static"))


def resolve_dataset_path() -> Path:
    if UPLOADED_DATASET.exists():
        return UPLOADED_DATASET
    if DEFAULT_DATASET.exists():
        return DEFAULT_DATASET
    raise FileNotFoundError("No dataset found. Upload a matches.csv file or add examples/sample_matches.csv.")


def dataset_summary(matches: pd.DataFrame) -> dict:
    venues = sorted(matches["venue"].dropna().astype(str).unique().tolist())
    cities = sorted(matches["city"].dropna().astype(str).unique().tolist()) if "city" in matches.columns else []
    teams = sorted(pd.unique(pd.concat([matches["team1"], matches["team2"]]).dropna()).tolist())
    seasons = sorted(matches["season"].dropna().astype(int).unique().tolist())
    return {
        "dataset_rows": int(len(matches)),
        "teams": teams,
        "venues": venues,
        "cities": cities,
        "seasons": seasons,
    }


def image_to_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def prepare_frontend_payload(dataset_path: Path) -> dict:
    matches = load_matches(dataset_path)
    report = build_report(dataset_path)
    metrics = train_model(dataset_path)

    payload = {
        "dataset": {
            "path": str(dataset_path),
            "name": dataset_path.name,
            **dataset_summary(matches),
        },
        "report": report,
        "metrics": {key: value for key, value in metrics.items() if key != "classification_report"},
        "charts": {
            "top_winning_teams": image_to_data_uri(REPORTS_DIR / "figures" / "top_winning_teams.png"),
            "top_toss_winners": image_to_data_uri(REPORTS_DIR / "figures" / "top_toss_winners.png"),
        },
    }
    return payload


def ensure_model_ready() -> tuple[Path, dict]:
    dataset_path = resolve_dataset_path()
    payload = prepare_frontend_payload(dataset_path)
    return dataset_path, payload


def load_saved_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Upload a dataset to train one first.")
    return joblib.load(MODEL_PATH)


def normalize_prediction_input(payload: dict) -> pd.DataFrame:
    row = {
        "season": payload["season"],
        "match_year": payload.get("match_year", payload["season"]),
        "team1": payload["team1"],
        "team2": payload["team2"],
        "toss_winner": payload["toss_winner"],
        "toss_decision": payload["toss_decision"],
        "venue": payload["venue"],
        "city": payload.get("city") or "Unknown",
        "dl_applied": payload.get("dl_applied", 0),
        "toss_winner_is_team1": int(payload["toss_winner"] == payload["team1"]),
    }
    return pd.DataFrame([row])


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/dashboard-data")
def dashboard_data():
    dataset_path, payload = ensure_model_ready()
    payload["dataset"]["active_dataset"] = str(dataset_path)
    return jsonify(payload)


@app.post("/api/upload")
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "Please select a CSV file to upload."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Uploaded file is empty."}), 400

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Uploaded file has no content."}), 400

    UPLOADED_DATASET.write_bytes(file_bytes)

    try:
        payload = prepare_frontend_payload(UPLOADED_DATASET)
    except Exception as exc:
        if UPLOADED_DATASET.exists():
            UPLOADED_DATASET.unlink()
        return jsonify({"error": str(exc)}), 400

    return jsonify(payload), 201


@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        model = load_saved_model()
        prediction_input = normalize_prediction_input(payload)
        probabilities = model.predict_proba(prediction_input)[0]
        classes = model.classes_
        ranked_probabilities = sorted(
            [
                {"team": str(team), "probability": round(float(prob), 4)}
                for team, prob in zip(classes, probabilities, strict=False)
            ],
            key=lambda item: item["probability"],
            reverse=True,
        )
        winner = str(model.predict(prediction_input)[0])
    except KeyError as exc:
        return jsonify({"error": f"Missing field: {exc.args[0]}"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "predicted_winner": winner,
            "confidence": ranked_probabilities[0]["probability"] if ranked_probabilities else None,
            "top_probabilities": ranked_probabilities[:3],
        }
    )


@app.get("/api/model-card")
def model_card():
    metrics_path = REPORTS_DIR / "model_metrics.json"
    if not metrics_path.exists():
        return jsonify({"error": "Metrics not found. Train the model first."}), 404
    return jsonify(json.loads(metrics_path.read_text(encoding="utf-8")))


def main() -> None:
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
