# IPL Match Data Analysis and Winner Prediction

This project now includes a complete IPL analytics web app with:

- a Python backend API
- a browser-based frontend dashboard
- CSV upload and retraining flow
- winner prediction for custom match scenarios

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   |-- processed/
|-- models/
|-- notebooks/
|-- reports/
|   |-- figures/
|-- src/
|   |-- ipl_analysis/
|-- web/
|   |-- templates/
|   |-- static/
|-- requirements.txt
```

## Expected Dataset

Place your IPL `matches.csv` file inside `data/raw/`.

The starter code expects a dataset with columns similar to the common IPL match dataset:

- `season`
- `date`
- `team1`
- `team2`
- `toss_winner`
- `toss_decision`
- `venue`
- `city`
- `winner`
- `result`

If your dataset has slightly different column names, you can update the mappings in [src/ipl_analysis/data.py](C:/Users/paary/Documents/New project/src/ipl_analysis/data.py).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Analysis From the Command Line

Generate summary charts and a JSON report:

```powershell
python -m src.ipl_analysis.report --matches data/raw/matches.csv
```

## Train Winner Prediction Model From the Command Line

```powershell
python -m src.ipl_analysis.train --matches data/raw/matches.csv
```

This saves:

- trained model to `models/ipl_winner_model.joblib`
- metrics to `reports/model_metrics.json`
- processed training data to `data/processed/modeling_dataset.csv`

## Run the Full Web App

Start the backend and frontend together with:

```powershell
python -m src.ipl_analysis.api
```

Then open:

```text
http://127.0.0.1:5000
```

## Web App Features

- upload an IPL `matches.csv` file from the browser
- retrain the model automatically after upload
- view dataset stats and model metrics
- inspect generated charts for winning teams and toss winners
- predict the winner for a custom match setup

## API Endpoints

- `GET /api/health`
- `GET /api/dashboard-data`
- `POST /api/upload`
- `POST /api/predict`
- `GET /api/model-card`

## Verified Commands

These commands were run successfully in this workspace:

```powershell
python -m src.ipl_analysis.report --matches examples/sample_matches.csv
python -m src.ipl_analysis.train --matches examples/sample_matches.csv
python -m src.ipl_analysis.api
```

## Suggested Next Upgrades

- Add player-level or ball-by-ball features from a `deliveries.csv` dataset
- Compare `RandomForestClassifier` with `XGBoost` or `LightGBM`
- Add recent-form, head-to-head, and venue-strength features
- Add authentication and saved experiment history
- Add notebook experiments inside `notebooks/`
