from pathlib import Path
import joblib
import pandas as pd

DATA_PATH = Path("data/processed/training_data.csv")
MODEL_PATH = Path("models/delay_risk_model.joblib")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
model = joblib.load(MODEL_PATH)

latest_day = df["date"].max()
future = df[df["date"] == latest_day].copy()

feature_cols = [
    "appointment_hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_weekday",
    "is_rush_hour",
    "temp_max",
    "temp_min",
    "temp_range",
    "precipitation_sum",
    "snowfall_sum",
    "wind_speed_max",
    "precip_flag",
    "snow_flag",
    "wind_flag",
]

future["predicted_delay_risk_probability"] = model.predict_proba(future[feature_cols])[:, 1]
future["predicted_delay_risk"] = model.predict(future[feature_cols])

out_cols = [
    "date",
    "appointment_hour",
    "predicted_delay_risk_probability",
    "predicted_delay_risk",
]

future[out_cols].to_csv(REPORTS_DIR / "predictions.csv", index=False)

high_risk = int(future["predicted_delay_risk"].sum())
total = int(len(future))

summary = f"""# Latest Delay Risk Summary

Scored {total} simulated appointments.

High-risk appointments: {high_risk}

Average predicted delay risk probability: {future["predicted_delay_risk_probability"].mean():.3f}
"""

with open(REPORTS_DIR / "latest_summary.md", "w", encoding="utf-8") as f:
    f.write(summary)

print("Predictions complete.")