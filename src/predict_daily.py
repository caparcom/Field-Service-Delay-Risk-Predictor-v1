# This step generates predictions using the most recent available data.
# forward-looking forecast data to support real-time operational decision-making.

from pathlib import Path
import os
import joblib
import pandas as pd

FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "7"))

RAW_PATH = Path("data/raw/daily_forecast.csv")
MODEL_PATH = Path("models/delay_risk_model.joblib")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
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

def build_future_appointments(weather_df: pd.DataFrame) -> pd.DataFrame:
    appointment_hours = [8, 10, 12, 14, 16]

    appt_df = weather_df.loc[weather_df.index.repeat(len(appointment_hours))].copy()
    appt_df["appointment_hour"] = appointment_hours * len(weather_df)

    appt_df["date"] = pd.to_datetime(appt_df["date"])
    appt_df["day_of_week"] = appt_df["date"].dt.dayofweek
    appt_df["month"] = appt_df["date"].dt.month
    appt_df["is_weekend"] = appt_df["day_of_week"].isin([5, 6]).astype(int)
    appt_df["is_weekday"] = 1 - appt_df["is_weekend"]
    appt_df["is_rush_hour"] = appt_df["appointment_hour"].isin([8, 16]).astype(int)

    appt_df["temp_range"] = appt_df["temp_max"] - appt_df["temp_min"]
    appt_df["precip_flag"] = (appt_df["precipitation_sum"] > 0).astype(int)
    appt_df["snow_flag"] = (appt_df["snowfall_sum"] > 0).astype(int)
    appt_df["wind_flag"] = (appt_df["wind_speed_max"] >= 20).astype(int)

    return appt_df

def add_risk_band(df: pd.DataFrame) -> pd.DataFrame:
    def label(prob: float) -> str:
        if prob >= 0.70:
            return "High"
        if prob >= 0.40:
            return "Medium"
        return "Low"

    df["risk_band"] = df["predicted_delay_risk_probability"].apply(label)
    return df

def write_summary(df: pd.DataFrame) -> None:
    total = len(df)
    high = int((df["risk_band"] == "High").sum())
    medium = int((df["risk_band"] == "Medium").sum())
    low = int((df["risk_band"] == "Low").sum())
    avg_prob = float(df["predicted_delay_risk_probability"].mean())

    summary = f"""# 7-Day Delay Risk Forecast

Scored {total} simulated appointments across the next {FORECAST_DAYS} days.

## Risk Breakdown
- High risk: {high}
- Medium risk: {medium}
- Low risk: {low}

## Average Predicted Delay Risk Probability
- {avg_prob:.3f}
"""

    with open(REPORTS_DIR / "latest_summary.md", "w", encoding="utf-8") as f:
        f.write(summary)

def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Forecast data not found at {RAW_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    weather_df = pd.read_csv(RAW_PATH)
    model = joblib.load(MODEL_PATH)

    future_df = build_future_appointments(weather_df)

    future_df["predicted_delay_risk_probability"] = model.predict_proba(
        future_df[FEATURE_COLS]
    )[:, 1]
    future_df["predicted_delay_risk"] = model.predict(future_df[FEATURE_COLS])

    future_df = add_risk_band(future_df)

    out_cols = [
        "date",
        "appointment_hour",
        "temp_max",
        "temp_min",
        "precipitation_sum",
        "snowfall_sum",
        "wind_speed_max",
        "predicted_delay_risk_probability",
        "predicted_delay_risk",
        "risk_band",
    ]

    future_df[out_cols].to_csv(REPORTS_DIR / "predictions.csv", index=False)
    write_summary(future_df)

    print("Daily forecast scoring complete.")
    print(future_df[out_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()