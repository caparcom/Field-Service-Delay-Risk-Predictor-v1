from pathlib import Path
import os
import requests
import pandas as pd

LAT = float(os.getenv("LATITUDE", "38.7140"))
LONG = float(os.getenv("LONGITUDE", "-90.4786"))

TIMEZONE = os.getenv("TIMEZONE", "America/Chicago")
# Will make predictions on next seven days total
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "7"))

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# forecast endpoint
def fetch_forecast() -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LONG,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "snowfall_sum",
            "wind_speed_10m_max",
        ],
        "forecast_days": FORECAST_DAYS,
        "timezone": TIMEZONE,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    daily = payload["daily"]

    df = pd.DataFrame(
        {
            "date": daily["time"],
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
            "precipitation_sum": daily["precipitation_sum"],
            "snowfall_sum": daily["snowfall_sum"],
            "wind_speed_max": daily["wind_speed_10m_max"],
        }
    )

    output_path = RAW_DIR / "daily_forecast.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} forecast rows to {output_path}")

if __name__ == "__main__":
    fetch_forecast()

