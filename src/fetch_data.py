# Pulling historical weather data from a public API.
# This acts as our external signal source (simulating real-world data ingestion).
# In a production system, this step would likely include:
# - authentication
# - pagination handling
# - retry logic
# - monitoring/alerting

from pathlib import Path
import requests
import pandas as pd

LAT = 38.7140
LONG = -90.4786
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LAT,
    "longitude": LONG,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "snowfall_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Chicago",
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()
payload = response.json()

daily = payload["daily"]

df = pd.DataFrame({
    "date": daily["time"],
    "temp_max": daily["temperature_2m_max"],
    "temp_min": daily["temperature_2m_min"],
    "precipitation_sum": daily["precipitation_sum"],
    "snowfall_sum": daily["snowfall_sum"],
    "wind_speed_max": daily["wind_speed_10m_max"],
})

df.to_csv(RAW_DIR / "weather_history.csv", index=False)
print(f"Saved {len(df)} rows to data/raw/weather_history.csv")