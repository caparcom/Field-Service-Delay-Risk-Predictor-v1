# This step transforms raw weather data into a modeling dataset.
# We expand each day into multiple simulated appointment slots and engineer
# features that represent operational conditions (time, weather, etc.).
# This mimics how raw external data would be shaped into business-relevant features
# in an analytics engineering workflow.

from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

RAW_PATH = Path("data/raw/weather_history.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

weather = pd.read_csv(RAW_PATH, parse_dates=["date"])

# Simulating discrete appointment slots throughout the day.
# This allows us to model intra-day variation (e.g., rush hour vs midday).
# In a real system, this would come from actual scheduling data.
appointment_hours = [8, 10, 12, 14, 16]
appt_df = weather.loc[weather.index.repeat(len(appointment_hours))].copy()
appt_df["appointment_hour"] = appointment_hours * len(weather)

# Calendar-based features that capture predictable patterns in operations.
# For example:
# - rush hour → increased travel time
# - weekday vs weekend → different demand/congestion patterns
appt_df["day_of_week"] = appt_df["date"].dt.dayofweek
appt_df["month"] = appt_df["date"].dt.month
appt_df["is_weekend"] = appt_df["day_of_week"].isin([5, 6]).astype(int)
appt_df["is_weekday"] = 1 - appt_df["is_weekend"]
appt_df["is_rush_hour"] = appt_df["appointment_hour"].isin([8, 16]).astype(int)

appt_df["temp_range"] = appt_df["temp_max"] - appt_df["temp_min"]
appt_df["precip_flag"] = (appt_df["precipitation_sum"] > 0).astype(int)
appt_df["snow_flag"] = (appt_df["snowfall_sum"] > 0).astype(int)
appt_df["wind_flag"] = (appt_df["wind_speed_max"] >= 20).astype(int)

# Adding stochastic noise to avoid a perfectly deterministic relationship.
# This helps simulate real-world unpredictability and prevents the model
# from trivially memorizing the synthetic labeling function.
noise = np.random.normal(loc=0.0, scale=0.2, size=len(appt_df))

# NOTE:
# We do not have real labels like "this appointment was delayed".
# Instead, we construct a synthetic "operational friction index" to simulate how
# difficult an appointment is under given conditions (weather, time, etc.).

# The delay_risk label is then derived from this index.
# This means the model is effectively learning to approximate this heuristic.

# Important caveat:
# Feature importance is NOT discovering new patterns in real-world data
# it is largely recovering the assumptions encoded below.

# This approach is useful for demonstrating pipeline design and modeling workflows,
# but in a real system, these labels would come from actual observed outcomes.
appt_df["operational_friction_index"] = (
    1.0
    + 1.4 * appt_df["is_rush_hour"]
    + 0.10 * appt_df["precipitation_sum"]
    + 0.18 * appt_df["snowfall_sum"]
    + 0.05 * appt_df["wind_speed_max"]
    + 0.25 * appt_df["is_weekday"]
    + noise
)

threshold = appt_df["operational_friction_index"].quantile(0.65)
appt_df["delay_risk"] = (appt_df["operational_friction_index"] >= threshold).astype(int)

columns = [
    "date",
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
    "operational_friction_index",
    "delay_risk",
]

appt_df[columns].to_csv(OUT_DIR / "training_data.csv", index=False)
print(f"Saved {len(appt_df)} rows to data/processed/training_data.csv")