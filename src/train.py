# Training a classification model to predict delay risk based on engineered features.
# Note:
# The model is learning from synthetic labels derived from a heuristic,
# not real observed outcomes. This is intentional for demonstration purposes.

from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/training_data.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

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

X = df[feature_cols]
y = df["delay_risk"]

# Splitting data to evaluate generalization.
# Even though labels are synthetic, this still ensures the model
# is not simply memorizing the training data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Standard classification metrics to evaluate model performance.
# In a real system, additional monitoring would be required over time
# to detect drift or degradation.
metrics = {
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "precision": round(float(precision_score(y_test, y_pred)), 4),
    "recall": round(float(recall_score(y_test, y_pred)), 4),
    "f1": round(float(f1_score(y_test, y_pred)), 4),
    "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
}

with open(REPORTS_DIR / "model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(model, MODELS_DIR / "delay_risk_model.joblib")

# Feature importance provides insight into which variables the model
# relied on most when approximating the delay risk signal.
# Since labels are derived from a predefined function,
# importance values should broadly align with the weights used
# in the operational friction index.
importance_df = (
    pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    })
    .sort_values("importance", ascending=True)
)

plt.figure(figsize=(8, 5))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.title("Feature Importance - Delay Risk Model")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
plt.close()

print("Training complete:")
print(metrics)