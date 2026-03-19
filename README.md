# Field Service Delay Risk Predictor

A lightweight ML pipeline that ingests public weather data, engineers operational risk features, and scores simulated field service appointments for delay risk.

## Architecture

- Training pipeline (monthly)
  - Historical weather ingestion
  - Feature engineering
  - Model training

- Scoring pipeline (daily)
  - Forecast ingestion
  - Feature engineering
  - Prediction generation

## What it demonstrates
- API ingestion from Open-Meteo
- feature engineering for operational scenarios
- binary classification with scikit-learn
- automated training/prediction workflow with GitHub Actions
- artifact generation for reporting and review

## Important note
This project uses real weather data and an engineered operational friction proxy for appointment delay risk. It is intended as a portfolio demonstration of pipeline design, automation, and modeling workflow.



# DESIGN NOTE:
This project intentionally prioritizes pipeline structure and automation over model complexity.
The goal is to demonstrate how external data can be ingested, transformed, and used to drive predictions in a reproducible workflow, similar to how analytics engineering and ML systems operate in practice.
