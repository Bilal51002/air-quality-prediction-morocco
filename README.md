# Air Quality Prediction in Moroccan Cities

This project aims to predict air quality in Moroccan cities using real-world weather and air pollution data collected from the OpenWeatherMap API.

## Project Overview

The objective of this project is to build machine learning models capable of predicting PM2.5 concentration in Moroccan cities based on meteorological and pollution-related features.

The project also includes a simple federated learning simulation where each city is considered as a local client.

## Cities Used

- Casablanca
- Rabat
- Marrakech
- Fes

## Dataset

The data was collected using the OpenWeatherMap API and includes:

### Weather features

- Temperature
- Feels-like temperature
- Humidity
- Pressure
- Wind speed
- Wind direction
- Cloud coverage
- Weather description

### Air pollution features

- AQI
- CO
- NO
- NO2
- O3
- SO2
- PM2.5
- PM10
- NH3

## Project Structure

```text
air-quality-prediction-morocco/
│
├── data/
│   ├── morocco_air_quality_data.csv
│   ├── morocco_air_quality_data_clean.csv
│   └── morocco_air_quality_data_model_ready.csv
│
├── results/
│   ├── step3_model_results.csv
│   └── federated_vs_centralized_results.csv
│
├── scripts/
│   ├── 01_collect_data.py
│   ├── 02_prepare_data.py
│   ├── 03_build_model.py
│   └── 04_federated_learning.py
│
├── report/
│   └── rapport_Air_Quality_Prediction_in_Moroccan_Cities.pdf
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE