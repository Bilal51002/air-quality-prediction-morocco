"""
05_weather_only_comparison.py

Compare les performances du modele en gardant TOUTES les features
(y compris pm10, tres correle a la cible) vs en ne gardant QUE les
variables meteo/spatiales/temporelles, pour repondre honnetement a la
question posee par l'objectif du projet : "que peut-on predire a
partir de la meteo seule ?"

Constat de depart (notebook EDA) : pm10 est correle a 0.87 avec pm2_5,
et capte a lui seul 87% de l'importance des features du Random Forest
entraine dans 03_build_model.py. Ce script quantifie l'ecart.
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_FILE = "../data/morocco_air_quality_data_model_ready.csv"
OUTPUT_RESULTS = "../results/weather_only_vs_full_results.csv"
TARGET = "pm2_5"

# Colonnes de pollution (derivees de mesures de particules/gaz, pas de la meteo)
POLLUTION_COLS = ["aqi", "no2", "o3", "pm10"]

df = pd.read_csv(INPUT_FILE)

train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

all_features = [c for c in df.columns if c not in [TARGET, "split"]]
weather_only_features = [c for c in all_features if c not in POLLUTION_COLS]

print(f"Nombre de features (complet)      : {len(all_features)}")
print(f"Nombre de features (meteo seule)  : {len(weather_only_features)}")
print(f"Colonnes de pollution retirees    : {POLLUTION_COLS}")

X_train_full, y_train = train_df[all_features], train_df[TARGET]
X_test_full, y_test = test_df[all_features], test_df[TARGET]

X_train_weather = train_df[weather_only_features]
X_test_weather = test_df[weather_only_features]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

results = []

for model_name, model_template in models.items():
    for label, X_train, X_test in [
        ("Complet (avec pm10/aqi/no2/o3)", X_train_full, X_test_full),
        ("Meteo seule (sans pollution)", X_train_weather, X_test_weather),
    ]:
        # Nouvelle instance a chaque fois pour ne pas reutiliser un modele deja fit
        model = type(model_template)(**model_template.get_params())
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"\n{model_name} — {label}")
        print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

        results.append({
            "Model": model_name,
            "Feature set": label,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

results_df = pd.DataFrame(results)
print("\n===== Comparaison complete =====")
print(results_df.to_string(index=False))

results_df.to_csv(OUTPUT_RESULTS, index=False)
print(f"\nResultats sauvegardes dans : {OUTPUT_RESULTS}")

# Feature importance du Random Forest "meteo seule", pour voir quelle
# variable meteo pese le plus une fois la pollution retiree.
rf_weather = RandomForestRegressor(n_estimators=100, random_state=42)
rf_weather.fit(X_train_weather, y_train)
importances = pd.Series(rf_weather.feature_importances_, index=weather_only_features)
importances = importances.sort_values(ascending=False)

print("\n===== Top 10 features (Random Forest, meteo seule) =====")
print(importances.head(10))
importances.to_csv("../results/feature_importance_weather_only.csv", header=["importance"])
