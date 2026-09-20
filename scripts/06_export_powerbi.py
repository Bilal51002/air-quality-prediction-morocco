"""
06_export_for_powerbi.py

Prepare des fichiers CSV propres, avec des noms de colonnes lisibles et
des formats adaptes, pour construire un dashboard Power BI.
Sortie : data/powerbi/*.csv (6 fichiers, ou 4 si 04_federated_learning.py
n'a pas encore tourne — voir sections 3 et 4)
"""

import pandas as pd
import os

DATA_DIR = "../data"
RESULTS_DIR = "../results"
OUT_DIR = "../data/powerbi"

os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# 1. Table "mesures" : donnees nettoyees, pretes a filtrer/agreger
# =========================================================
df = pd.read_csv(f"{DATA_DIR}/morocco_air_quality_data_clean.csv", parse_dates=["timestamp"])

mesures = df.rename(columns={
    "city": "Ville", "timestamp": "DateHeure", "pm2_5": "PM2_5", "pm10": "PM10",
    "temperature": "Temperature", "humidity": "Humidite", "pressure": "Pression",
    "wind_speed": "VitesseVent", "aqi": "AQI", "no2": "NO2", "o3": "O3",
    "weather_description": "MeteoDescription", "lat": "Latitude", "lon": "Longitude",
    "split": "JeuDonnees",
})[["DateHeure", "Ville", "Latitude", "Longitude", "Temperature", "Humidite",
    "Pression", "VitesseVent", "MeteoDescription", "AQI", "NO2", "O3", "PM10",
    "PM2_5", "JeuDonnees"]]

mesures["JeuDonnees"] = mesures["JeuDonnees"].map({"train": "Entrainement", "test": "Test"})
mesures.to_csv(f"{OUT_DIR}/mesures.csv", index=False)
print(f"mesures.csv : {mesures.shape}")

# =========================================================
# 2. Table "villes" : une ligne par ville, agregats pour la carte
# =========================================================
villes = (df.groupby("city")
            .agg(Latitude=("lat", "first"), Longitude=("lon", "first"),
                 PM2_5_Moyen=("pm2_5", "mean"), PM2_5_Max=("pm2_5", "max"),
                 NbMesures=("pm2_5", "count"))
            .reset_index()
            .rename(columns={"city": "Ville"}))
villes["PM2_5_Moyen"] = villes["PM2_5_Moyen"].round(2)
villes.to_csv(f"{OUT_DIR}/villes.csv", index=False)
print(f"villes.csv : {villes.shape}")

# =========================================================
# 3. Table "comparaison_modeles" : les 3 modeles centralises (etape 3)
#    + la comparaison federe/centralise (etape 4), format long unifie
#
#    federated_vs_centralized_results.csv est produit par
#    04_federated_learning.py, lance a la main (hors Airflow, incompatible
#    avec flwr — voir le DAG). S'il n'existe pas encore, on continue sans
#    cette partie plutot que de planter.
# =========================================================
step3 = pd.read_csv(f"{RESULTS_DIR}/step3_model_results.csv")
step3["Categorie"] = "Modele centralise"

fed_path = f"{RESULTS_DIR}/federated_vs_centralized_results.csv"
if os.path.exists(fed_path):
    fed = pd.read_csv(fed_path)
    fed = fed.rename(columns={"Approach": "Model"})
    fed["Categorie"] = "Federe vs centralise"
    comparaison = pd.concat([step3, fed], ignore_index=True)
else:
    print(f"ATTENTION : {fed_path} introuvable (lance 04_federated_learning.py a la main "
          f"pour l'inclure). Export limite aux modeles centralises.")
    comparaison = step3

comparaison = comparaison.rename(columns={"Model": "Modele"})
comparaison.to_csv(f"{OUT_DIR}/comparaison_modeles.csv", index=False)
print(f"comparaison_modeles.csv : {comparaison.shape}")

# =========================================================
# 4. Table "courbe_federee" : convergence par round (pour un line chart)
#    Meme logique : optionnelle, produite uniquement par 04.
# =========================================================
courbe_path = f"{RESULTS_DIR}/federated_learning_curve.csv"
if os.path.exists(courbe_path):
    courbe = pd.read_csv(courbe_path)
    courbe = courbe.rename(columns={"round": "Round", "mae": "MAE", "rmse": "RMSE", "r2": "R2"})
    courbe.to_csv(f"{OUT_DIR}/courbe_federee.csv", index=False)
    print(f"courbe_federee.csv : {courbe.shape}")
else:
    print(f"ATTENTION : {courbe_path} introuvable. courbe_federee.csv non genere.")

# =========================================================
# 5. Table "importance_features" : Random Forest complet vs meteo seule
# =========================================================
imp_full = pd.read_csv(f"{RESULTS_DIR}/feature_importance.csv")
imp_full.columns = ["Feature", "Importance"]
imp_full["ModeleType"] = "Complet (avec pollution)"

imp_weather = pd.read_csv(f"{RESULTS_DIR}/feature_importance_weather_only.csv")
imp_weather.columns = ["Feature", "Importance"]
imp_weather["ModeleType"] = "Meteo seule"

importance = pd.concat([imp_full, imp_weather], ignore_index=True)
importance["Importance"] = (importance["Importance"] * 100).round(2)
importance = importance.rename(columns={"Importance": "Importance_pct"})
importance.to_csv(f"{OUT_DIR}/importance_features.csv", index=False)
print(f"importance_features.csv : {importance.shape}")

# =========================================================
# 6. Table "meteo_vs_complet" : resultat du script 05
# =========================================================
weather_comp = pd.read_csv(f"{RESULTS_DIR}/weather_only_vs_full_results.csv")
weather_comp = weather_comp.rename(columns={
    "Model": "Modele", "Feature set": "JeuFeatures",
})
weather_comp.to_csv(f"{OUT_DIR}/meteo_vs_complet.csv", index=False)
print(f"meteo_vs_complet.csv : {weather_comp.shape}")

print(f"\nTous les fichiers exportes dans : {OUT_DIR}/")