import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. Charger le dataset prêt
# =========================
INPUT_FILE = "../data/morocco_air_quality_data_model_ready.csv"
OUTPUT_RESULTS = "../results/step3_model_results.csv"
TARGET = "pm2_5"

df = pd.read_csv(INPUT_FILE)

print("Aperçu du dataset :")
print(df.head())
print("\nTaille du dataset :", df.shape)

# =========================
# 2. Utiliser le split chronologique déjà fait dans 02_prepare_data.py
#    (au lieu d'un train_test_split aléatoire qui recréerait une fuite
#    temporelle entre mesures consécutives)
# =========================
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

X_train = train_df.drop(columns=[TARGET, "split"])
y_train = train_df[TARGET]
X_test = test_df.drop(columns=[TARGET, "split"])
y_test = test_df[TARGET]

print("\nTaille X_train :", X_train.shape)
print("Taille X_test :", X_test.shape)
print("Taille y_train :", y_train.shape)
print("Taille y_test :", y_test.shape)

# =========================
# 3. Définir les modèles
# =========================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

# =========================
# 4. Entraîner et évaluer
# =========================
results = []
trained_models = {}

for model_name, model in models.items():
    print(f"\n===== {model_name} =====")

    # Entraînement
    model.fit(X_train, y_train)
    trained_models[model_name] = model

    # Prédiction
    y_pred = model.predict(X_test)

    # Métriques
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("MAE  :", round(mae, 4))
    print("RMSE :", round(rmse, 4))
    print("R²   :", round(r2, 4))

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# =========================
# 5. Tableau récapitulatif
# =========================
results_df = pd.DataFrame(results)

print("\n===== Résultats finaux (split chronologique) =====")
print(results_df.sort_values(by="RMSE"))

# =========================
# 6. Importance des features (Random Forest)
#    Utile pour expliquer le modèle, pas juste ses métriques.
# =========================
rf = trained_models["Random Forest Regressor"]
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)

print("\n===== Top 10 features les plus importantes (Random Forest) =====")
print(importances.head(10))

importances.to_csv("../results/feature_importance.csv", header=["importance"])

# =========================
# 7. Sauvegarder les résultats
# =========================
results_df.to_csv(OUTPUT_RESULTS, index=False)
print(f"\nRésultats sauvegardés dans : {OUTPUT_RESULTS}")