import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. Charger le dataset prêt
# =========================
INPUT_FILE = "morocco_air_quality_data_model_ready.csv"
TARGET = "pm2_5"

df = pd.read_csv(INPUT_FILE)

print("Aperçu du dataset :")
print(df.head())

print("\nTaille du dataset :", df.shape)

# =========================
# 2. Séparer X et y
# =========================
X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nShape de X :", X.shape)
print("Shape de y :", y.shape)

# =========================
# 3. Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTaille X_train :", X_train.shape)
print("Taille X_test :", X_test.shape)
print("Taille y_train :", y_train.shape)
print("Taille y_test :", y_test.shape)

# =========================
# 4. Définir les modèles
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
# 5. Entraîner et évaluer
# =========================
results = []

for model_name, model in models.items():
    print(f"\n===== {model_name} =====")
    
    # Entraînement
    model.fit(X_train, y_train)
    
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
# 6. Tableau récapitulatif
# =========================
results_df = pd.DataFrame(results)

print("\n===== Résultats finaux =====")
print(results_df.sort_values(by="RMSE"))

# =========================
# 7. Sauvegarder les résultats
# =========================
results_df.to_csv("step3_model_results.csv", index=False)
print("\nRésultats sauvegardés dans : step3_model_results.csv")