import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================
# 1. Charger le dataset nettoyé
# =========================================
INPUT_FILE = "morocco_air_quality_data_clean.csv"
TARGET = "pm2_5"

df = pd.read_csv(INPUT_FILE)

print("Aperçu initial :")
print(df.head())
print("\nTaille du dataset :", df.shape)

# =========================================
# 2. Vérifier les colonnes nécessaires
# =========================================
required_cols = ["city", TARGET]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Colonne manquante : {col}")

# =========================================
# 3. Encodage des variables catégorielles
# =========================================
categorical_columns = ["city", "weather_main", "weather_description"]
existing_categorical = [col for col in categorical_columns if col in df.columns]

df_model = pd.get_dummies(
    df,
    columns=existing_categorical,
    drop_first=False
)

# =========================================
# 4. Définir les colonnes à retirer
# =========================================
drop_columns = ["timestamp", "timestamp_hour"]
existing_drop_columns = [col for col in drop_columns if col in df_model.columns]

# Important :
# On garde temporairement la colonne "city" originale à part,
# car elle sert à partitionner les clients fédérés.
city_series = df["city"].copy()

# Construire X et y
X = df_model.drop(columns=[TARGET] + existing_drop_columns, errors="ignore")
y = df_model[TARGET]

# Si la colonne city originale existe encore dans X, on la supprime
# car le modèle doit recevoir des colonnes numériques / encodées.
if "city" in X.columns:
    X = X.drop(columns=["city"])

print("\nShape de X :", X.shape)
print("Shape de y :", y.shape)

# =========================================
# 5. Train / Test split
# =========================================
X_train, X_test, y_train, y_test, city_train, city_test = train_test_split(
    X, y, city_series,
    test_size=0.2,
    random_state=42,
    stratify=city_series
)

print("\nTaille X_train :", X_train.shape)
print("Taille X_test :", X_test.shape)
print("Taille y_train :", y_train.shape)
print("Taille y_test :", y_test.shape)

print("\nRépartition des clients dans train :")
print(city_train.value_counts())

# =========================================
# 6. Baseline centralisée
# =========================================
central_model = LinearRegression()
central_model.fit(X_train, y_train)
y_pred_central = central_model.predict(X_test)

central_mae = mean_absolute_error(y_test, y_pred_central)
central_rmse = np.sqrt(mean_squared_error(y_test, y_pred_central))
central_r2 = r2_score(y_test, y_pred_central)

print("\n===== Modèle centralisé =====")
print("MAE  :", round(central_mae, 4))
print("RMSE :", round(central_rmse, 4))
print("R²   :", round(central_r2, 4))

# =========================================
# 7. Préparer les clients fédérés
# =========================================
clients = sorted(city_train.unique())
print("\nClients fédérés :", clients)

client_data = {}
for city in clients:
    idx = city_train[city_train == city].index
    X_city = X_train.loc[idx]
    y_city = y_train.loc[idx]

    client_data[city] = {
        "X": X_city,
        "y": y_city,
        "n_samples": len(X_city)
    }

# =========================================
# 8. Initialiser le modèle global fédéré
# =========================================
# Pour LinearRegression, les paramètres sont :
# - coef_
# - intercept_
#
# On initialise à zéro avant les rounds.
global_coef = np.zeros(X_train.shape[1])
global_intercept = 0.0

# Nombre de rounds fédérés
NUM_ROUNDS = 5

# =========================================
# 9. Boucle fédérée : entraînement local + FedAvg
# =========================================
for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n===== Round fédéré {round_num}/{NUM_ROUNDS} =====")

    local_coefs = []
    local_intercepts = []
    local_sizes = []

    for city in clients:
        X_city = client_data[city]["X"]
        y_city = client_data[city]["y"]
        n_city = client_data[city]["n_samples"]

        # En pratique, sklearn LinearRegression ne permet pas
        # d’initialiser directement avec un coef global avant fit.
        # Donc ici on simule le round fédéré pédagogique :
        # chaque client entraîne un modèle local sur ses données,
        # puis on agrège les paramètres obtenus.
        local_model = LinearRegression()
        local_model.fit(X_city, y_city)

        local_coefs.append(local_model.coef_)
        local_intercepts.append(local_model.intercept_)
        local_sizes.append(n_city)

        print(f"{city}: {n_city} échantillons")

    # =====================================
    # FedAvg : moyenne pondérée
    # =====================================
    total_samples = sum(local_sizes)

    weighted_coef_sum = np.zeros_like(global_coef, dtype=float)
    weighted_intercept_sum = 0.0

    for coef, intercept, n_samples in zip(local_coefs, local_intercepts, local_sizes):
        weight = n_samples / total_samples
        weighted_coef_sum += weight * coef
        weighted_intercept_sum += weight * intercept

    global_coef = weighted_coef_sum
    global_intercept = weighted_intercept_sum

    print("Agrégation FedAvg terminée.")

# =========================================
# 10. Prédiction avec le modèle global fédéré
# =========================================
# Comme on a agrégé manuellement coef/intercept,
# on calcule les prédictions à la main :
#
# y_pred = X @ coef + intercept
#
# On convertit X_test en numpy array.
X_test_array = X_test.to_numpy()
y_pred_federated = X_test_array @ global_coef + global_intercept

federated_mae = mean_absolute_error(y_test, y_pred_federated)
federated_rmse = np.sqrt(mean_squared_error(y_test, y_pred_federated))
federated_r2 = r2_score(y_test, y_pred_federated)

print("\n===== Modèle fédéré final =====")
print("MAE  :", round(federated_mae, 4))
print("RMSE :", round(federated_rmse, 4))
print("R²   :", round(federated_r2, 4))

# =========================================
# 11. Comparaison centralisé vs fédéré
# =========================================
results = pd.DataFrame([
    {
        "Approach": "Centralized Linear Regression",
        "MAE": central_mae,
        "RMSE": central_rmse,
        "R2": central_r2
    },
    {
        "Approach": "Federated Linear Regression (FedAvg)",
        "MAE": federated_mae,
        "RMSE": federated_rmse,
        "R2": federated_r2
    }
])

print("\n===== Comparaison finale =====")
print(results)

# =========================================
# 12. Sauvegarder les résultats
# =========================================
results.to_csv("federated_vs_centralized_results.csv", index=False)
print("\nRésultats sauvegardés dans : federated_vs_centralized_results.csv")