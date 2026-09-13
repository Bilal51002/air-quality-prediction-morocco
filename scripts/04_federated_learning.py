"""
04_federated_learning.py

Simulation de federated learning avec le framework Flower (flwr), en
partitionnant les donnees par ville (chaque ville = un client federe),
conformement au brief. Remplace la simulation manuelle precedente qui
moyennait des modeles OLS deja converges independamment (d'ou le R2
negatif obtenu avant).

Ici, chaque client entraine un SGDRegressor en partant des poids globaux
recus du serveur (quelques pas de descente de gradient locaux), puis le
serveur agrege via FedAvg (moyenne ponderee par nombre d'echantillons,
geree nativement par la strategie FedAvg de Flower).
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

# =========================================================
# 0. CONFIGURATION
# =========================================================

INPUT_FILE = "../data/morocco_air_quality_data_clean.csv"
OUTPUT_RESULTS = "../results/federated_vs_centralized_results.csv"
TARGET = "pm2_5"

NUM_ROUNDS = 15       # rounds federes
LOCAL_EPOCHS = 5      # passes locales (partial_fit) par client et par round

# =========================================================
# 1. Charger le dataset nettoye (issu de 02_prepare_data.py, avec
#    split chronologique deja fait et colonne "city" encore en clair)
# =========================================================

df = pd.read_csv(INPUT_FILE)
print("Taille du dataset :", df.shape)

for col in ["city", TARGET, "split"]:
    if col not in df.columns:
        raise ValueError(f"Colonne manquante : {col}")

df_model = pd.get_dummies(df, columns=["weather_description"], drop_first=True)

feature_cols = [
    c for c in df_model.columns
    if c not in [TARGET, "split", "city", "timestamp", "timestamp_hour", "source_file"]
]

train_df = df_model[df_model["split"] == "train"]
test_df = df_model[df_model["split"] == "test"]

city_train = df.loc[train_df.index, "city"]

# Normalisation : scaler ajuste sur le train uniquement (meme principe
# que dans 02_prepare_data.py, pour ne pas fuiter d'info du test).
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(train_df[feature_cols]),
    columns=feature_cols, index=train_df.index
)
X_test = pd.DataFrame(
    scaler.transform(test_df[feature_cols]),
    columns=feature_cols, index=test_df.index
)
y_train = train_df[TARGET]
y_test = test_df[TARGET]

n_features = X_train.shape[1]
print(f"Nombre de features : {n_features}")

# =========================================================
# 2. Baseline centralisee (toutes les villes melangees, un seul modele)
# =========================================================

central_model = LinearRegression()
central_model.fit(X_train, y_train)
y_pred_central = central_model.predict(X_test)

central_mae = mean_absolute_error(y_test, y_pred_central)
central_rmse = np.sqrt(mean_squared_error(y_test, y_pred_central))
central_r2 = r2_score(y_test, y_pred_central)

print("\n===== Modele centralise (LinearRegression) =====")
print("MAE  :", round(central_mae, 4))
print("RMSE :", round(central_rmse, 4))
print("R²   :", round(central_r2, 4))

# =========================================================
# 3. Preparer les clients federes (un par ville)
# =========================================================

clients_cities = sorted(city_train.unique())
print("\nClients federes :", clients_cities)

client_data = {}
for i, city in enumerate(clients_cities):
    idx = city_train[city_train == city].index
    client_data[str(i)] = (
        X_train.loc[idx].to_numpy(dtype=np.float64),
        y_train.loc[idx].to_numpy(dtype=np.float64),
    )
    print(f"  {city}: {len(idx)} echantillons")


def get_model_params(model):
    return [model.coef_, model.intercept_]


def set_model_params(model, params):
    model.coef_ = params[0]
    model.intercept_ = params[1]


def make_model():
    """SGDRegressor initialise a zero, pret a recevoir des poids
    globaux et a continuer l'entrainement via partial_fit (warm start).
    eta0 volontairement bas : avec un taux d'apprentissage par defaut
    (0.01), les poids divergent au bout de quelques rounds a cause de
    l'effet cumulatif de la moyenne FedAvg sur plusieurs clients."""
    model = SGDRegressor(
        max_iter=1, warm_start=True, random_state=42,
        learning_rate="constant", eta0=0.0005, alpha=0.0001,
    )
    model.coef_ = np.zeros(n_features)
    model.intercept_ = np.zeros(1)
    return model


# =========================================================
# 4. Client Flower
# =========================================================

class CityClient(NumPyClient):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = make_model()

    def get_parameters(self, config):
        return get_model_params(self.model)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        for _ in range(LOCAL_EPOCHS):
            self.model.partial_fit(self.X, self.y)
        return get_model_params(self.model), len(self.X), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        preds = self.model.predict(self.X)
        loss = mean_squared_error(self.y, preds)
        return float(loss), len(self.X), {"mae": float(mean_absolute_error(self.y, preds))}


def client_fn(context: Context):
    cid = context.node_config["partition-id"]
    X, y = client_data[str(cid)]
    return CityClient(X, y).to_client()


client_app = ClientApp(client_fn=client_fn)

# =========================================================
# 5. Serveur Flower : strategie FedAvg + evaluation centralisee
#    sur le vrai test set apres chaque round
# =========================================================

history = []


def evaluate_fn(server_round, parameters, config):
    model = make_model()
    set_model_params(model, parameters)
    preds = model.predict(X_test.to_numpy())

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"[Round {server_round}] test -> MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    history.append({"round": server_round, "mae": mae, "rmse": rmse, "r2": r2})

    return float(rmse), {"mae": mae, "r2": r2}


def server_fn(context: Context):
    initial_parameters = ndarrays_to_parameters(get_model_params(make_model()))

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,   # on evalue cote serveur via evaluate_fn (test set global)
        min_fit_clients=len(clients_cities),
        min_available_clients=len(clients_cities),
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=NUM_ROUNDS))


server_app = ServerApp(server_fn=server_fn)

# =========================================================
# 6. Lancer la simulation federee
# =========================================================

print(f"\n===== Simulation federee : {NUM_ROUNDS} rounds, {len(clients_cities)} clients =====\n")

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=len(clients_cities),
    backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0}},
)

federated_final = history[-1]

print("\n===== Modele federe final (FedAvg, Flower) =====")
print("MAE  :", round(federated_final["mae"], 4))
print("RMSE :", round(federated_final["rmse"], 4))
print("R²   :", round(federated_final["r2"], 4))

# =========================================================
# 7. Comparaison centralise vs federe
# =========================================================

results = pd.DataFrame([
    {"Approach": "Centralized Linear Regression", "MAE": central_mae, "RMSE": central_rmse, "R2": central_r2},
    {"Approach": "Federated Learning (Flower, FedAvg)", "MAE": federated_final["mae"],
     "RMSE": federated_final["rmse"], "R2": federated_final["r2"]},
])

print("\n===== Comparaison finale =====")
print(results)

results.to_csv(OUTPUT_RESULTS, index=False)
print(f"\nResultats sauvegardes dans : {OUTPUT_RESULTS}")

pd.DataFrame(history).to_csv("../results/federated_learning_curve.csv", index=False)
print("Courbe d'apprentissage federee sauvegardee dans : ../results/federated_learning_curve.csv")