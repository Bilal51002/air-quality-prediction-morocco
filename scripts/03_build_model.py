import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeRegressor


# ============================================================
# 1. Configuration MLflow
# ============================================================

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    # Docker / Airflow
    mlflow.set_tracking_uri(tracking_uri)
else:
    # GitHub Actions / execution locale sans serveur MLflow
    db_path = Path("../results/mlflow_ci.db").resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(
        f"sqlite:///{db_path}"
    )

mlflow.set_experiment(
    "Air Quality Prediction Morocco"
)


# ============================================================
# 2. Charger le dataset prêt
# ============================================================

INPUT_FILE = (
    "../data/morocco_air_quality_data_model_ready.csv"
)

OUTPUT_RESULTS = (
    "../results/step3_model_results.csv"
)

FEATURE_IMPORTANCE_FILE = (
    "../results/feature_importance.csv"
)

TARGET = "pm2_5"


df = pd.read_csv(INPUT_FILE)

print("Aperçu du dataset :")
print(df.head())

print(
    "\nTaille du dataset :",
    df.shape
)


# ============================================================
# 3. Split chronologique
# ============================================================
# Le split est déjà créé dans 02_prepare_data.py.
#
# On conserve le split chronologique afin d'éviter une fuite
# temporelle entre les observations proches dans le temps.
# ============================================================

train_df = df[
    df["split"] == "train"
]

test_df = df[
    df["split"] == "test"
]


X_train = train_df.drop(
    columns=[
        TARGET,
        "split"
    ]
)

y_train = train_df[
    TARGET
]


X_test = test_df.drop(
    columns=[
        TARGET,
        "split"
    ]
)

y_test = test_df[
    TARGET
]


print(
    "\nTaille X_train :",
    X_train.shape
)

print(
    "Taille X_test  :",
    X_test.shape
)

print(
    "Taille y_train :",
    y_train.shape
)

print(
    "Taille y_test  :",
    y_test.shape
)


# ============================================================
# 4. Définir les modèles
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree Regressor":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest Regressor":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
}


# ============================================================
# 5. Entraîner et évaluer les modèles
# ============================================================

results = []

trained_models = {}


for model_name, model in models.items():

    print(
        f"\n===== {model_name} ====="
    )

    # --------------------------------------------------------
    # Démarrer un run MLflow
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=model_name
    ):

        # ----------------------------------------------------
        # Paramètres généraux
        # ----------------------------------------------------

        mlflow.log_param(
            "model_name",
            model_name
        )

        mlflow.log_param(
            "target",
            TARGET
        )

        mlflow.log_param(
            "train_size",
            len(X_train)
        )

        mlflow.log_param(
            "test_size",
            len(X_test)
        )

        mlflow.log_param(
            "n_features",
            X_train.shape[1]
        )


        # ----------------------------------------------------
        # Hyperparamètres spécifiques
        # ----------------------------------------------------

        if (
            model_name
            == "Decision Tree Regressor"
        ):

            mlflow.log_param(
                "random_state",
                42
            )


        elif (
            model_name
            == "Random Forest Regressor"
        ):

            mlflow.log_param(
                "n_estimators",
                100
            )

            mlflow.log_param(
                "random_state",
                42
            )


        # ----------------------------------------------------
        # Entraînement
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )


        trained_models[
            model_name
        ] = model


        # ----------------------------------------------------
        # Prédictions
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )


        # ----------------------------------------------------
        # Métriques
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            y_pred
        )


        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                y_pred
            )
        )


        r2 = r2_score(
            y_test,
            y_pred
        )


        # ----------------------------------------------------
        # Affichage
        # ----------------------------------------------------

        print(
            "MAE  :",
            round(mae, 4)
        )

        print(
            "RMSE :",
            round(rmse, 4)
        )

        print(
            "R²   :",
            round(r2, 4)
        )


        # ----------------------------------------------------
        # Enregistrer les métriques dans MLflow
        # ----------------------------------------------------

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "RMSE",
            rmse
        )

        mlflow.log_metric(
            "R2",
            r2
        )


        # ----------------------------------------------------
        # Enregistrer le modèle dans MLflow
        # ----------------------------------------------------

        if model_name in [
            "Decision Tree Regressor",
            "Random Forest Regressor",
        ]:

            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

        else:

            mlflow.sklearn.log_model(
                model,
                name="model"
            )


        # ----------------------------------------------------
        # Ajouter les résultats au tableau final
        # ----------------------------------------------------

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })


# ============================================================
# 6. Tableau récapitulatif
# ============================================================

results_df = pd.DataFrame(
    results
)


print(
    "\n===== Résultats finaux "
    "(split chronologique) ====="
)


print(
    results_df.sort_values(
        by="RMSE"
    )
)


# ============================================================
# 7. Feature Importance - Random Forest
# ============================================================

rf = trained_models[
    "Random Forest Regressor"
]


importances = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
)


importances = importances.sort_values(
    ascending=False
)


print(
    "\n===== Top 10 features "
    "les plus importantes "
    "(Random Forest) ====="
)


print(
    importances.head(10)
)


# ============================================================
# 8. Sauvegarder Feature Importance
# ============================================================

Path(
    FEATURE_IMPORTANCE_FILE
).parent.mkdir(
    parents=True,
    exist_ok=True
)


importances.to_csv(
    FEATURE_IMPORTANCE_FILE,
    header=[
        "importance"
    ]
)


print(
    "\nFeature importance "
    "sauvegardée dans :",
    FEATURE_IMPORTANCE_FILE
)


# ============================================================
# 9. Sauvegarder les résultats
# ============================================================

Path(
    OUTPUT_RESULTS
).parent.mkdir(
    parents=True,
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)


print(
    "\nRésultats sauvegardés dans :",
    OUTPUT_RESULTS
)


# ============================================================
# 10. Fin
# ============================================================

print(
    "\n===== MLflow Tracking terminé ====="
)