from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

# --------------------------------------------------------------
# Deux scripts sont volontairement HORS de ce DAG :
#
# - 01_collect_data.py : boucle infinie (une collecte par heure, arret
#   manuel), pas une tache batch qui se termine.
#
# - 04_federated_learning.py : depend de flwr, qui entre en conflit
#   dur avec les dependances internes d'Airflow (SQLAlchemy 1.4 vs 2.x,
#   entre autres). L'image Airflow n'installe donc pas flwr.
#
# Les deux se lancent a la main, hors Airflow :
#   docker compose run air-quality python 01_collect_data.py
#   docker compose run air-quality python 04_federated_learning.py
# --------------------------------------------------------------

with DAG(
    dag_id="air_quality_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["air-quality", "machine-learning"],
) as dag:

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            "cd /opt/airflow/project/scripts && "
            "python 02_prepare_data.py"
        ),
    )

    # Independantes entre elles : toutes deux repartent des fichiers
    # produits par prepare_data, aucune ne lit le resultat de l'autre.
    build_model = BashOperator(
        task_id="build_model",
        bash_command=(
            "cd /opt/airflow/project/scripts && "
            "python 03_build_model.py"
        ),
    )

    weather_only_comparison = BashOperator(
        task_id="weather_only_comparison",
        bash_command=(
            "cd /opt/airflow/project/scripts && "
            "python 05_weather_only_comparison.py"
        ),
    )

    # Lit les resultats de build_model et weather_only_comparison. Lit
    # aussi results/federated_vs_centralized_results.csv s'il existe deja
    # (produit a la main via 04) mais ne bloque pas dessus s'il est absent.
    export_powerbi = BashOperator(
        task_id="export_powerbi",
        bash_command=(
            "cd /opt/airflow/project/scripts && "
            "python 06_export_powerbi.py"
        ),
    )

    # ============================================================
    # PIPELINE ORDER
    # ============================================================

    prepare_data >> [build_model, weather_only_comparison] >> export_powerbi
