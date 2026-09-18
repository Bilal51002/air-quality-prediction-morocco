from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="air_quality_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["air-quality", "machine-learning"],
) as dag:

    collect_data = BashOperator(
        task_id="collect_data",
        bash_command=(
            "cd /opt/airflow/project/scripts && "
            "python 01_collect_data.py"
        ),
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            "cd /opt/airflow/project/scripts && "
            "python 02_prepare_data.py"
        ),
    )

    build_model = BashOperator(
        task_id="build_model",
        bash_command=(
            "cd /opt/airflow/project/scripts && "
            "python 03_build_model.py"
        ),
    )

    federated_learning = BashOperator(
        task_id="federated_learning",
        bash_command=(
            "cd /opt/airflow/project/scripts && "
            "python 04_federated_learning.py"
        ),
    )

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

    collect_data >> prepare_data >> build_model >> federated_learning >> export_powerbi