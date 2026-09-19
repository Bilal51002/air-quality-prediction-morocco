# Image Airflow avec un SOUS-ENSEMBLE des dependances du projet.
#
# flwr (Flower, federated learning) est volontairement EXCLU ici : la
# version actuelle de flwr exige SQLAlchemy >=2.0, protobuf >=5.28,
# grpcio >=1.70 et cryptography >=46 — Airflow 3.0.6 est construit en
# interne sur SQLAlchemy 1.4.x et des versions figees de ces memes
# librairies. Les deux ne peuvent pas coexister dans un seul environnement
# Python : ce n'est pas un probleme de contrainte pip, c'est une
# incompatibilite reelle entre les deux frameworks.
#
# Consequence : 04_federated_learning.py n'est PAS orchestre par Airflow
# (comme 01_collect_data.py). Lance-le a la main :
#   docker compose run air-quality python 04_federated_learning.py
#
# (Pour une isolation complete en vraie prod, on ferait tourner cette
# tache dans son propre conteneur via DockerOperator/KubernetesPodOperator
# plutot que de l'exclure — plus robuste, mais demande de monter des
# chemins hote absolus, plus complexe a mettre au point sous Windows.)
FROM apache/airflow:3.0.6-python3.11

USER airflow

COPY requirements-airflow.txt /requirements-airflow.txt

RUN pip install --no-cache-dir -r /requirements-airflow.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.11.txt"
