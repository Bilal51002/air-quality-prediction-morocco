FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installation des dépendances Python du projet
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copie des scripts et des données
COPY scripts/ ./scripts/
COPY data/raw/ ./data/raw/

# Création des dossiers utilisés par le pipeline
RUN mkdir -p /app/data /app/results

WORKDIR /app/scripts

# Le pipeline sera maintenant orchestré par Airflow
# Le conteneur reste actif en attendant les tâches Airflow
CMD ["tail", "-f", "/dev/null"]