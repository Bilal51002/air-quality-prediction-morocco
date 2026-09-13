# Image de base legere avec Python 3.11 (coherente avec ta version locale)
FROM python:3.11-slim

# Empeche pip/python de generer des .pyc et force les logs a s'afficher direct
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# On copie d'abord uniquement requirements.txt pour profiter du cache Docker :
# tant que ce fichier ne change pas, "pip install" n'est pas relance a chaque
# build (gain de temps considerable pendant le developpement).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Puis on copie le reste du projet
COPY scripts/ ./scripts/
COPY data/raw/ ./data/raw/

# Les resultats et les fichiers generes dans data/ seront ecrits dans des
# volumes montes au lancement (voir docker-compose.yml), pour qu'ils
# persistent sur la machine hote et ne restent pas coinces dans le conteneur.

WORKDIR /app/scripts

# Commande par defaut : execute tout le pipeline dans l'ordre.
# 02_prepare_data.py fait deja la fusion automatique des fichiers de
# data/raw/ en interne (00_merge_sources.py n'est plus necessaire dans
# ce pipeline, garde a titre de reference si besoin de la fusion seule).
# Peut etre surchargee, ex: docker run <image> python 03_build_model.py
CMD ["sh", "-c", "python 02_prepare_data.py && python 03_build_model.py && python 04_federated_learning.py"]
