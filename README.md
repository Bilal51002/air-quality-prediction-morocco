# Prédiction de la qualité de l'air dans les villes marocaines

Projet de machine learning prédisant la concentration de PM2.5 dans 7 villes
marocaines à partir de données météo et de pollution collectées en temps
réel via l'API OpenWeatherMap, avec une exploration du **federated
learning** (Flower) comme alternative à l'apprentissage centralisé.

## Objectif

Prédire la concentration de PM2.5 (particules fines, µg/m³) à partir de
variables météorologiques (température, humidité, pression, vent) et de
polluants complémentaires (AQI, NO2, O3, PM10), en conditions réelles
(données collectées sur ~1 mois, pas simulées).

## Résultats

### Modèles centralisés (split chronologique 80/20)

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest** | **0.73** | **1.13** | **0.632** |
| Linear Regression | 1.02 | 1.22 | 0.569 |
| Decision Tree | 0.97 | 1.47 | 0.375 |

### Federated Learning (Flower, FedAvg, 7 clients = 7 villes)

| Approche | MAE | RMSE | R² |
|---|---|---|---|
| Centralisé (LinearRegression) | 0.93 | 1.13 | 0.630 |
| Centralisé (SGDRegressor, même config que le fédéré) | 0.73 | 0.98 | **0.724** |
| Fédéré (SGDRegressor, FedAvg) | 0.79 | 1.08 | 0.663 |

En comparant des modèles identiques (SGDRegressor des deux côtés), le
centralisé surpasse légèrement le fédéré — c'est le compromis attendu du
federated learning : une performance un peu en retrait contre la
préservation de la confidentialité des données par ville (aucune donnée
brute ne quitte son client).

### Importance des features (Random Forest)

`pm10` domine à lui seul l'importance (87%) : le modèle apprend surtout la
relation physique PM2.5 ↔ PM10 (deux mesures de particules très corrélées),
plus qu'un vrai lien météo → pollution. Les variables météo (température,
humidité, pression, vent) contribuent ensemble à moins de 5% de
l'importance totale — une limite honnête à garder en tête plutôt qu'à
cacher.

## Structure du projet

```
Project/
├── data/
│   ├── raw/                                    # 3 fichiers de collecte bruts (3 membres d'equipe)
│   ├── morocco_air_quality_data_merged.csv     # genere par 02 : fusion des 3 sources, 7 villes
│   ├── morocco_air_quality_data_clean.csv      # genere par 02 : nettoye + split chronologique
│   └── morocco_air_quality_data_model_ready.csv# genere par 02 : encode + normalise (sans fuite)
├── scripts/
│   ├── 00_merge_sources.py       # fusion autonome des 3 sources (reference / debug)
│   ├── 01_collect_data.py        # collecte API OpenWeatherMap (7 villes)
│   ├── 02_prepare_data.py        # fusion auto + nettoyage + split + normalisation
│   ├── 03_build_model.py         # Linear / Decision Tree / Random Forest
│   └── 04_federated_learning.py  # FedAvg via Flower (SGDRegressor, 7 clients)
├── results/
│   ├── step3_model_results.csv
│   ├── federated_vs_centralized_results.csv
│   ├── federated_learning_curve.csv
│   └── feature_importance.csv
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env                            # OPENWEATHER_API_KEY (non versionne)
└── README.md
```

## Lancer le projet

### Option A — avec Docker (recommandé, evite tout conflit de dependances)

```bash
docker compose build
docker compose run air-quality                              # pipeline complet
docker compose run air-quality python 04_federated_learning.py  # un seul script
```

### Option B — en local (environnement virtuel recommande)

```bash
python -m venv venv
# Windows :
.\venv\Scripts\activate
# Mac/Linux :
source venv/bin/activate

pip install -r requirements.txt

cd scripts
python 02_prepare_data.py
python 03_build_model.py
python 04_federated_learning.py
```

### Collecter de nouvelles données (optionnel)

Nécessite une clé API OpenWeatherMap dans un fichier `.env` à la racine :
```
OPENWEATHER_API_KEY=ta_cle_ici
```
Puis :
```bash
python scripts/01_collect_data.py
```
Le script tourne en continu (une collecte par heure, 7 villes). Interrompre
avec `Ctrl+C` quand assez de données ont été collectées, puis relancer
`02_prepare_data.py` pour régénérer les fichiers préparés.

## Méthodologie et choix techniques

### Données
- **3 sources fusionnées automatiquement** : chaque membre de l'équipe a
  collecté séparément (villes différentes), le script `02_prepare_data.py`
  détecte et fusionne tous les CSV présents dans `data/raw/` sans
  modification de code nécessaire si un nouveau fichier est ajouté.
- **7 villes** : Casablanca, Rabat, Marrakech, Fès, Agadir, Tanger, Karia Ba
  Mohamed.
- **2099 lignes** au total après fusion et dédoublonnage, couvrant du
  26 avril au 23 mai 2026.
- Colonnes gardées uniquement si présentes dans au moins 80% des lignes
  (`co`, `so2`, `nh3`, `feels_like`, `wind_deg`, `visibility` exclues car
  collectées par une seule des 3 sources).

### Éviter la fuite de données (data leakage)
- **Split chronologique** (80% le plus ancien = train, 20% le plus récent =
  test) plutôt qu'un split aléatoire, pour éviter que des mesures horaires
  quasi identiques se retrouvent des deux côtés du split (la pollution est
  fortement autocorrélée dans le temps).
- **Normalisation (`StandardScaler`) ajustée uniquement sur le train**, puis
  appliquée au test — le test set n'influence jamais les statistiques de
  normalisation.

### Federated Learning
- Implémenté avec le framework **Flower**, pas une simulation manuelle :
  chaque ville reçoit les poids globaux, entraîne localement un
  `SGDRegressor` sur ses données (jamais partagées), renvoie ses poids ; le
  serveur agrège via `FedAvg` (moyenne pondérée par nombre d'échantillons,
  gérée nativement par Flower).
- Taux d'apprentissage volontairement bas (`eta0=0.0005`) : avec la valeur
  par défaut de scikit-learn, les poids divergent après quelques rounds à
  cause de l'effet cumulatif de la moyenne FedAvg sur 7 clients.

## Limites connues

- **`pm10` domine la prédiction** (87% d'importance) — le modèle capture
  surtout une relation physique entre deux mesures de particules, pas
  principalement le lien météo → pollution annoncé dans l'objectif initial.
- **Période de collecte courte** (~1 mois, printemps uniquement) : aucune
  variabilité saisonnière capturée, les résultats ne généralisent pas
  forcément à l'année entière.
- **Collecte discontinue** : un trou de ~5,6 jours sans données a été
  identifié (la machine de collecte s'est probablement éteinte), réduisant
  la couverture réelle à 23 des 27 jours de la période.
- **Valeurs de PM2.5 relativement basses** (moyenne 4,5 µg/m³) comparées aux
  moyennes généralement rapportées pour les villes marocaines — à vérifier
  si l'usage se prolonge (unités, position des capteurs modélisés par
  OpenWeatherMap plutôt que mesurés au sol).
- **Agadir absent du test set** : sa fenêtre de collecte (30 avril–8 mai) se
  termine avant le début de la période de test (12 mai), donc le modèle
  n'est jamais évalué spécifiquement sur cette ville.

## Stack technique

Python 3.11 · pandas · scikit-learn · Flower (flwr) · Docker
