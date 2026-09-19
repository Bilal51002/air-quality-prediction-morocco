# Dashboard Power BI — Air Quality Prediction Morocco

Couche de restitution du projet : les CSV du modèle en étoile sont générés par
`scripts/06_export_powerbi.py`, puis assemblés dans Power BI Desktop.

## 1. Générer les données

```bash
cd scripts
python 02_prepare_data.py      # si data/ n'est pas déjà à jour
python 03_build_model.py
python 06_export_powerbi.py
```

Le dossier `powerbi/` est créé avec six fichiers :

| Fichier | Grain | Rôle |
|---|---|---|
| `dim_city.csv` | 1 ligne / ville | Dimension géographique (lat/lon pour la carte) |
| `dim_date.csv` | 1 ligne / jour | Table de dates continue |
| `fact_measurements.csv` | 1 ligne / ville / heure | Mesures observées |
| `fact_predictions.csv` | 1 ligne / point de test / modèle | Réel vs prédit |
| `model_metrics.csv` | 1 ligne / modèle / expérience | MAE, RMSE, R² |
| `feature_importance.csv` | 1 ligne / variable | Importance Random Forest |

## 2. Charger et modéliser

**Obtenir des données → Dossier →** sélectionner `powerbi/`, puis **Combiner et
transformer**. Vérifier dans Power Query que `timestamp` et `date` sont typés
*Date/heure* et *Date*, et que les colonnes numériques n'ont pas été lues en texte
(le séparateur décimal est le point).

### Relations à créer (vue Modèle)

| De | Vers | Cardinalité | Direction |
|---|---|---|---|
| `fact_measurements[city]` | `dim_city[city]` | plusieurs à un | simple |
| `fact_measurements[date]` | `dim_date[date]` | plusieurs à un | simple |
| `fact_predictions[city]` | `dim_city[city]` | plusieurs à un | simple |
| `fact_predictions[date]` | `dim_date[date]` | plusieurs à un | simple |

Puis **Marquer comme table de dates** sur `dim_date` (colonne `date`). Sans ça,
les hiérarchies temporelles et les fonctions `DATEADD` / `SAMEPERIODLASTYEAR` ne
fonctionnent pas correctement.

### Tri des colonnes catégorielles

Sélectionner `fact_measurements[pm25_categorie]` → **Trier par colonne** →
`pm25_ordre`. Idem pour `aqi_categorie` avec `aqi_ordre`. Sinon les catégories
s'affichent par ordre alphabétique, ce qui donne « Bon, Mauvais, Modéré… ».

## 3. Mesures DAX

Créer une table vide nommée `_Mesures` (**Entrer des données** → table sans
colonne) et y placer toutes les mesures, pour ne pas les disperser.

### Qualité de l'air

```dax
PM2.5 moyen = AVERAGE(fact_measurements[pm2_5])

PM2.5 max = MAX(fact_measurements[pm2_5])

PM10 moyen = AVERAGE(fact_measurements[pm10])

Nb mesures = COUNTROWS(fact_measurements)

Villes suivies = DISTINCTCOUNT(fact_measurements[city])

Taux dépassement OMS =
DIVIDE(
    CALCULATE([Nb mesures], fact_measurements[depasse_seuil_oms] = TRUE()),
    [Nb mesures]
)

PM2.5 moyen 24h glissantes =
CALCULATE(
    [PM2.5 moyen],
    DATESINPERIOD(dim_date[date], MAX(dim_date[date]), -1, DAY)
)

Écart vs moyenne nationale =
[PM2.5 moyen] - CALCULATE([PM2.5 moyen], ALL(dim_city))
```

### Performance des modèles

```dax
MAE = AVERAGE(fact_predictions[erreur_absolue])

RMSE = SQRT(AVERAGE(fact_predictions[erreur_carree]))

Biais moyen = AVERAGE(fact_predictions[erreur])     -- > 0 : le modèle surestime

R2 =
VAR MoyenneReelle = CALCULATE(AVERAGE(fact_predictions[pm25_reel]), ALLSELECTED(fact_predictions))
VAR SSres = SUMX(fact_predictions, fact_predictions[erreur] ^ 2)
VAR SStot = SUMX(fact_predictions, (fact_predictions[pm25_reel] - MoyenneReelle) ^ 2)
RETURN DIVIDE(SStot - SSres, SStot)

Erreur relative moyenne =
DIVIDE(
    AVERAGE(fact_predictions[erreur_absolue]),
    AVERAGE(fact_predictions[pm25_reel])
)

Meilleur modèle (RMSE) =
VAR Classement =
    ADDCOLUMNS(
        VALUES(fact_predictions[modele]),
        "@RMSE", CALCULATE([RMSE])
    )
RETURN MINX(TOPN(1, Classement, [@RMSE], ASC), fact_predictions[modele])
```

### Mise en forme conditionnelle

```dax
Couleur PM2.5 =
SWITCH(
    TRUE(),
    [PM2.5 moyen] <= 12,   "#2E933C",
    [PM2.5 moyen] <= 35.4, "#E8B14A",
    [PM2.5 moyen] <= 55.4, "#E07B39",
    "#C0392B"
)
```

À brancher via **Format → Couleur → fx → Format par : valeur du champ**.

## 4. Les quatre pages

### Page 1 — Vue d'ensemble

- Bandeau de cartes KPI : `PM2.5 moyen`, `PM2.5 max`, `Villes suivies`,
  `Nb mesures`, `Taux dépassement OMS`.
- **Carte (Map)** : `dim_city[lat]` / `dim_city[lon]` en latitude/longitude,
  taille des bulles = `PM2.5 moyen`, couleur = `Couleur PM2.5`. C'est la vue qui
  justifie à elle seule le dashboard — les 7 villes sont géographiquement
  dispersées.
- **Courbe** : `PM2.5 moyen` par `dim_date[date]`, une ligne par ville. Le trou
  de collecte de ~5,6 jours apparaîtra comme une interruption : c'est voulu, ne
  pas le combler par une interpolation.
- **Histogramme empilé 100 %** : répartition des `pm25_categorie` par ville.
- Segments : ville, plage de dates, `split` (train/test).

### Page 2 — Analyse temporelle

- **Matrice** heure × jour de la semaine, valeurs = `PM2.5 moyen`, mise en forme
  conditionnelle en dégradé. C'est le meilleur moyen de faire ressortir le profil
  journalier (pics du matin et du soir liés au trafic, s'il y en a).
- **Barres** : `PM2.5 moyen` par `tranche_horaire`.
- **Nuage de points** : `temperature` en X, `pm2_5` en Y, taille = `humidity`,
  légende = ville. Ce visuel illustre visuellement le résultat central du
  projet : le nuage est diffus, la météo seule n'explique presque rien.
- **Jauge** : `PM2.5 moyen` avec valeur cible 15 (ligne directrice OMS).

### Page 3 — Performance des modèles

- Cartes KPI : `MAE`, `RMSE`, `R2`, `Biais moyen`, filtrées par un segment
  `fact_predictions[modele]`.
- **Réel vs prédit dans le temps** : deux lignes (`pm25_reel`, `pm25_predit`) sur
  l'axe `timestamp`, filtrées sur un seul modèle.
- **Nuage réel vs prédit** : `pm25_reel` en X, `pm25_predit` en Y. Ajouter une
  ligne de référence y = x via **Analyses → Ligne de tendance** pour voir
  l'écart à la prédiction parfaite.
- **Barres horizontales** : `MAE` par ville. Cette vue montre immédiatement que
  Agadir est absente du test set — s'appuyer sur `dim_city[present_dans_test]`
  pour l'expliquer dans une zone de texte plutôt que de laisser un vide.
- **Barres groupées** : `model_metrics` — RMSE par modèle et par `experience`
  (centralisé, fédéré, météo seule).

### Page 4 — Interprétabilité et limites

- **Barres horizontales** : `importance_pct` par variable, top 10. `pm10` va
  écraser le reste (~87 %) : c'est le message, pas un défaut de visuel.
- **Barres** : R² par configuration de variables (complet vs météo seule). Les R²
  négatifs doivent rester visibles — un axe forcé à démarrer à 0 masquerait le
  résultat le plus intéressant du projet.
- **Courbe** : `federated_learning_curve.csv`, perte par round FedAvg.
- Zone de texte reprenant les limites connues (période courte, trou de collecte,
  valeurs PM2.5 basses, Agadir hors test).

## 5. Actualisation

Les CSV sont regénérés à chaque exécution de `06_export_powerbi.py`. Dans Power
BI Desktop, **Actualiser** relit le dossier — aucune reconfiguration n'est
nécessaire tant que les noms de fichiers et de colonnes ne changent pas.

Pour une actualisation automatique côté Power BI Service, il faut publier le
rapport et installer une passerelle de données locale pointant vers le dossier,
ou basculer la source vers un stockage cloud (SharePoint, OneDrive, Blob Azure).
Une alternative plus propre à terme : faire écrire `06_export_powerbi.py` dans
une base (PostgreSQL, SQLite) et connecter Power BI dessus via DirectQuery.

## 6. À ajouter au README principal

```markdown
## Dashboard Power BI

Un modèle en étoile prêt pour Power BI est généré par
`scripts/06_export_powerbi.py` (dimensions ville/date, faits mesures et
prédictions, métriques et importance des variables). Voir [POWERBI.md](POWERBI.md)
pour le détail des relations, des mesures DAX et de la construction des pages.
```

Penser à ajouter `powerbi/*.csv` au `.gitignore` si tu ne veux pas versionner des
données regénérables — ou au contraire à les committer pour qu'un relecteur
puisse ouvrir le rapport sans exécuter le pipeline.
