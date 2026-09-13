"""
02_prepare_data.py

- Fusionne AUTOMATIQUEMENT tous les fichiers CSV presents dans data/raw/
  (peu importe leur nombre : 3 aujourd'hui, 4 demain si quelqu'un rajoute
  un fichier, sans avoir a modifier ce script).
- Harmonise les schemas differents (pm25 -> pm2_5, weather_desc -> ...).
- Nettoie, encode, split train/test de facon CHRONOLOGIQUE (pas aleatoire),
  puis normalise SANS fuite de donnees (scaler ajuste sur le train
  uniquement).
- Reprend le meme dict CITIES que 01_collect_data.py pour rester coherent
  (memes villes, memes coordonnees) quand une source ne fournit pas
  lat/lon.
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================================================
# 0. CONFIGURATION
# =========================================================

RAW_DIR = "../data/raw"
MERGED_FILE = "../data/morocco_air_quality_data_merged.csv"
CLEAN_FILE = "../data/morocco_air_quality_data_clean.csv"
MODEL_READY_FILE = "../data/morocco_air_quality_data_model_ready.csv"

TARGET = "pm2_5"
TEST_SIZE = 0.2          # 20% le plus RECENT sert de test (split chronologique)
MAX_MISSING_RATIO = 0.20  # colonnes avec >20% de manquants -> exclues

# Meme dict que 01_collect_data.py : sert a completer lat/lon quand une
# source de donnees ne les fournit pas.
CITIES = {
    "Casablanca":       {"lat": 33.5731, "lon": -7.5898},
    "Rabat":             {"lat": 34.0209, "lon": -6.8416},
    "Marrakech":         {"lat": 31.6295, "lon": -7.9811},
    "Fes":               {"lat": 34.0331, "lon": -5.0003},
    "Agadir":            {"lat": 30.4278, "lon": -9.5981},
    "Tanger":            {"lat": 35.7595, "lon": -5.8340},
    "Karia Ba Mohamed":  {"lat": 34.3667, "lon": -5.2139},
}

# Colonnes alternatives rencontrees selon la source qui a collecte le fichier
RENAME_MAP = {
    "pm25": "pm2_5",
    "weather_desc": "weather_description",
}

COMMON_SCHEMA = [
    "timestamp", "city", "lat", "lon",
    "temperature", "feels_like", "humidity", "pressure",
    "wind_speed", "wind_deg", "clouds", "visibility",
    "weather_main", "weather_description",
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
]

# =========================================================
# 1. FUSION AUTOMATIQUE DE TOUS LES FICHIERS DE data/raw/
# =========================================================

def load_and_harmonize(filepath):
    """Lit un CSV brut et le ramene au schema commun, quelle que soit
    la source qui l'a produit."""
    df = pd.read_csv(filepath)
    df = df.rename(columns=RENAME_MAP)

    # Timestamp converti ICI, fichier par fichier : si on le fait apres
    # avoir concatene plusieurs fichiers a formats differents (avec ou
    # sans microsecondes par ex.), pandas deduit un seul format a partir
    # de la premiere ligne et transforme en NaT toutes les lignes qui ne
    # matchent pas -> perte silencieuse de donnees.
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Complete lat/lon a partir du nom de ville si absents de la source
    if "lat" not in df.columns or "lon" not in df.columns or df.get("lat", pd.Series(dtype=float)).isna().all():
        df["lat"] = df["city"].map(lambda c: CITIES.get(c, {}).get("lat"))
        df["lon"] = df["city"].map(lambda c: CITIES.get(c, {}).get("lon"))

    df["source_file"] = os.path.basename(filepath)

    for col in COMMON_SCHEMA:
        if col not in df.columns:
            df[col] = np.nan

    return df[COMMON_SCHEMA + ["source_file"]]


raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
print(f"Fichiers bruts detectes dans {RAW_DIR}/ : {len(raw_files)}")
for f in raw_files:
    print("  -", os.path.basename(f))

if not raw_files:
    raise FileNotFoundError(f"Aucun fichier CSV trouve dans {RAW_DIR}/")

merged = pd.concat([load_and_harmonize(f) for f in raw_files], ignore_index=True)

numeric_cols = [
    "lat", "lon", "temperature", "feels_like", "humidity", "pressure",
    "wind_speed", "wind_deg", "clouds", "visibility",
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
]
for col in numeric_cols:
    merged[col] = pd.to_numeric(merged[col], errors="coerce")

before = len(merged)
merged = merged.drop_duplicates()
merged = merged.drop_duplicates(subset=["city", "timestamp"], keep="first")
print(f"\nLignes apres fusion : {len(merged)} (doublons retires : {before - len(merged)})")
print("Villes :", merged["city"].value_counts().to_dict())

merged.to_csv(MERGED_FILE, index=False)
print(f"Dataset fusionne sauvegarde : {MERGED_FILE}")

# =========================================================
# 2. NETTOYAGE
# =========================================================

df = merged.copy()

# Colonnes trop creuses (presentes dans une seule source sur plusieurs) :
# les remplir par la mediane inventerait une grande partie des donnees.
# On les exclut plutot que de les imputer.
missing_ratio = df.isna().mean()
sparse_cols = missing_ratio[missing_ratio > MAX_MISSING_RATIO].index.tolist()
sparse_cols = [c for c in sparse_cols if c != TARGET]

print(f"\nColonnes exclues (>{int(MAX_MISSING_RATIO*100)}% de manquants) : {sparse_cols}")
df = df.drop(columns=sparse_cols)

# Lignes sans cible ou sans les features essentielles restantes
essential = [c for c in ["temperature", "humidity", "pressure", TARGET] if c in df.columns]
df = df.dropna(subset=essential)

# Pour les colonnes numeriques restantes, mediane sur les rares NaN residuels
remaining_numeric = [c for c in numeric_cols if c in df.columns]
for col in remaining_numeric:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

categorical_columns = [c for c in ["city", "weather_main", "weather_description"] if c in df.columns]
for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

print("\nValeurs manquantes apres nettoyage :")
print(df.isna().sum()[df.isna().sum() > 0])

# Features temporelles
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour

# =========================================================
# 3. SPLIT CHRONOLOGIQUE (avant tout calcul de statistiques)
# =========================================================
# Un split aleatoire melangerait des mesures horaires tres proches entre
# train et test ; la pollution etant fortement autocorrelee dans le temps,
# ca gonflerait artificiellement le score. On trie par date et on garde
# les TEST_SIZE% les plus recents comme test, comme en conditions reelles
# de prediction du futur.

df = df.sort_values("timestamp").reset_index(drop=True)
split_idx = int(len(df) * (1 - TEST_SIZE))
df["split"] = ["train"] * split_idx + ["test"] * (len(df) - split_idx)

print(f"\nSplit chronologique : {split_idx} train / {len(df) - split_idx} test")
print("Periode train :", df.loc[df.split == "train", "timestamp"].min(), "->",
      df.loc[df.split == "train", "timestamp"].max())
print("Periode test  :", df.loc[df.split == "test", "timestamp"].min(), "->",
      df.loc[df.split == "test", "timestamp"].max())

df.to_csv(CLEAN_FILE, index=False)
print(f"\nFichier nettoye sauvegarde : {CLEAN_FILE}")

# =========================================================
# 4. ENCODAGE + NORMALISATION SANS FUITE
# =========================================================

drop_columns = ["timestamp", "timestamp_hour", "source_file"]
existing_drop = [c for c in drop_columns if c in df.columns]

df_model = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

split_col = df_model.pop("split")
y = df_model.pop(TARGET)
X = df_model.drop(columns=existing_drop, errors="ignore")

numeric_features_in_X = X.select_dtypes(include=[np.number]).columns.tolist()

train_mask = (split_col == "train").values
test_mask = (split_col == "test").values

scaler = StandardScaler()
X_scaled = X.copy()
# Cast en float pour eviter une erreur de dtype (les colonnes entieres
# comme year/month/day/hour ne peuvent pas recevoir de valeurs normalisees
# en place sans etre converties d'abord).
X_scaled[numeric_features_in_X] = X_scaled[numeric_features_in_X].astype(float)

# Le scaler est ajuste UNIQUEMENT sur le train, puis applique tel quel
# (memes moyenne/ecart-type appris) au test. C'est ce qui corrige la
# fuite de donnees de la version precedente du script.
scaler.fit(X.loc[train_mask, numeric_features_in_X])
X_scaled.loc[train_mask, numeric_features_in_X] = scaler.transform(X.loc[train_mask, numeric_features_in_X])
X_scaled.loc[test_mask, numeric_features_in_X] = scaler.transform(X.loc[test_mask, numeric_features_in_X])

model_ready_df = X_scaled.copy()
model_ready_df[TARGET] = y.values
model_ready_df["split"] = split_col.values

model_ready_df.to_csv(MODEL_READY_FILE, index=False)

print(f"\nFichier pret pour le modele sauvegarde : {MODEL_READY_FILE}")
print("Shape finale :", model_ready_df.shape)
print("Colonnes :", model_ready_df.columns.tolist())