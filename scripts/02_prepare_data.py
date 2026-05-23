import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Charger les données
# =========================
INPUT_FILE = r"c:\Users\hp\Desktop\Data sciences\Project\morocco_air_quality_data.csv"
OUTPUT_CLEAN_FILE = "morocco_air_quality_data_clean.csv"
OUTPUT_MODEL_FILE = "morocco_air_quality_data_model_ready.csv"

df = pd.read_csv(INPUT_FILE)

print("Aperçu initial :")
print(df.head())
print("\nTaille initiale :", df.shape)

# =========================
# 2. Vérification générale
# =========================
print("\nInformations sur les colonnes :")
print(df.info())

print("\nValeurs manquantes par colonne :")
print(df.isna().sum())

print("\nNombre de doublons exacts :", df.duplicated().sum())

# =========================
# 3. Supprimer les doublons
# =========================
# Doublons complets
df = df.drop_duplicates()

# Doublons logiques : une seule ligne par ville et heure
if "city" in df.columns and "timestamp_hour" in df.columns:
    df = df.drop_duplicates(subset=["city", "timestamp_hour"], keep="first")

print("\nTaille après suppression des doublons :", df.shape)

# =========================
# 4. Conversion des types
# =========================
# Dates
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

if "timestamp_hour" in df.columns:
    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"], errors="coerce")

# Colonnes numériques attendues
numeric_columns = [
    "lat", "lon",
    "temperature", "feels_like", "humidity", "pressure",
    "wind_speed", "wind_deg", "clouds",
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nTypes après conversion :")
print(df.dtypes)

# =========================
# 5. Gestion des valeurs manquantes
# =========================
# Supprimer les lignes sans cible essentielle
# Ici on choisit PM2.5 comme cible principale pour la régression
TARGET = "pm2_5"

required_columns = ["temperature", "humidity", "pressure", TARGET]
existing_required = [col for col in required_columns if col in df.columns]

df = df.dropna(subset=existing_required)

# Pour les autres colonnes numériques, on remplit par la médiane
for col in numeric_columns:
    if col in df.columns and df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Pour les colonnes catégorielles, on remplit par "Unknown"
categorical_columns = ["city", "weather_main", "weather_description"]
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

print("\nValeurs manquantes après traitement :")
print(df.isna().sum())

# =========================
# 6. Créer des features temporelles utiles
# =========================
if "timestamp" in df.columns:
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour

# =========================
# 7. Sauvegarder une version nettoyée
# =========================
df = df.sort_values(by=["timestamp_hour", "city"]) if "timestamp_hour" in df.columns and "city" in df.columns else df
df.to_csv(OUTPUT_CLEAN_FILE, index=False)

print(f"\nFichier nettoyé sauvegardé : {OUTPUT_CLEAN_FILE}")
print("Taille finale du dataset nettoyé :", df.shape)

# =========================
# 8. Préparer les données pour le modèle
# =========================
# Colonnes à exclure de X
drop_columns = ["timestamp", "timestamp_hour"]

existing_drop_columns = [col for col in drop_columns if col in df.columns]

# Encodage des variables catégorielles
df_model = pd.get_dummies(
    df,
    columns=[col for col in categorical_columns if col in df.columns],
    drop_first=True
)

# Séparer X et y
X = df_model.drop(columns=[TARGET] + existing_drop_columns, errors="ignore")
y = df_model[TARGET]

print("\nShape de X :", X.shape)
print("Shape de y :", y.shape)

# =========================
# 9. Normalisation des variables numériques
# =========================
# Utile surtout pour les modèles linéaires, KNN, réseaux de neurones
numeric_features_in_X = X.select_dtypes(include=[np.number]).columns.tolist()

scaler = StandardScaler()
X_scaled = X.copy()

if len(numeric_features_in_X) > 0:
    X_scaled[numeric_features_in_X] = scaler.fit_transform(X[numeric_features_in_X])

# =========================
# 10. Sauvegarder le dataset prêt pour le modèle
# =========================
model_ready_df = X_scaled.copy()
model_ready_df[TARGET] = y.values

model_ready_df.to_csv(OUTPUT_MODEL_FILE, index=False)

print(f"\nFichier prêt pour le modèle sauvegardé : {OUTPUT_MODEL_FILE}")
print("Taille du dataset final prêt ML :", model_ready_df.shape)

# =========================
# 11. Affichage final
# =========================
print("\nAperçu final :")
print(model_ready_df.head())

print("\nColonnes finales :")
print(model_ready_df.columns.tolist())