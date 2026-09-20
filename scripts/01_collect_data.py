import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise EnvironmentError(
        "OPENWEATHER_API_KEY manquant. "
        "Vérifie que ton fichier .env existe et contient : "
        "OPENWEATHER_API_KEY=ta_cle_ici"
    )

CSV_FILE = "../data/raw/morocco_air_quality_data.csv"

COLLECTION_INTERVAL_SECONDS = 3600  # 1 heure

REQUEST_TIMEOUT = 40
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10

# 7 villes marocaines (base commune de l'equipe, unifiee apres fusion
# des 3 collectes paralleles menees separement au debut du projet)
CITIES = {
    "Casablanca":       {"lat": 33.5731, "lon": -7.5898},
    "Rabat":             {"lat": 34.0209, "lon": -6.8416},
    "Marrakech":         {"lat": 31.6295, "lon": -7.9811},
    "Fes":               {"lat": 34.0331, "lon": -5.0003},
    "Agadir":            {"lat": 30.4278, "lon": -9.5981},
    "Tanger":            {"lat": 35.7595, "lon": -5.8340},
    "Karia Ba Mohamed":  {"lat": 34.3667, "lon": -5.2139},
}

# =========================================================
# API FUNCTIONS WITH RETRY
# =========================================================

def request_with_retry(url, params, api_name):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            try:
                return response.json()
            except ValueError as e:
                # Reponse HTTP 200 mais corps non-JSON (rare mais arrive) :
                # on traite ca comme un echec a retenter, pas comme un crash.
                raise requests.exceptions.RequestException(
                    f"Reponse non-JSON recue de {api_name}: {e}"
                )

        except requests.exceptions.RequestException as e:
            print(f"{api_name} attempt {attempt}/{MAX_RETRIES} failed: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise Exception(f"{api_name} failed after {MAX_RETRIES} attempts")


def get_weather_data(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
    return request_with_retry(url, params, "Weather API")


def get_air_quality_data(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": API_KEY}
    return request_with_retry(url, params, "Air Quality API")


# =========================================================
# DATA COLLECTION
# =========================================================

def collect_one_city(city_name, lat, lon):
    weather = get_weather_data(lat, lon)
    time.sleep(2)
    air = get_air_quality_data(lat, lon)

    air_item = air["list"][0]
    components = air_item["components"]

    now = datetime.now()
    timestamp_hour = now.strftime("%Y-%m-%d %H:00:00")

    row = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_hour": timestamp_hour,
        "city": city_name,
        "lat": lat,
        "lon": lon,

        # Weather data
        "temperature": weather["main"].get("temp"),
        "feels_like": weather["main"].get("feels_like"),
        "humidity": weather["main"].get("humidity"),
        "pressure": weather["main"].get("pressure"),
        "wind_speed": weather.get("wind", {}).get("speed"),
        "wind_deg": weather.get("wind", {}).get("deg"),
        "clouds": weather.get("clouds", {}).get("all"),
        "weather_main": weather["weather"][0].get("main"),
        "weather_description": weather["weather"][0].get("description"),

        # Air quality data
        "aqi": air_item["main"].get("aqi"),
        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "nh3": components.get("nh3"),
    }

    return row


# =========================================================
# SAVE DATA
# =========================================================

def save_rows_without_duplicates(new_rows):
    new_df = pd.DataFrame(new_rows)

    if os.path.exists(CSV_FILE):
        old_df = pd.read_csv(CSV_FILE)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df

    df = df.drop_duplicates(subset=["city", "timestamp_hour"], keep="first")
    df = df.sort_values(by=["timestamp_hour", "city"])

    try:
        df.to_csv(CSV_FILE, index=False)
        print(f"Saved data to {CSV_FILE}")
        print(f"Total rows: {len(df)}")

    except PermissionError:
        backup_file = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(backup_file, index=False)
        print("ERROR: CSV file is open in Excel or another program.")
        print(f"Data saved instead to backup file: {backup_file}")


# =========================================================
# COLLECT ALL CITIES
# =========================================================

def collect_all_cities_once():
    rows = []

    for city_name, coords in CITIES.items():
        try:
            print(f"Collecting data for {city_name}...")
            row = collect_one_city(city_name, coords["lat"], coords["lon"])
            rows.append(row)
            print(f"{city_name}: OK")

        except Exception as e:
            print(f"{city_name}: ERROR -> {e}")

        time.sleep(3)

    if rows:
        save_rows_without_duplicates(rows)
    else:
        print("No data collected in this round.")


# =========================================================
# MAIN
# =========================================================
# NOTE : le mode boucle infinie doit rester actif dans un terminal
# ouvert pour collecter en continu en local. Le mode --once, lui, est
# concu pour etre lance par un declencheur externe (GitHub Actions
# planifie via cron, ou un Planificateur de taches) qui gere lui-meme
# la periodicite : le script fait une seule collecte puis se termine.

if __name__ == "__main__":
    if "--once" in sys.argv:
        print("Mode --once : une seule collecte.")
        collect_all_cities_once()
    else:
        while True:
            print("\n====================================")
            print("New collection round")
            print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("====================================")

            collect_all_cities_once()

            print("\nWaiting 1 hour before next collection...\n")
            time.sleep(COLLECTION_INTERVAL_SECONDS)