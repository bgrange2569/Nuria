from garminconnect import Garmin
from dotenv import load_dotenv
from datetime import date, timedelta
import os
import json

# Nombre de jours d'historique à récupérer pour les données de bien-être
NB_JOURS_BIEN_ETRE = 30

# Charger les identifiants depuis le fichier .env
load_dotenv()
email = os.getenv("GARMIN_EMAIL")
password = os.getenv("GARMIN_PASSWORD")

# Connexion à Garmin Connect
print("🔗 Connexion à Garmin Connect...")
client = Garmin(email, password)
client.login()
print("✅ Connecté !")

# Récupérer les 100 dernières activités
print("📥 Récupération de vos activités...")
activities = client.get_activities(0, 100)
print(f"✅ {len(activities)} activités récupérées !")

# Sauvegarder en JSON
with open("garmin_activities.json", "w", encoding="utf-8") as f:
    json.dump(activities, f, indent=2, ensure_ascii=False)

print("💾 Données sauvegardées dans garmin_activities.json")

# Récupérer les données de bien-être (sommeil, stress, FC repos, HRV, statut d'entraînement, Body Battery)
print(f"📥 Récupération de vos données de bien-être sur les {NB_JOURS_BIEN_ETRE} derniers jours...")

today = date.today()
start_date = today - timedelta(days=NB_JOURS_BIEN_ETRE - 1)

try:
    body_battery_range = client.get_body_battery(start_date.isoformat(), today.isoformat())
except Exception as e:
    print(f"⚠️  Impossible de récupérer le Body Battery : {e}")
    body_battery_range = []

body_battery_par_date = {entry.get("date"): entry for entry in body_battery_range}

wellness = []
for i in range(NB_JOURS_BIEN_ETRE):
    cdate = (start_date + timedelta(days=i)).isoformat()
    entry = {"date": cdate}

    try:
        entry["sleep"] = client.get_sleep_data(cdate)
    except Exception as e:
        print(f"⚠️  Sommeil indisponible pour {cdate} : {e}")
        entry["sleep"] = None

    try:
        entry["stress"] = client.get_stress_data(cdate)
    except Exception as e:
        print(f"⚠️  Stress indisponible pour {cdate} : {e}")
        entry["stress"] = None

    try:
        entry["rhr"] = client.get_rhr_day(cdate)
    except Exception as e:
        print(f"⚠️  FC repos indisponible pour {cdate} : {e}")
        entry["rhr"] = None

    try:
        entry["hrv"] = client.get_hrv_data(cdate)
    except Exception as e:
        print(f"⚠️  HRV indisponible pour {cdate} : {e}")
        entry["hrv"] = None

    try:
        entry["training_status"] = client.get_training_status(cdate)
    except Exception as e:
        print(f"⚠️  Statut d'entraînement indisponible pour {cdate} : {e}")
        entry["training_status"] = None

    entry["body_battery"] = body_battery_par_date.get(cdate)

    wellness.append(entry)

with open("garmin_wellness.json", "w", encoding="utf-8") as f:
    json.dump(wellness, f, indent=2, ensure_ascii=False)

print(f"💾 Données de bien-être sauvegardées dans garmin_wellness.json ({len(wellness)} jours)")
print("🎉 Terminé ! Vous pouvez passer à l'étape suivante.")
