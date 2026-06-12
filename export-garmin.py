from garminconnect import Garmin
from dotenv import load_dotenv
import os
import json

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
print("🎉 Terminé ! Vous pouvez passer à l'étape suivante.")