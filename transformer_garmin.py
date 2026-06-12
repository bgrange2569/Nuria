import json

# Charger les données exportées
print("📂 Chargement des données Garmin...")
with open("garmin_activities.json", "r", encoding="utf-8") as f:
    activities = json.load(f)

print(f"✅ {len(activities)} activités chargées !")

docs = []
for act in activities:
    # Récupérer le type d'activité
    activity_type = act.get("activityType", {}).get("typeKey", "Inconnu")
    
    # Convertir la durée en minutes
    duree_minutes = round(act.get("duration", 0) / 60, 1)
    
    # Convertir la distance en km
    distance_km = round(act.get("distance", 0) / 1000, 2)

    texte = f"""
Activité du {act.get('startTimeLocal', 'Date inconnue')}
Type : {activity_type}
Distance : {distance_km} km
Durée : {duree_minutes} minutes
Fréquence cardiaque moyenne : {act.get('averageHR', 'N/A')} bpm
Fréquence cardiaque max : {act.get('maxHR', 'N/A')} bpm
Dénivelé positif : {act.get('elevationGain', 'N/A')} m
Calories brûlées : {act.get('calories', 'N/A')} kcal
VO2 Max estimé : {act.get('vO2MaxValue', 'N/A')}
Charge d'entraînement : {act.get('activityTrainingLoad', 'N/A')}
Score aérobie : {act.get('aerobicTrainingEffect', 'N/A')}
Score anaérobie : {act.get('anaerobicTrainingEffect', 'N/A')}
    """.strip()
    
    docs.append(texte)

# Sauvegarder les données transformées
with open("garmin_docs.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2, ensure_ascii=False)

print(f"💾 {len(docs)} activités transformées et sauvegardées dans garmin_docs.json")
print("🎉 Terminé ! Vous pouvez passer à l'étape suivante.")