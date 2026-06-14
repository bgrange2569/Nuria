import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_path(d, *keys):
    """Accède à une suite de clés imbriquées, retourne None si absent."""
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def to_hours(seconds):
    if seconds is None:
        return "N/A"
    return round(seconds / 3600, 1)


def or_na(value):
    return "N/A" if value is None else value


# 1. Transformer les activités
print("📂 Chargement des activités Garmin...")
with open(DATA_DIR / "nuria_activities.json", "r", encoding="utf-8") as f:
    activities = json.load(f)

print(f"✅ {len(activities)} activités chargées !")

docs = []
for act in activities:
    activity_type = act.get("activityType", {}).get("typeKey", "Inconnu")
    duree_minutes = round(act.get("duration", 0) / 60, 1)
    distance_km = round(act.get("distance", 0) / 1000, 2)

    date_activite = act.get('startTimeLocal', 'Date inconnue').split(" ")[0]

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

    docs.append({"type": "activite", "date": date_activite, "texte": texte})

# 2. Transformer les données de bien-être (sommeil, stress, HRV, FC repos, Body Battery, statut d'entraînement)
if os.path.exists(DATA_DIR / "nuria_wellness.json"):
    print("📂 Chargement des données de bien-être Garmin...")
    with open(DATA_DIR / "nuria_wellness.json", "r", encoding="utf-8") as f:
        wellness = json.load(f)

    print(f"✅ {len(wellness)} jours de données de bien-être chargés !")

    for jour in wellness:
        sleep = jour.get("sleep") or {}
        sleep_dto = get_path(sleep, "dailySleepDTO") or {}
        sommeil_total = to_hours(sleep_dto.get("sleepTimeSeconds"))
        sommeil_profond = to_hours(sleep_dto.get("deepSleepSeconds"))
        sommeil_leger = to_hours(sleep_dto.get("lightSleepSeconds"))
        sommeil_rem = to_hours(sleep_dto.get("remSleepSeconds"))
        sommeil_eveil = to_hours(sleep_dto.get("awakeSleepSeconds"))
        score_sommeil = or_na(get_path(sleep_dto, "sleepScores", "overall", "value"))

        stress = jour.get("stress") or {}
        stress_moyen = or_na(stress.get("avgStressLevel"))
        stress_max = or_na(stress.get("maxStressLevel"))

        rhr_liste = get_path(jour.get("rhr") or {}, "allMetrics", "metricsMap", "WELLNESS_RESTING_HEART_RATE")
        fc_repos = or_na(rhr_liste[0].get("value")) if rhr_liste else "N/A"

        hrv_summary = get_path(jour.get("hrv") or {}, "hrvSummary") or {}
        hrv_derniere_nuit = or_na(hrv_summary.get("lastNightAvg"))
        hrv_statut = or_na(hrv_summary.get("status"))

        training_status = jour.get("training_status") or {}
        latest_status = get_path(training_status, "mostRecentTrainingStatus", "latestTrainingStatusData") or {}
        statut_entrainement = "N/A"
        for device_data in latest_status.values():
            statut_entrainement = or_na(device_data.get("trainingStatusFeedbackPhrase") or device_data.get("trainingStatus"))
            break
        vo2max = or_na(
            get_path(training_status, "mostRecentVO2Max", "generic", "vo2MaxPreciseValue")
            or get_path(training_status, "mostRecentVO2Max", "generic", "vo2MaxValue")
        )

        body_battery = jour.get("body_battery") or {}
        bb_charge = or_na(body_battery.get("charged"))
        bb_decharge = or_na(body_battery.get("drained"))

        texte = f"""
Bien-être du {jour.get('date')}
Sommeil total : {sommeil_total} h (score : {score_sommeil})
Sommeil profond : {sommeil_profond} h | léger : {sommeil_leger} h | REM : {sommeil_rem} h | éveil : {sommeil_eveil} h
Niveau de stress moyen : {stress_moyen} | maximum : {stress_max}
Fréquence cardiaque au repos : {fc_repos} bpm
VFC (HRV) dernière nuit : {hrv_derniere_nuit} ms (statut : {hrv_statut})
Statut d'entraînement : {statut_entrainement}
VO2 Max estimé : {vo2max}
Body Battery rechargé : {bb_charge} | déchargé : {bb_decharge}
        """.strip()

        docs.append({"type": "bien_etre", "date": jour.get('date'), "texte": texte})
else:
    print("ℹ️  Aucun fichier nuria_wellness.json trouvé, données de bien-être ignorées.")

# Sauvegarder les données transformées
with open(DATA_DIR / "nuria_docs.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, indent=2, ensure_ascii=False)

print(f"💾 {len(docs)} documents transformés et sauvegardés dans data/nuria_docs.json")
print("🎉 Terminé ! Vous pouvez passer à l'étape suivante.")
