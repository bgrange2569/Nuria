import json
import sys
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
    try:
        return round(seconds / 3600, 1)
    except (TypeError, ValueError):
        return "N/A"


def or_na(value):
    return "N/A" if value is None else value


def charger_json(chemin, description):
    """Charge un fichier JSON avec gestion des erreurs de lecture."""
    if not chemin.exists():
        print(f"❌ Le fichier {chemin} est introuvable.")
        print(f"   Lancez d'abord l'étape d'export pour générer {description}.")
        sys.exit(1)

    try:
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Le fichier {chemin} est corrompu (JSON invalide) : {e}")
        sys.exit(1)
    except OSError as e:
        print(f"❌ Impossible de lire le fichier {chemin} : {e}")
        sys.exit(1)


def transformer_activite(act):
    """Transforme une activité en document texte. Lève une exception si l'activité est invalide."""
    if not isinstance(act, dict):
        raise ValueError("l'activité n'est pas un objet JSON valide")

    activity_type = (act.get("activityType") or {}).get("typeKey", "Inconnu")
    duree_minutes = round((act.get("duration") or 0) / 60, 1)
    distance_km = round((act.get("distance") or 0) / 1000, 2)

    date_activite = act.get("startTimeLocal", "Date inconnue").split(" ")[0]

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

    return {"type": "activite", "date": date_activite, "texte": texte}


def transformer_jour_bien_etre(jour):
    """Transforme un jour de données de bien-être en document texte."""
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

    return {"type": "bien_etre", "date": jour.get("date"), "texte": texte}


def main():
    # 1. Transformer les activités
    print("📂 Chargement des activités Garmin...")
    activities = charger_json(DATA_DIR / "nuria_activities.json", "le fichier d'activités")

    if not isinstance(activities, list):
        print("❌ Le fichier nuria_activities.json ne contient pas une liste d'activités valide.")
        sys.exit(1)

    print(f"✅ {len(activities)} activités chargées !")

    docs = []
    nb_erreurs = 0
    for i, act in enumerate(activities):
        try:
            docs.append(transformer_activite(act))
        except Exception as e:
            nb_erreurs += 1
            print(f"⚠️  Activité {i + 1} ignorée (données manquantes ou corrompues) : {e}")

    print(f"✅ {len(activities) - nb_erreurs}/{len(activities)} activités transformées avec succès.")
    if nb_erreurs:
        print(f"⚠️  {nb_erreurs} activité(s) ignorée(s) en raison d'erreurs.")

    # 2. Transformer les données de bien-être (sommeil, stress, HRV, FC repos, Body Battery, statut d'entraînement)
    wellness_path = DATA_DIR / "nuria_wellness.json"
    if wellness_path.exists():
        print("📂 Chargement des données de bien-être Garmin...")
        wellness = charger_json(wellness_path, "le fichier de bien-être")

        if not isinstance(wellness, list):
            print("⚠️  Le fichier nuria_wellness.json ne contient pas une liste valide, données ignorées.")
            wellness = []

        print(f"✅ {len(wellness)} jours de données de bien-être chargés !")

        nb_erreurs_bien_etre = 0
        for jour in wellness:
            try:
                docs.append(transformer_jour_bien_etre(jour))
            except Exception as e:
                nb_erreurs_bien_etre += 1
                print(f"⚠️  Jour de bien-être ignoré (données manquantes ou corrompues) : {e}")

        if nb_erreurs_bien_etre:
            print(f"⚠️  {nb_erreurs_bien_etre} jour(s) de bien-être ignoré(s) en raison d'erreurs.")
    else:
        print("ℹ️  Aucun fichier nuria_wellness.json trouvé, données de bien-être ignorées.")

    # Sauvegarder les données transformées
    try:
        with open(DATA_DIR / "nuria_docs.json", "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"❌ Impossible d'écrire le fichier nuria_docs.json : {e}")
        sys.exit(1)

    print(f"💾 {len(docs)} documents transformés et sauvegardés dans data/nuria_docs.json")
    print("🎉 Terminé ! Vous pouvez passer à l'étape suivante.")


if __name__ == "__main__":
    main()
