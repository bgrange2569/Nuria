import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Nombre de jours pris en compte pour la progression du VO2max
NB_JOURS_VO2MAX = 90


def charger_activites():
    """Charge les activités Garmin avec gestion des erreurs de lecture."""
    chemin = DATA_DIR / "nuria_activities.json"

    if not chemin.exists():
        print(f"❌ Le fichier {chemin} est introuvable.")
        print("   Lancez d'abord l'étape d'export pour générer ce fichier.")
        sys.exit(1)

    try:
        with open(chemin, "r", encoding="utf-8") as f:
            activities = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Le fichier {chemin} est corrompu (JSON invalide) : {e}")
        sys.exit(1)
    except OSError as e:
        print(f"❌ Impossible de lire le fichier {chemin} : {e}")
        sys.exit(1)

    if not isinstance(activities, list):
        print(f"❌ Le fichier {chemin} ne contient pas une liste d'activités valide.")
        sys.exit(1)

    return activities


def parser_date(act):
    """Convertit le champ startTimeLocal en datetime, ou None si invalide."""
    try:
        return datetime.strptime(act.get("startTimeLocal", ""), "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def cle_semaine(d):
    annee, semaine, _ = d.isocalendar()
    return f"{annee}-S{semaine:02d}"


def cle_mois(d):
    return d.strftime("%Y-%m")


def calculer_charge_par_periode(activites_dates, cle_fn):
    """Additionne la charge d'entraînement par période (semaine ou mois)."""
    charges = defaultdict(float)
    for d, act in activites_dates:
        charge = act.get("activityTrainingLoad") or 0
        charges[cle_fn(d)] += charge
    return dict(sorted((k, round(v, 1)) for k, v in charges.items()))


def calculer_vo2max_progression(activites_dates, depuis):
    """Retourne la progression du VO2max (date, valeur) depuis une date donnée."""
    progression = []
    for d, act in activites_dates:
        vo2 = act.get("vO2MaxValue")
        if vo2 is not None and d.date() >= depuis:
            progression.append({"date": d.strftime("%Y-%m-%d"), "valeur": vo2})

    return sorted(progression, key=lambda x: x["date"])


def calculer_repartition_sports(activites):
    """Compte le nombre d'activités par type de sport."""
    return dict(Counter(
        (act.get("activityType") or {}).get("typeKey", "inconnu")
        for act in activites
    ))


def calculer_stats_periode(activites_dates, debut, fin):
    """Calcule les statistiques d'entraînement pour une période [debut, fin)."""
    nb_activites = 0
    duree_totale_min = 0.0
    distance_totale_km = 0.0
    charge_totale = 0.0
    jours_actifs = set()

    for d, act in activites_dates:
        if debut <= d.date() < fin:
            nb_activites += 1
            duree_totale_min += (act.get("duration") or 0) / 60
            distance_totale_km += (act.get("distance") or 0) / 1000
            charge_totale += act.get("activityTrainingLoad") or 0
            jours_actifs.add(d.date())

    return {
        "nb_activites": nb_activites,
        "duree_totale_minutes": round(duree_totale_min, 1),
        "distance_totale_km": round(distance_totale_km, 2),
        "charge_totale": round(charge_totale, 1),
        "jours_actifs": len(jours_actifs),
    }


def calculer_derniere_activite(activites_dates):
    """Retourne un résumé de la dernière activité, ou None si aucune."""
    if not activites_dates:
        return None

    d, act = activites_dates[-1]
    charge = act.get("activityTrainingLoad")
    return {
        "date": d.strftime("%Y-%m-%d %H:%M:%S"),
        "type": (act.get("activityType") or {}).get("typeKey", "inconnu"),
        "distance_km": round((act.get("distance") or 0) / 1000, 2),
        "duree_minutes": round((act.get("duration") or 0) / 60, 1),
        "charge": round(charge, 1) if charge is not None else None,
    }


def calculer_agregats(activites, aujourd_hui=None):
    """Calcule l'ensemble des agrégats à partir de la liste d'activités."""
    if aujourd_hui is None:
        aujourd_hui = date.today()

    activites_dates = []
    for act in activites:
        d = parser_date(act)
        if d is not None:
            activites_dates.append((d, act))

    activites_dates.sort(key=lambda x: x[0])

    debut_semaine_actuelle = aujourd_hui - timedelta(days=aujourd_hui.weekday())
    debut_semaine_precedente = debut_semaine_actuelle - timedelta(days=7)
    fin_semaine_actuelle = aujourd_hui + timedelta(days=1)

    charge_par_semaine = calculer_charge_par_periode(activites_dates, cle_semaine)
    charge_par_mois = calculer_charge_par_periode(activites_dates, cle_mois)

    charge_moyenne_hebdomadaire = (
        round(sum(charge_par_semaine.values()) / len(charge_par_semaine), 1)
        if charge_par_semaine else 0.0
    )
    charge_moyenne_mensuelle = (
        round(sum(charge_par_mois.values()) / len(charge_par_mois), 1)
        if charge_par_mois else 0.0
    )

    depuis_3_mois = aujourd_hui - timedelta(days=NB_JOURS_VO2MAX)
    vo2max_progression = calculer_vo2max_progression(activites_dates, depuis_3_mois)

    repartition_sports = calculer_repartition_sports(activites)

    semaine_actuelle = calculer_stats_periode(activites_dates, debut_semaine_actuelle, fin_semaine_actuelle)
    semaine_precedente = calculer_stats_periode(activites_dates, debut_semaine_precedente, debut_semaine_actuelle)

    nb_jours_ecoules = (aujourd_hui - debut_semaine_actuelle).days + 1
    jours_repos_semaine_actuelle = max(0, nb_jours_ecoules - semaine_actuelle["jours_actifs"])

    return {
        "date_calcul": datetime.now().isoformat(),
        "charge_par_semaine": charge_par_semaine,
        "charge_par_mois": charge_par_mois,
        "charge_moyenne_hebdomadaire": charge_moyenne_hebdomadaire,
        "charge_moyenne_mensuelle": charge_moyenne_mensuelle,
        "vo2max_progression_3_mois": vo2max_progression,
        "repartition_sports": repartition_sports,
        "semaine_actuelle": semaine_actuelle,
        "semaine_precedente": semaine_precedente,
        "jours_repos_semaine_actuelle": jours_repos_semaine_actuelle,
        "derniere_activite": calculer_derniere_activite(activites_dates),
    }


def formater_resume_agregats(agregats):
    """Construit un résumé compact en français à partir des agrégats, pour le prompt du chatbot."""
    lignes = []

    derniere = agregats.get("derniere_activite")
    if derniere:
        lignes.append(
            f"Dernière séance : {derniere['date']} - {derniere['type']} - "
            f"{derniere['distance_km']} km en {derniere['duree_minutes']} min "
            f"(charge : {derniere['charge']})"
        )

    lignes.append(
        f"Charge d'entraînement moyenne : {agregats['charge_moyenne_hebdomadaire']}/semaine, "
        f"{agregats['charge_moyenne_mensuelle']}/mois"
    )

    sa = agregats["semaine_actuelle"]
    sp = agregats["semaine_precedente"]
    lignes.append(
        f"Semaine actuelle : {sa['nb_activites']} séance(s), {sa['distance_totale_km']} km, "
        f"{sa['duree_totale_minutes']} min, charge {sa['charge_totale']} "
        f"(semaine précédente : {sp['nb_activites']} séance(s), {sp['distance_totale_km']} km, "
        f"{sp['duree_totale_minutes']} min, charge {sp['charge_totale']})"
    )

    lignes.append(f"Jours de repos cette semaine : {agregats['jours_repos_semaine_actuelle']}")

    sports = agregats.get("repartition_sports") or {}
    if sports:
        repartition = ", ".join(f"{sport} : {nb}" for sport, nb in sorted(sports.items(), key=lambda x: -x[1]))
        lignes.append(f"Répartition des sports (total) : {repartition}")

    vo2 = agregats.get("vo2max_progression_3_mois") or []
    if vo2:
        premier, dernier = vo2[0], vo2[-1]
        lignes.append(
            f"VO2max sur les 3 derniers mois : de {premier['valeur']} ({premier['date']}) "
            f"à {dernier['valeur']} ({dernier['date']})"
        )

    return "\n".join(lignes)


def main():
    print("📂 Chargement des activités Garmin...")
    activites = charger_activites()
    print(f"✅ {len(activites)} activités chargées !")

    print("📊 Calcul des agrégats...")
    agregats = calculer_agregats(activites)

    chemin_sortie = DATA_DIR / "agregats.json"
    try:
        with open(chemin_sortie, "w", encoding="utf-8") as f:
            json.dump(agregats, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"❌ Impossible d'écrire le fichier {chemin_sortie} : {e}")
        sys.exit(1)

    print(f"💾 Agrégats sauvegardés dans data/agregats.json")
    print("🎉 Terminé !")


if __name__ == "__main__":
    main()
