from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)
from dotenv import load_dotenv
from datetime import date, timedelta
from pathlib import Path
import os
import sys
import json
import requests
import ollama

# Nombre de jours d'historique à récupérer pour les données de bien-être
NB_JOURS_BIEN_ETRE = 30

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def verifier_ollama():
    """Vérifie qu'Ollama est lancé et accessible."""
    print("🔍 Vérification d'Ollama...")
    try:
        ollama.list()
    except Exception:
        print("❌ Ollama n'est pas lancé. Démarrez-le avec la commande 'ollama serve' puis relancez le script.")
        sys.exit(1)
    print("✅ Ollama est lancé !")


def charger_identifiants():
    """Charge les identifiants Garmin depuis le fichier .env."""
    load_dotenv()
    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")

    if not email or not password:
        print("❌ Le fichier .env est manquant ou incomplet.")
        print("   Vérifiez qu'il existe à la racine du projet et qu'il contient :")
        print("   GARMIN_EMAIL=votre_email")
        print("   GARMIN_PASSWORD=votre_mot_de_passe")
        sys.exit(1)

    return email, password


def connexion_garmin(email, password):
    """Se connecte à Garmin Connect avec gestion des erreurs courantes."""
    print("🔗 Connexion à Garmin Connect...")
    try:
        client = Garmin(email, password)
        client.login()
    except GarminConnectAuthenticationError:
        print("❌ Échec de l'authentification Garmin Connect.")
        print("   Vérifiez vos identifiants dans le fichier .env, et assurez-vous que")
        print("   la double authentification (2FA) n'est pas requise pour ce compte.")
        sys.exit(1)
    except GarminConnectTooManyRequestsError:
        print("❌ Trop de requêtes envoyées à Garmin Connect. Réessayez dans quelques minutes.")
        sys.exit(1)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, GarminConnectConnectionError):
        print("❌ Impossible de contacter Garmin Connect. Vérifiez votre connexion internet et réessayez.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur inattendue lors de la connexion à Garmin Connect : {e}")
        sys.exit(1)

    print("✅ Connecté !")
    return client


def recuperer_activites(client):
    """Récupère les 100 dernières activités Garmin."""
    print("📥 Récupération de vos activités...")
    try:
        activities = client.get_activities(0, 100)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        print("❌ Délai d'attente ou problème réseau lors de la récupération des activités.")
        sys.exit(1)
    except GarminConnectTooManyRequestsError:
        print("❌ Trop de requêtes envoyées à Garmin Connect. Réessayez dans quelques minutes.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur inattendue lors de la récupération des activités : {e}")
        sys.exit(1)

    print(f"✅ {len(activities)} activités récupérées !")
    return activities


def sauvegarder_json(donnees, chemin, description):
    """Sauvegarde des données en JSON avec gestion des erreurs d'écriture."""
    try:
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(donnees, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"❌ Impossible d'écrire le fichier {chemin} : {e}")
        sys.exit(1)

    print(f"💾 {description} sauvegardées dans {chemin.relative_to(chemin.parent.parent)}")


def main():
    verifier_ollama()
    email, password = charger_identifiants()
    client = connexion_garmin(email, password)

    activities = recuperer_activites(client)
    sauvegarder_json(activities, DATA_DIR / "nuria_activities.json", "Données d'activités")

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

    sauvegarder_json(wellness, DATA_DIR / "nuria_wellness.json", f"Données de bien-être ({len(wellness)} jours)")
    print("🎉 Terminé ! Vous pouvez passer à l'étape suivante.")


if __name__ == "__main__":
    main()
