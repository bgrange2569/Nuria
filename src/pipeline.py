import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ETAPES = [
    ("Export des données Garmin", "export_nuria.py"),
    ("Transformation des données", "transformer_nuria.py"),
    ("Vectorisation des données", "vectoriser_nuria.py"),
]


def formater_duree(secondes):
    minutes, secondes = divmod(round(secondes), 60)
    if minutes:
        return f"{minutes} min {secondes} s"
    return f"{secondes} s"


def main():
    resultats = []

    for nom, script in ETAPES:
        print(f"\n=== Étape : {nom} ({script}) ===")
        debut = time.time()
        resultat = subprocess.run([sys.executable, str(SCRIPT_DIR / script)])
        duree = time.time() - debut

        if resultat.returncode != 0:
            print(f"\n❌ Échec de l'étape '{nom}' (code {resultat.returncode}) après {formater_duree(duree)}. Arrêt du pipeline.")
            resultats.append((nom, False, duree))
            afficher_resume(resultats, succes_global=False)
            sys.exit(resultat.returncode)

        print(f"✅ Étape '{nom}' terminée avec succès en {formater_duree(duree)}.")
        resultats.append((nom, True, duree))

    afficher_resume(resultats, succes_global=True)


def afficher_resume(resultats, succes_global):
    print("\n" + "=" * 50)
    print("📋 Résumé du pipeline")
    print("=" * 50)
    for nom, succes, duree in resultats:
        statut = "✅ Succès" if succes else "❌ Échec"
        print(f"{statut} — {nom} ({formater_duree(duree)})")

    duree_totale = sum(duree for _, _, duree in resultats)
    print(f"\nDurée totale : {formater_duree(duree_totale)}")

    if succes_global:
        print("\n🎉 Pipeline terminé avec succès !")
    else:
        print("\n💥 Pipeline interrompu en raison d'une erreur.")


if __name__ == "__main__":
    main()
