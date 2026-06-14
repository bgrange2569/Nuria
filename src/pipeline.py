import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ETAPES = [
    ("Export des données Garmin", "export_nuria.py"),
    ("Transformation des données", "transformer_nuria.py"),
    ("Vectorisation des données", "vectoriser_nuria.py"),
]


def main():
    for nom, script in ETAPES:
        print(f"\n=== Étape : {nom} ({script}) ===")
        resultat = subprocess.run([sys.executable, str(SCRIPT_DIR / script)])
        if resultat.returncode != 0:
            print(f"\n❌ Échec de l'étape '{nom}' (code {resultat.returncode}). Arrêt du pipeline.")
            sys.exit(resultat.returncode)
        print(f"✅ Étape '{nom}' terminée avec succès.")

    print("\n🎉 Pipeline terminé avec succès !")


if __name__ == "__main__":
    main()
