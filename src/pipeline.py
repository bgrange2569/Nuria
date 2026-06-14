import sys

from nuria_coach import NuriaCoach, formater_duree


def main():
    try:
        coach = NuriaCoach()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    resume = coach.sync()

    print("\n" + "=" * 50)
    print("📋 Résumé du pipeline")
    print("=" * 50)
    for etape in resume["etapes"]:
        statut = "✅ Succès" if etape["succes"] else "❌ Échec"
        print(f"{statut} — {etape['etape']} ({formater_duree(etape['duree'])})")

    print(f"\nDurée totale : {formater_duree(resume['duree_totale'])}")

    if resume["succes"]:
        print("\n🎉 Pipeline terminé avec succès !")
    else:
        print("\n💥 Pipeline interrompu en raison d'une erreur.")
        sys.exit(1)


if __name__ == "__main__":
    main()
