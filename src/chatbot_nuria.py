import sys
import time

from nuria_coach import NuriaCoach

print("📂 Initialisation de Nuria...")
try:
    coach = NuriaCoach()
except RuntimeError as e:
    print(f"❌ {e}")
    sys.exit(1)

if coach.qa_chain is None:
    print("❌ Aucune donnée disponible. Lancez d'abord 'python pipeline.py' pour générer les données.")
    sys.exit(1)

print("✅ Nuria est prête !")

print("=" * 50)
print("🏃 Bienvenue sur Nuria, votre coach d'entraînement IA !")
print("💡 Exemples de questions :")
print("   - Quelle est ma dernière séance ?")
print("   - Quelle est ma progression ce mois-ci ?")
print("   - Quelle est ma séance la plus longue ?")
print("   - Analyse ma charge d'entraînement")
print("   - Construis-moi un plan d'entraînement pour la semaine prochaine")
print("   Tapez 'quit' pour quitter")
print("=" * 50)

# Boucle du chatbot
while True:
    print("")
    question = input("Vous 👤 : ")

    if question.lower() in ["quit", "exit", "quitter"]:
        print("👋 À bientôt pour votre prochain entraînement !")
        break

    if not question.strip():
        print("⚠️  Veuillez poser une question.")
        continue

    print("\n🤖 Nuria : ", end="", flush=True)
    debut = time.time()
    try:
        for morceau in coach.chat(question):
            print(morceau, end="", flush=True)
    except RuntimeError as e:
        print(f"\n❌ {e}")
        print("-" * 50)
        continue
    duree = time.time() - debut
    print(f"\n⏱️  Temps de réponse total : {duree:.1f}s")
    print("-" * 50)
