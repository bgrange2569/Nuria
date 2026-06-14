import json
import sys
import time
from pathlib import Path

import ollama
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Nombre de documents récents toujours inclus dans le contexte, triés par date
NB_ACTIVITES_RECENTES = 6

LLM_MODEL = "qwen2.5:14b-instruct"
EMBEDDING_MODEL = "nomic-embed-text"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIRECTORY = BASE_DIR / "db" / "nuria_db"


def verifier_ollama():
    """Vérifie qu'Ollama est lancé et que les modèles nécessaires sont disponibles."""
    print("🔍 Vérification d'Ollama...")
    try:
        reponse = ollama.list()
    except Exception:
        print("❌ Ollama n'est pas lancé. Démarrez-le avec la commande 'ollama serve' puis relancez le chatbot.")
        sys.exit(1)

    modeles = [m.model for m in reponse.models]
    for modele in (LLM_MODEL, EMBEDDING_MODEL):
        if not any(m.startswith(modele) for m in modeles):
            print(f"❌ Le modèle '{modele}' n'est pas disponible dans Ollama.")
            print(f"   Installez-le avec la commande : ollama pull {modele}")
            sys.exit(1)

    print("✅ Ollama est lancé et les modèles nécessaires sont disponibles !")


verifier_ollama()

# 1. Charger la base de données vectorielle
print("📂 Chargement de la base de données...")
if not PERSIST_DIRECTORY.exists():
    print(f"❌ La base de données vectorielle {PERSIST_DIRECTORY} est introuvable.")
    print("   Lancez d'abord le pipeline (python pipeline.py) pour la générer.")
    sys.exit(1)

try:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=embeddings
    )
except Exception as e:
    print(f"❌ Impossible de charger la base de données vectorielle : {e}")
    sys.exit(1)
print("✅ Base de données chargée !")

# 1bis. Charger les documents bruts pour construire un contexte récent trié par date
# (la recherche par similarité seule ne garantit pas de retrouver les données les plus récentes)
print("📂 Chargement des données récentes...")

docs_path = DATA_DIR / "nuria_docs.json"
if not docs_path.exists():
    print(f"❌ Le fichier {docs_path} est introuvable.")
    print("   Lancez d'abord le pipeline (python pipeline.py) pour générer les données.")
    sys.exit(1)

try:
    with open(docs_path, "r", encoding="utf-8") as f:
        tous_les_docs = json.load(f)
except json.JSONDecodeError as e:
    print(f"❌ Le fichier {docs_path} est corrompu (JSON invalide) : {e}")
    sys.exit(1)

activites = sorted(
    (d for d in tous_les_docs if d["type"] == "activite"),
    key=lambda d: d["date"], reverse=True
)

activites_recentes = "\n\n".join(d["texte"] for d in activites[:NB_ACTIVITES_RECENTES])
print(f"✅ {min(len(activites), NB_ACTIVITES_RECENTES)} activités récentes chargées !")

# 2. Définir le prompt système
prompt_template = """
Tu es Nuria, un COACH D'ENTRAÎNEMENT SPORTIF. Ton rôle est d'analyser les séances de la
personne que tu coaches et de construire des plans d'entraînement, à partir de ses données Garmin.
Tu n'es PAS un coach bien-être ou santé : ne parle jamais de sommeil, de stress, de VFC,
de Body Battery ou de récupération, et ne donne pas de conseils de bien-être généraux
(hydratation, sommeil, gestion du stress...), même si la question porte là-dessus.
Concentre-toi uniquement sur les séances, la charge d'entraînement, la progression,
l'allure et les plans d'entraînement.

Voici d'autres séances pertinentes pour répondre à la question :
{context}

Voici les séances d'entraînement les plus récentes, triées de la plus récente (en premier) à la plus ancienne.
C'est la source la plus fiable pour savoir quelle est la dernière séance ou pour construire un plan d'entraînement :
{activites_recentes}

Question : {question}

Réponds en français, de manière claire et structurée, en te concentrant sur l'entraînement.
Si tu ne trouves pas l'information dans les données, dis-le honnêtement.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["activites_recentes", "context", "question"]
)

# 3. Initialiser le LLM
print("🧠 Initialisation du modèle LLM (qwen2.5:14b-instruct)...")
llm = ChatOllama(model="qwen2.5:14b-instruct", temperature=0.3, num_ctx=8192, num_predict=900)

# 4. Créer la chaîne RAG (on exclut les documents de bien-être de la recherche)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2, "filter": {"type": "activite"}})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def afficher_taille_prompt(valeur_prompt):
    texte = valeur_prompt.to_string()
    print(f"📏 Taille du prompt : {len(texte)} caractères (~{len(texte) // 4} tokens estimés)")
    return valeur_prompt


qa_chain = (
    {
        "activites_recentes": lambda _: activites_recentes,
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | RunnableLambda(afficher_taille_prompt)
    | llm
    | StrOutputParser()
)

print("✅ Chatbot prêt !")
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

# 5. Boucle du chatbot
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
        for morceau in qa_chain.stream(question):
            print(morceau, end="", flush=True)
    except Exception as e:
        print(f"\n❌ Erreur lors de la génération de la réponse : {e}")
        print("   Vérifiez qu'Ollama est toujours lancé (commande 'ollama serve').")
        print("-" * 50)
        continue
    duree = time.time() - debut
    print(f"\n⏱️  Temps de réponse total : {duree:.1f}s")
    print("-" * 50)
