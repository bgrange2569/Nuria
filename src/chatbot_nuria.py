import json
import time
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Nombre de documents récents toujours inclus dans le contexte, triés par date
NB_ACTIVITES_RECENTES = 6

# 1. Charger la base de données vectorielle
print("📂 Chargement de la base de données...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="./nuria_db",
    embedding_function=embeddings
)
print("✅ Base de données chargée !")

# 1bis. Charger les documents bruts pour construire un contexte récent trié par date
# (la recherche par similarité seule ne garantit pas de retrouver les données les plus récentes)
print("📂 Chargement des données récentes...")

with open("nuria_docs.json", "r", encoding="utf-8") as f:
    tous_les_docs = json.load(f)

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
    for morceau in qa_chain.stream(question):
        print(morceau, end="", flush=True)
    duree = time.time() - debut
    print(f"\n⏱️  Temps de réponse total : {duree:.1f}s")
    print("-" * 50)
