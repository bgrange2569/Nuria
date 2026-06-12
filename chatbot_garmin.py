from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Charger la base de données vectorielle
print("📂 Chargement de la base de données...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="./garmin_db",
    embedding_function=embeddings
)
print("✅ Base de données chargée !")

# 2. Définir le prompt système
prompt_template = """
Tu es un coach sportif expert en analyse de données d'entraînement et de bien-être Garmin
(activités, sommeil, stress, fréquence cardiaque au repos, VFC, Body Battery, statut d'entraînement).
Tu analyses ces données et tu donnes des conseils personnalisés, bienveillants et motivants en français,
en tenant compte à la fois de la charge d'entraînement et de la récupération.

Voici les données pertinentes :
{context}

Question : {question}

Réponds en français, de manière claire et structurée.
Si tu ne trouves pas l'information dans les données, dis-le honnêtement.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 3. Initialiser le LLM
print("🧠 Initialisation du modèle LLM (llama3.1)...")
llm = ChatOllama(model="llama3.1", temperature=0.3)

# 4. Créer la chaîne RAG
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ Chatbot prêt !")
print("=" * 50)
print("🏃 Bienvenue sur votre Coach Garmin IA !")
print("💡 Exemples de questions :")
print("   - Quelle est ma progression ce mois-ci ?")
print("   - Quelle est ma séance la plus longue ?")
print("   - Analyse ma charge d'entraînement")
print("   - Donne moi des conseils pour progresser")
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

    print("\n🤖 Coach IA (en cours de réflexion...)\n")
    reponse = qa_chain.invoke(question)
    print(f"🤖 Coach IA : {reponse}")
    print("-" * 50)
