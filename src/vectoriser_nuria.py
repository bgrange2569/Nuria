import json
import shutil
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

PERSIST_DIRECTORY = "./nuria_db"

# 1. Charger les données transformées
print("📂 Chargement des données transformées...")
with open("nuria_docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

print(f"✅ {len(docs)} activités chargées !")

# 2. Convertir en documents LangChain
print("📄 Préparation des documents...")
documents = [
    Document(page_content=doc["texte"], metadata={"type": doc["type"], "date": doc["date"]})
    for doc in docs
]

# 3. Initialiser le modèle d'embedding
print("🧠 Initialisation du modèle d'embedding (nomic-embed-text)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Repartir d'une base vide pour éviter les doublons lors des ré-exécutions
shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)

# 5. Vectoriser et stocker dans ChromaDB
print("⚙️  Vectorisation en cours... (peut prendre quelques minutes)")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY
)

print("✅ Vectorisation terminée !")
print(f"💾 Base de données sauvegardée dans le dossier {PERSIST_DIRECTORY}")
print("🎉 Terminé ! Vous pouvez passer à l'étape suivante.")
