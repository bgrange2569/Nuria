import json
import shutil
import sys
from pathlib import Path

import ollama
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIRECTORY = BASE_DIR / "db" / "nuria_db"

EMBEDDING_MODEL = "nomic-embed-text"

# Nombre de documents traités par lot lors de la vectorisation
TAILLE_LOT = 20


def verifier_ollama():
    """Vérifie qu'Ollama est lancé et que le modèle d'embedding est disponible."""
    print("🔍 Vérification d'Ollama...")
    try:
        reponse = ollama.list()
    except Exception:
        print("❌ Ollama n'est pas lancé. Démarrez-le avec la commande 'ollama serve' puis relancez le script.")
        sys.exit(1)

    modeles = [m.model for m in reponse.models]
    if not any(m.startswith(EMBEDDING_MODEL) for m in modeles):
        print(f"❌ Le modèle '{EMBEDDING_MODEL}' n'est pas disponible dans Ollama.")
        print(f"   Installez-le avec la commande : ollama pull {EMBEDDING_MODEL}")
        sys.exit(1)

    print("✅ Ollama est lancé et le modèle d'embedding est disponible !")


def charger_documents():
    """Charge les documents transformés depuis nuria_docs.json."""
    chemin = DATA_DIR / "nuria_docs.json"

    if not chemin.exists():
        print(f"❌ Le fichier {chemin} est introuvable.")
        print("   Lancez d'abord l'étape de transformation pour générer ce fichier.")
        sys.exit(1)

    print("📂 Chargement des données transformées...")
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            docs = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Le fichier {chemin} est corrompu (JSON invalide) : {e}")
        sys.exit(1)
    except OSError as e:
        print(f"❌ Impossible de lire le fichier {chemin} : {e}")
        sys.exit(1)

    if not isinstance(docs, list) or not docs:
        print(f"❌ Le fichier {chemin} ne contient aucun document à vectoriser.")
        sys.exit(1)

    print(f"✅ {len(docs)} documents chargés !")
    return docs


def main():
    verifier_ollama()
    docs = charger_documents()

    # Convertir en documents LangChain
    print("📄 Préparation des documents...")
    documents = [
        Document(page_content=doc["texte"], metadata={"type": doc["type"], "date": doc["date"]})
        for doc in docs
    ]

    # Initialiser le modèle d'embedding
    print(f"🧠 Initialisation du modèle d'embedding ({EMBEDDING_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Repartir d'une base vide pour éviter les doublons lors des ré-exécutions
    try:
        shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=False)
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"❌ Impossible de supprimer l'ancienne base {PERSIST_DIRECTORY} : {e}")
        sys.exit(1)

    try:
        PERSIST_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"❌ Impossible de créer le dossier {PERSIST_DIRECTORY.parent} : {e}")
        sys.exit(1)

    # Vectoriser et stocker dans ChromaDB, par lots, avec une barre de progression
    print("⚙️  Vectorisation en cours...")
    try:
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIRECTORY)
        )

        for i in tqdm(range(0, len(documents), TAILLE_LOT), unit="lot", desc="Vectorisation"):
            lot = documents[i:i + TAILLE_LOT]
            vectorstore.add_documents(lot)
    except Exception as e:
        print(f"❌ Erreur lors de la vectorisation ou de l'écriture dans ChromaDB : {e}")
        sys.exit(1)

    print("✅ Vectorisation terminée !")
    print(f"💾 Base de données sauvegardée dans le dossier {PERSIST_DIRECTORY}")
    print("🎉 Terminé ! Vous pouvez passer à l'étape suivante.")


if __name__ == "__main__":
    main()
