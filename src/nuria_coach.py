import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import ollama
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIRECTORY = BASE_DIR / "db" / "nuria_db"
DOCS_PATH = DATA_DIR / "nuria_docs.json"

LLM_MODEL = "qwen2.5:14b-instruct"
EMBEDDING_MODEL = "nomic-embed-text"

# Nombre de documents récents toujours inclus dans le contexte, triés par date
NB_ACTIVITES_RECENTES = 6

# Étapes du pipeline de synchronisation
ETAPES_SYNC = [
    ("Export des données Garmin", "export_nuria.py"),
    ("Transformation des données", "transformer_nuria.py"),
    ("Vectorisation des données", "vectoriser_nuria.py"),
]

PROMPT_TEMPLATE = """
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


def formater_duree(secondes):
    minutes, secondes = divmod(round(secondes), 60)
    if minutes:
        return f"{minutes} min {secondes} s"
    return f"{secondes} s"


class NuriaCoach:
    """Point d'entrée central pour le pipeline de données et le chatbot Nuria."""

    def __init__(self):
        # Charger la configuration depuis le fichier .env
        load_dotenv()

        # Vérifier qu'Ollama est lancé et que les modèles nécessaires sont disponibles
        self.ollama_client = ollama.Client()
        self._verifier_ollama()

        # Initialiser les embeddings et le LLM
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.3, num_ctx=8192, num_predict=900)

        # Initialiser la connexion ChromaDB et charger les données récentes
        self.vectorstore = None
        self.tous_les_docs = []
        self.activites_recentes = ""
        self.qa_chain = None
        self._charger_donnees()

    # ------------------------------------------------------------------
    # Initialisation / rechargement des données
    # ------------------------------------------------------------------

    def _verifier_ollama(self):
        """Vérifie qu'Ollama est lancé et que les modèles nécessaires sont disponibles."""
        try:
            reponse = self.ollama_client.list()
        except Exception:
            raise RuntimeError(
                "Ollama n'est pas lancé. Démarrez-le avec la commande 'ollama serve' puis réessayez."
            )

        modeles = [m.model for m in reponse.models]
        for modele in (LLM_MODEL, EMBEDDING_MODEL):
            if not any(m.startswith(modele) for m in modeles):
                raise RuntimeError(
                    f"Le modèle '{modele}' n'est pas disponible dans Ollama. "
                    f"Installez-le avec la commande : ollama pull {modele}"
                )

    def _charger_donnees(self):
        """(Re)charge la base vectorielle et les documents récents depuis le disque.

        Si aucune donnée n'a encore été synchronisée, laisse l'instance dans un état
        "vide" (vectorstore/qa_chain à None) plutôt que d'échouer, pour permettre
        un premier appel à sync().
        """
        if not PERSIST_DIRECTORY.exists() or not DOCS_PATH.exists():
            self.vectorstore = None
            self.tous_les_docs = []
            self.activites_recentes = ""
            self.qa_chain = None
            return

        try:
            self.vectorstore = Chroma(
                persist_directory=str(PERSIST_DIRECTORY),
                embedding_function=self.embeddings
            )
        except Exception as e:
            raise RuntimeError(f"Impossible de charger la base de données vectorielle : {e}")

        try:
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                self.tous_les_docs = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Le fichier {DOCS_PATH} est corrompu (JSON invalide) : {e}")

        activites = sorted(
            (d for d in self.tous_les_docs if d.get("type") == "activite"),
            key=lambda d: d["date"], reverse=True
        )
        self.activites_recentes = "\n\n".join(d["texte"] for d in activites[:NB_ACTIVITES_RECENTES])

        self.qa_chain = self._build_chain()

    def _build_chain(self):
        """Construit la chaîne RAG (recherche + prompt + LLM)."""
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["activites_recentes", "context", "question"]
        )

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2, "filter": {"type": "activite"}})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            {
                "activites_recentes": lambda _: self.activites_recentes,
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    # ------------------------------------------------------------------
    # Synchronisation des données (export -> transform -> vectorise)
    # ------------------------------------------------------------------

    def sync(self):
        """Exécute le pipeline de synchronisation (export, transformation, vectorisation)."""
        resultats = []

        for nom, script in ETAPES_SYNC:
            print(f"\n=== Étape : {nom} ({script}) ===")
            debut = time.time()
            resultat = subprocess.run([sys.executable, str(SRC_DIR / script)])
            duree = time.time() - debut

            if resultat.returncode != 0:
                print(f"\n❌ Échec de l'étape '{nom}' (code {resultat.returncode}) après {formater_duree(duree)}.")
                resultats.append({"etape": nom, "succes": False, "duree": duree})
                return {
                    "succes": False,
                    "etapes": resultats,
                    "duree_totale": sum(r["duree"] for r in resultats),
                }

            print(f"✅ Étape '{nom}' terminée avec succès en {formater_duree(duree)}.")
            resultats.append({"etape": nom, "succes": True, "duree": duree})

        # Recharger la base vectorielle et les données récentes après la synchronisation
        self._charger_donnees()

        return {
            "succes": True,
            "etapes": resultats,
            "duree_totale": sum(r["duree"] for r in resultats),
        }

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, question):
        """Génère la réponse à une question, en streaming (générateur de morceaux de texte)."""
        if self.qa_chain is None:
            raise RuntimeError(
                "Aucune donnée disponible. Lancez une synchronisation (sync) avant de discuter."
            )

        try:
            for morceau in self.qa_chain.stream(question):
                yield morceau
        except Exception as e:
            raise RuntimeError(
                f"Erreur lors de la génération de la réponse : {e}. "
                "Vérifiez qu'Ollama est toujours lancé (commande 'ollama serve')."
            )

    # ------------------------------------------------------------------
    # Statut
    # ------------------------------------------------------------------

    def status(self):
        """Retourne un dictionnaire décrivant l'état de Nuria (Ollama, base vectorielle, dernière sync)."""
        statut = {
            "ollama_ok": False,
            "chromadb_ok": False,
            "derniere_sync": None,
            "nb_activites": None,
        }

        try:
            self.ollama_client.list()
            statut["ollama_ok"] = True
        except Exception:
            statut["ollama_ok"] = False

        if self.vectorstore is not None:
            try:
                resultat = self.vectorstore.get(where={"type": "activite"})
                statut["chromadb_ok"] = True
                statut["nb_activites"] = len(resultat["ids"])
            except Exception:
                statut["chromadb_ok"] = False

        if DOCS_PATH.exists():
            timestamp = DOCS_PATH.stat().st_mtime
            statut["derniere_sync"] = datetime.fromtimestamp(timestamp)

        return statut
