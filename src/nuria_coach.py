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

from agregats_nuria import formater_resume_agregats

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIRECTORY = BASE_DIR / "db" / "nuria_db"
DOCS_PATH = DATA_DIR / "nuria_docs.json"
AGREGATS_PATH = DATA_DIR / "agregats.json"
HISTORIQUE_PATH = DATA_DIR / "historique.json"

LLM_MODEL = "qwen2.5:14b-instruct"
EMBEDDING_MODEL = "nomic-embed-text"

# Nombre d'échanges précédents conservés dans l'historique de conversation
NB_ECHANGES_HISTORIQUE = 5

# Nombre de séances brutes injectées dans le contexte (RAG)
K_CONTEXTE_DEFAUT = 2
K_CONTEXTE_SPECIFIQUE = 10

# Mots-clés indiquant une question portant sur une séance spécifique
# (plutôt qu'une tendance générale déjà couverte par les agrégats)
MOTS_CLES_SEANCE_SPECIFIQUE = [
    "plus longue", "plus long", "plus courte", "plus court",
    "plus rapide", "plus lente", "record", "meilleure", "pire",
    "le plus", "la plus", "dernière séance", "dernier entraînement",
    "dernière activité",
]

# Étapes du pipeline de synchronisation
ETAPES_SYNC = [
    ("Export des données Garmin", "export_nuria.py"),
    ("Transformation des données", "transformer_nuria.py"),
    ("Vectorisation des données", "vectoriser_nuria.py"),
    ("Calcul des agrégats", "agregats_nuria.py"),
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

Voici un résumé des statistiques d'entraînement de la personne (charge, progression, répartition des sports...) :
{agregats}

Voici les derniers échanges de la conversation, du plus ancien au plus récent :
{historique}

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
        self.agregats_texte = ""
        self.qa_chain = None
        self.historique = []
        self.historique_texte = ""
        self._charger_historique()
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
            self.agregats_texte = ""
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

        self.agregats_texte = self._charger_agregats()

        self.qa_chain = self._build_chain()

    def _charger_agregats(self):
        """Charge les agrégats pré-calculés et retourne leur résumé en français.

        Retourne une chaîne vide si le fichier n'existe pas encore (avant la
        première synchronisation incluant l'étape de calcul des agrégats).
        """
        if not AGREGATS_PATH.exists():
            return ""

        try:
            with open(AGREGATS_PATH, "r", encoding="utf-8") as f:
                agregats = json.load(f)
        except (json.JSONDecodeError, OSError):
            return ""

        return formater_resume_agregats(agregats)

    # ------------------------------------------------------------------
    # Historique de conversation
    # ------------------------------------------------------------------

    def _charger_historique(self):
        """Charge l'historique de conversation depuis le disque, ou repart à vide."""
        if not HISTORIQUE_PATH.exists():
            self.historique = []
        else:
            try:
                with open(HISTORIQUE_PATH, "r", encoding="utf-8") as f:
                    self.historique = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.historique = []

        self.historique_texte = self._formater_historique()

    def _sauvegarder_historique(self):
        """Sauvegarde l'historique de conversation sur le disque."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(HISTORIQUE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.historique, f, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"⚠️  Impossible de sauvegarder l'historique de conversation : {e}")

    def _formater_historique(self):
        """Construit le texte des derniers échanges, pour le prompt du chatbot."""
        derniers = self.historique[-NB_ECHANGES_HISTORIQUE:]
        if not derniers:
            return "(aucun échange précédent)"

        return "\n\n".join(
            f"Question : {echange['question']}\nRéponse : {echange['reponse']}"
            for echange in derniers
        )

    def enregistrer_echange(self, question, reponse):
        """Ajoute un échange à l'historique et le sauvegarde sur le disque."""
        self.historique.append({"question": question, "reponse": reponse})
        self.historique_texte = self._formater_historique()
        self._sauvegarder_historique()

    def effacer_historique(self):
        """Efface l'historique de conversation, en mémoire et sur le disque."""
        self.historique = []
        self.historique_texte = self._formater_historique()
        try:
            HISTORIQUE_PATH.unlink(missing_ok=True)
        except OSError as e:
            print(f"⚠️  Impossible de supprimer le fichier d'historique : {e}")

    def obtenir_historique(self, n=5):
        """Retourne les n derniers échanges de l'historique."""
        return self.historique[-n:]

    def _choisir_k(self, question):
        """Choisit le nombre de séances brutes à récupérer selon le type de question.

        Une question portant sur une séance spécifique (record, plus longue, dernière...)
        a besoin de plus de séances brutes en contexte qu'une question de tendance
        générale, déjà couverte par les agrégats.
        """
        question_minuscule = question.lower()
        if any(mot in question_minuscule for mot in MOTS_CLES_SEANCE_SPECIFIQUE):
            return K_CONTEXTE_SPECIFIQUE
        return K_CONTEXTE_DEFAUT

    def _recuperer_contexte(self, question):
        """Récupère les séances brutes les plus pertinentes pour la question."""
        k = self._choisir_k(question)
        docs = self.vectorstore.similarity_search(question, k=k, filter={"type": "activite"})
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """Construit la chaîne RAG (recherche + prompt + LLM)."""
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["agregats", "context", "historique", "question"]
        )

        return (
            {
                "agregats": lambda _: self.agregats_texte,
                "historique": lambda _: self.historique_texte,
                "context": RunnableLambda(self._recuperer_contexte),
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

        reponse_complete = []
        try:
            for morceau in self.qa_chain.stream(question):
                reponse_complete.append(morceau)
                yield morceau
        except Exception as e:
            raise RuntimeError(
                f"Erreur lors de la génération de la réponse : {e}. "
                "Vérifiez qu'Ollama est toujours lancé (commande 'ollama serve')."
            )

        self.enregistrer_echange(question, "".join(reponse_complete))

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
