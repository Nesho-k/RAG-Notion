"""
05_rag_pipeline.py
------------------
Étape 5 : Pipeline RAG complet — Recherche + Génération avec Gemini 2.5 Flash.

POURQUOI RAG (Retrieval-Augmented Generation) ?
  Un LLM comme Gemini a été entraîné sur des données générales jusqu'à une
  certaine date. Il ne connaît pas votre documentation spécifique, vos données
  internes, ni les mises à jour récentes. RAG résout ça :

    [Question]
        ↓
    [Recherche FAISS] → 3 chunks pertinents
        ↓
    [Prompt = Question + Chunks] → envoyé à Gemini
        ↓
    [Réponse ancrée dans VOS données]

  Avantage clé : Gemini ne "hallucine" pas sur votre domaine car on lui fournit
  le contexte exact. S'il ne sait pas, il le dit.

CONFIGURATION REQUISE :
  Variable d'environnement GEMINI_API_KEY doit être définie.
  Sous Windows (Git Bash) :
    export GEMINI_API_KEY="votre_clé_ici"
  Ou dans un fichier .env (voir commentaire dans le code).

UTILISATION :
  python 05_rag_pipeline.py
"""

import json
import sys
import os
import time      # Pour le délai entre les tentatives (erreur 429)
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai   # SDK officiel Google Gemini
from dotenv import load_dotenv        # Charge les variables du fichier .env

# Dossier racine du projet (parent du dossier pipeline/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Charge le fichier .env depuis la racine du projet
# override=False : si la variable existe déjà dans le shell, elle n'est pas écrasée
load_dotenv(dotenv_path=os.path.join(ROOT, ".env"), override=False)

# ============================================================
# CONFIGURATION
# ============================================================

FICHIER_DOCUMENTS  = os.path.join(ROOT, "documents.json")
FICHIER_INDEX      = os.path.join(ROOT, "faiss_index.bin")
NOM_MODELE         = "paraphrase-multilingual-MiniLM-L12-v2"
NOM_MODELE_GEMINI  = "gemini-2.5-flash"   # Modèle Gemini à utiliser
NB_CHUNKS          = 5                     # Chunks récupérés par FAISS (robustesse aux fautes)
NB_CHUNKS_AFFICHES = 3                     # Chunks affichés comme sources dans le terminal
SCORE_MIN          = 0.30                  # Score minimum pour qu'un chunk soit utilisé
MAX_TENTATIVES     = 3                     # Tentatives max si erreur 429
DELAI_RETRY        = 10                    # Secondes d'attente entre tentatives

# Template du prompt RAG — exactement celui défini dans le cahier des charges
PROMPT_TEMPLATE = """Tu es un assistant expert sur Notion.
Réponds en français uniquement en te basant sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement
sans inventer d'information.

CONTEXTE :
{contexte}

QUESTION : {question}

RÉPONSE (en français) :"""


# ============================================================
# INITIALISATION DE L'API GEMINI
# ============================================================

def initialiser_gemini():
    """
    Configure le SDK Gemini avec la clé API depuis les variables d'environnement.

    POURQUOI os.environ et pas la clé en dur ?
      Écrire la clé dans le code = risque de la publier sur GitHub par erreur.
      os.environ.get() lit la variable d'environnement définie dans le shell,
      sans jamais toucher au fichier source.

    COMMENT définir la variable (à faire une seule fois dans le terminal) :
      Windows Git Bash : export GEMINI_API_KEY="AIza..."
      Windows cmd      : set GEMINI_API_KEY=AIza...
      PowerShell       : $env:GEMINI_API_KEY="AIza..."

    Pour une solution permanente, créez un fichier .env et utilisez
    la bibliothèque python-dotenv (non incluse ici pour rester simple).
    """

    # os.environ.get() retourne None si la variable n'existe pas
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("[ERREUR] Variable d'environnement GEMINI_API_KEY non définie.")
        print()
        print("  Définissez-la avant de lancer le script :")
        print('  Windows Git Bash : export GEMINI_API_KEY="votre_clé"')
        print("  Windows cmd      : set GEMINI_API_KEY=votre_clé")
        print("  PowerShell       : $env:GEMINI_API_KEY=\"votre_clé\"")
        print()
        print("  Obtenez une clé gratuite sur : https://aistudio.google.com/apikey")
        sys.exit(1)

    # configure() injecte la clé dans toutes les requêtes suivantes du SDK
    genai.configure(api_key=api_key)
    print(f"[OK] API Gemini configurée (modèle : {NOM_MODELE_GEMINI})")


# ============================================================
# CHARGEMENT DES RESSOURCES (identique à l'étape 4)
# ============================================================

def charger_ressources():
    """Charge les documents, l'index FAISS et le modèle d'embedding."""

    for fichier in [FICHIER_DOCUMENTS, FICHIER_INDEX]:
        if not os.path.exists(fichier):
            print(f"[ERREUR] Fichier introuvable : '{fichier}'")
            print("  → Vérifiez que les étapes 2 et 3 ont bien été exécutées.")
            sys.exit(1)

    with open(FICHIER_DOCUMENTS, "r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"[OK] {len(documents)} documents chargés")

    index = faiss.read_index(FICHIER_INDEX)
    print(f"[OK] Index FAISS chargé ({index.ntotal} vecteurs)")

    print(f"[...] Chargement du modèle d'embedding...")
    modele = SentenceTransformer(NOM_MODELE)
    print(f"[OK] Modèle chargé\n")

    return documents, index, modele


# ============================================================
# RECHERCHE FAISS (identique à l'étape 4)
# ============================================================

def rechercher_chunks(question, documents, index, modele, k=NB_CHUNKS):
    """Encode la question et retourne les k chunks les plus proches."""

    vecteur = modele.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype(np.float32)

    faiss.normalize_L2(vecteur)

    scores, indices = index.search(vecteur, k=k)

    resultats = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        # On ignore les chunks dont le score est en dessous du seuil minimum
        # Évite d'envoyer du bruit à Gemini quand la question est hors corpus
        if float(score) < SCORE_MIN:
            continue
        resultats.append((documents[idx], float(score)))

    return resultats


# ============================================================
# CONSTRUCTION DU CONTEXTE
# ============================================================

def construire_contexte(chunks):
    """
    Assemble les chunks en un bloc de texte structuré pour le prompt.

    FORMAT pour chaque chunk :
      Titre : [titre du document]
      [texte complet]

    POURQUOI inclure le titre ?
      Gemini peut ainsi citer sa source dans la réponse si nécessaire,
      et le texte est mieux ancré dans son contexte.

    Les chunks sont séparés par une ligne vide pour que Gemini les
    distingue clairement comme des sources indépendantes.
    """

    parties = []
    for doc, score in chunks:
        # On ajoute le titre comme en-tête du chunk
        partie = f"Titre : {doc['title']}\n{doc['text']}"
        parties.append(partie)

    # join avec double saut de ligne = séparation visuelle claire entre chunks
    return "\n\n".join(parties)


# ============================================================
# APPEL À GEMINI AVEC GESTION DES ERREURS
# ============================================================

def appeler_gemini(prompt):
    """
    Envoie le prompt à Gemini et retourne la réponse texte.

    GESTION DE L'ERREUR 429 (quota dépassé) :
      L'API Gemini gratuite a des limites de requêtes par minute.
      Si on dépasse le quota, l'API retourne une erreur 429.
      On attend DELAI_RETRY secondes et on réessaie, jusqu'à
      MAX_TENTATIVES fois.

    POURQUOI 3 tentatives et 10 secondes ?
      Le quota gratuit de Gemini se renouvelle par minute.
      10 secondes d'attente + 3 essais couvrent la plupart des cas.
    """

    # GenerativeModel instancie le modèle — l'objet est léger, pas de téléchargement
    model = genai.GenerativeModel(NOM_MODELE_GEMINI)

    for tentative in range(1, MAX_TENTATIVES + 1):
        try:
            # generate_content() envoie le prompt et retourne la réponse
            reponse = model.generate_content(prompt)

            # .text extrait le contenu textuel de la réponse
            return reponse.text

        except Exception as e:
            message_erreur = str(e)

            # Détection de l'erreur 429 dans le message d'exception
            if "429" in message_erreur or "quota" in message_erreur.lower() or "rate" in message_erreur.lower():
                if tentative < MAX_TENTATIVES:
                    print(f"\n  [QUOTA] Limite API atteinte (tentative {tentative}/{MAX_TENTATIVES}).")
                    print(f"  Attente de {DELAI_RETRY} secondes avant de réessayer...")
                    time.sleep(DELAI_RETRY)   # Pause avant nouvelle tentative
                else:
                    print(f"\n  [ERREUR] Quota API dépassé après {MAX_TENTATIVES} tentatives.")
                    print("  Attendez quelques minutes et relancez votre question.")
                    return None
            else:
                # Autre type d'erreur : on arrête immédiatement
                print(f"\n  [ERREUR] Problème avec l'API Gemini : {message_erreur}")
                return None

    return None


# ============================================================
# AFFICHAGE DE LA RÉPONSE ET DES SOURCES
# ============================================================

def afficher_reponse(question, reponse, chunks):
    """
    Affiche la réponse Gemini suivie des sources utilisées.

    POURQUOI toujours afficher les sources ?
      - Transparence : l'utilisateur sait sur quoi la réponse est basée.
      - Vérification : il peut aller lire la documentation officielle.
      - Confiance : une réponse sourcée est plus crédible qu'une réponse seule.
    """

    print(f"\n{'='*55}")
    print("  RÉPONSE")
    print(f"{'='*55}\n")

    # Affichage de la réponse Gemini
    print(reponse)

    # Affichage des sources
    print(f"\n{'─'*55}")
    print("  SOURCES UTILISÉES\n")

    # On n'affiche que les NB_CHUNKS_AFFICHES premières sources
    # (Gemini a reçu NB_CHUNKS chunks pour plus de robustesse, mais on n'affiche que les plus pertinents)
    for i, (doc, score) in enumerate(chunks[:NB_CHUNKS_AFFICHES], start=1):
        titre = doc.get("title", "Titre non disponible")
        # Récupération sécurisée de l'URL — affiche un message si absente
        url   = doc.get("url", "Source non disponible")

        print(f"  [{i}] {titre}")
        print(f"       {url}")
        print(f"       (pertinence : {score:.2%})")   # Score en pourcentage

    print(f"{'─'*55}\n")


# ============================================================
# BOUCLE INTERACTIVE PRINCIPALE
# ============================================================

def lancer_pipeline(documents, index, modele):
    """
    Boucle principale du pipeline RAG :
      1. Saisie de la question
      2. Recherche FAISS → 3 chunks
      3. Construction du contexte
      4. Construction du prompt
      5. Appel Gemini
      6. Affichage réponse + sources
    """

    print("="*55)
    print("  ASSISTANT NOTION — PIPELINE RAG COMPLET")
    print("="*55)
    print("  Posez vos questions sur Notion en français.")
    print("  Tapez 'quit' ou 'exit' pour quitter.\n")

    while True:
        try:
            question = input("  Votre question : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Au revoir !")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("\n  Au revoir !")
            break

        if not question:
            print("  (question vide, réessayez)\n")
            continue

        # -- Étape A : Recherche des chunks pertinents --
        print(f"\n  [1/3] Recherche des documents pertinents...")
        chunks = rechercher_chunks(question, documents, index, modele)

        # Si aucun chunk ne dépasse le seuil SCORE_MIN, la question est hors corpus
        if not chunks:
            print(f"\n  Aucun document pertinent trouvé (seuil : {SCORE_MIN:.0%}).")
            print(f"  Cette question semble hors du corpus Notion.\n")
            continue

        # -- Étape B : Construction du contexte et du prompt --
        print(f"  [2/3] Construction du prompt...")
        contexte = construire_contexte(chunks)

        # Insertion du contexte et de la question dans le template
        prompt = PROMPT_TEMPLATE.format(
            contexte=contexte,
            question=question
        )

        # -- Étape C : Appel à Gemini --
        print(f"  [3/3] Génération de la réponse (Gemini {NOM_MODELE_GEMINI})...")
        reponse = appeler_gemini(prompt)

        # Si Gemini a échoué, on passe à la question suivante
        if reponse is None:
            print("  Impossible de générer une réponse. Réessayez.\n")
            continue

        # -- Affichage final : réponse + sources --
        afficher_reponse(question, reponse, chunks)


# ============================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  DÉMARRAGE DU PIPELINE RAG NOTION")
    print(f"{'='*55}\n")

    initialiser_gemini()
    documents, index, modele = charger_ressources()
    lancer_pipeline(documents, index, modele)
