"""
04_search.py
------------
Étape 4 : Moteur de recherche sémantique interactif.

POURQUOI cette étape séparée du pipeline final ?
  Tester la recherche SANS Gemini permet de valider que FAISS retrouve
  les bons documents AVANT d'introduire la génération de texte.
  Si la réponse finale est mauvaise, on saura que le problème vient
  de la génération (étape 5) et pas de la recherche (étape 4).
  C'est le principe de "débogage par isolation".

CE QUE CE SCRIPT FAIT :
  1. Charge l'index FAISS et les documents
  2. Charge le modèle d'embedding
  3. Boucle interactive : pose une question → affiche les 3 chunks les plus proches
  4. Affiche le score de similarité, le titre et un extrait du texte

UTILISATION :
  python 04_search.py
  Puis tapez vos questions en français. "quit" ou "exit" pour arrêter.
"""

import json      # Lecture des documents
import sys       # Arrêt propre
import os        # Vérification des fichiers
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Dossier racine du projet (parent du dossier pipeline/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIGURATION
# ============================================================

FICHIER_DOCUMENTS  = os.path.join(ROOT, "documents.json")
FICHIER_INDEX      = os.path.join(ROOT, "faiss_index.bin")
NOM_MODELE         = "paraphrase-multilingual-MiniLM-L12-v2"
NB_RESULTATS       = 3    # Nombre de chunks à retourner
LONGUEUR_EXTRAIT   = 200  # Nombre de caractères affichés de chaque chunk


# ============================================================
# CHARGEMENT (une seule fois au démarrage)
# ============================================================

def charger_ressources():
    """
    Charge les trois ressources nécessaires à la recherche.

    POURQUOI charger une seule fois ?
      Le modèle d'embedding (~117 Mo) et l'index FAISS sont coûteux
      à charger. On les charge une fois au démarrage, puis on les
      réutilise pour chaque question dans la boucle interactive.
    """

    # -- Vérification des fichiers requis --
    for fichier in [FICHIER_DOCUMENTS, FICHIER_INDEX]:
        if not os.path.exists(fichier):
            print(f"[ERREUR] Fichier introuvable : '{fichier}'")
            print("  → Vérifiez que les étapes 2 et 3 ont bien été exécutées.")
            sys.exit(1)

    # -- Chargement des documents --
    with open(FICHIER_DOCUMENTS, "r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"[OK] {len(documents)} documents chargés")

    # -- Chargement de l'index FAISS --
    # faiss.read_index() relit le fichier binaire produit par l'étape 3
    index = faiss.read_index(FICHIER_INDEX)
    print(f"[OK] Index FAISS chargé ({index.ntotal} vecteurs)")

    # -- Chargement du modèle d'embedding --
    # Depuis le cache local — instantané si déjà téléchargé à l'étape 2
    print(f"[...] Chargement du modèle d'embedding...")
    modele = SentenceTransformer(NOM_MODELE)
    print(f"[OK] Modèle chargé\n")

    return documents, index, modele


# ============================================================
# FONCTION DE RECHERCHE
# ============================================================

def rechercher(question, documents, index, modele, k=NB_RESULTATS):
    """
    Transforme une question en vecteur, puis interroge FAISS.

    Paramètres :
      question  : texte de la question en français
      documents : liste des documents avec métadonnées
      index     : index FAISS chargé
      modele    : modèle sentence-transformers chargé
      k         : nombre de résultats à retourner

    Retourne :
      Liste de tuples (document, score) triés par score décroissant.

    ÉTAPES INTERNES :
      1. encode() → vecteur de 384 dimensions
      2. normalize_L2() → même espace que les vecteurs indexés
      3. index.search() → k indices + scores de similarité cosinus
      4. Reconstruction des documents à partir des indices
    """

    # -- Encoder la question --
    # encode() retourne un tableau de forme (384,)
    # On lui donne la forme (1, 384) car FAISS attend un tableau 2D
    vecteur_question = modele.encode(
        [question],             # Liste avec un seul élément
        convert_to_numpy=True,
        show_progress_bar=False # Pas de barre dans la boucle interactive
    ).astype(np.float32)       # float32 obligatoire pour FAISS

    # -- Normaliser le vecteur de la question --
    # CRITIQUE : la question doit être dans le même espace vectoriel
    # que les documents. Si on oublie cette normalisation, les scores
    # seront faux et les résultats incohérents.
    faiss.normalize_L2(vecteur_question)

    # -- Recherche FAISS --
    # search() retourne :
    #   scores  : tableau (1, k) des scores de similarité cosinus
    #   indices : tableau (1, k) des positions dans l'index
    scores, indices = index.search(vecteur_question, k=k)

    # -- Reconstruction des résultats --
    # indices[0] et scores[0] : résultats pour la première (unique) requête
    resultats = []
    for idx, score in zip(indices[0], scores[0]):
        # idx = -1 si FAISS n'a pas trouvé assez de résultats (corpus trop petit)
        if idx == -1:
            continue
        # On retrouve le document grâce à son indice de position
        resultats.append((documents[idx], float(score)))

    return resultats


# ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================

def afficher_resultats(question, resultats):
    """
    Affiche les résultats de manière lisible dans le terminal.

    L'extrait de texte est tronqué à LONGUEUR_EXTRAIT caractères
    pour ne pas surcharger l'affichage — le texte complet sera
    utilisé dans l'étape 5 pour le contexte Gemini.
    """

    print(f"\n{'─'*55}")
    print(f"  Question : {question}")
    print(f"{'─'*55}")

    for rang, (doc, score) in enumerate(resultats, start=1):
        # Barre visuelle du score (score entre 0 et 1 → barre sur 20 blocs)
        nb_blocs = int(score * 20)
        barre = "█" * nb_blocs + "░" * (20 - nb_blocs)

        # Récupération sécurisée de l'URL (peut être absente)
        url = doc.get("url", "Source non disponible")

        # Extrait du texte (les premiers LONGUEUR_EXTRAIT caractères)
        extrait = doc["text"][:LONGUEUR_EXTRAIT].replace("\n", " ")
        if len(doc["text"]) > LONGUEUR_EXTRAIT:
            extrait += "..."

        print(f"\n  [{rang}] Score : {score:.4f}  {barre}")
        print(f"      ID      : {doc['id']}")
        print(f"      Titre   : {doc['title']}")
        print(f"      Section : {doc['section']}")
        print(f"      URL     : {url}")
        print(f"      Extrait : {extrait}")

    print(f"\n{'─'*55}\n")


# ============================================================
# BOUCLE INTERACTIVE
# ============================================================

def lancer_recherche_interactive(documents, index, modele):
    """
    Boucle principale : attend une question, affiche les résultats,
    recommence jusqu'à ce que l'utilisateur tape 'quit' ou 'exit'.
    """

    print("="*55)
    print("  MOTEUR DE RECHERCHE NOTION — MODE INTERACTIF")
    print("="*55)
    print("  Posez vos questions en français.")
    print("  Tapez 'quit' ou 'exit' pour quitter.\n")

    # Questions de test suggérées pour valider le pipeline
    print("  Questions de test recommandées :")
    print("    → Comment créer une page ?")
    print("    → Comment lier deux bases de données ?")
    print("    → Comment utiliser les raccourcis clavier ?")
    print()

    while True:
        try:
            # input() bloque jusqu'à ce que l'utilisateur appuie sur Entrée
            question = input("  Votre question : ").strip()

        except (KeyboardInterrupt, EOFError):
            # Ctrl+C ou Ctrl+D : sortie propre
            print("\n\n  Au revoir !")
            break

        # Commandes de sortie
        if question.lower() in ("quit", "exit", "q"):
            print("\n  Au revoir !")
            break

        # Ignorer les entrées vides
        if not question:
            print("  (question vide, réessayez)\n")
            continue

        # Lancer la recherche et afficher les résultats
        resultats = rechercher(question, documents, index, modele)
        afficher_resultats(question, resultats)


# ============================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  CHARGEMENT DES RESSOURCES...")
    print(f"{'='*55}\n")

    documents, index, modele = charger_ressources()
    lancer_recherche_interactive(documents, index, modele)
