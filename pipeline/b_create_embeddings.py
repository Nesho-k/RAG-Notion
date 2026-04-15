"""
02_create_embeddings.py
-----------------------
Étape 2 : Transformer les textes en vecteurs numériques (embeddings).

POURQUOI cette étape ?
  FAISS ne comprend pas le texte — il ne travaille qu'avec des nombres.
  Un embedding est la "traduction" d'un texte en un vecteur de 384 dimensions
  (384 nombres décimaux). Des textes proches sémantiquement auront des vecteurs
  proches dans cet espace mathématique. C'est le cœur du RAG.

POURQUOI ce modèle : paraphrase-multilingual-MiniLM-L12-v2 ?
  - "multilingual" : entraîné sur 50+ langues dont le français — il comprend
    les nuances du français sans traduction préalable.
  - "MiniLM-L12" : version légère (117 Mo) mais très performante, idéale pour
    tourner sur CPU sans GPU.
  - "paraphrase" : optimisé pour mesurer la similarité sémantique entre phrases,
    ce qui est exactement notre cas d'usage (question ↔ document).
  - Alternative écartée : "text-embedding-ada-002" (OpenAI) = payant et nécessite
    internet à chaque requête. Notre modèle tourne 100% en local.

CE QUE CE SCRIPT PRODUIT :
  - embeddings.npy  : matrice numpy de forme (13, 384) — les 13 vecteurs
  - documents.json  : copie du corpus (pour que les étapes suivantes
                      puissent retrouver le texte et les métadonnées)

UTILISATION :
  python 02_create_embeddings.py
"""

import json          # Lecture du corpus et sauvegarde des documents
import sys           # Arrêt propre en cas d'erreur
import os            # Vérification de l'existence des fichiers
import numpy as np   # Sauvegarde de la matrice d'embeddings au format .npy

# sentence_transformers : bibliothèque qui encapsule le modèle Hugging Face
# SentenceTransformer est la classe principale pour charger et utiliser un modèle
from sentence_transformers import SentenceTransformer

# Dossier racine du projet (parent du dossier pipeline/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIGURATION
# ============================================================

FICHIER_JSON        = os.path.join(ROOT, "notion_rag_corpus.json")    # Corpus original
FICHIER_NOUVEAUX    = os.path.join(ROOT, "notion_rag_new_docs.json")  # Nouveaux docs (optionnel)
FICHIER_EMBEDDINGS  = os.path.join(ROOT, "embeddings.npy")            # Sortie : matrice numpy
FICHIER_DOCUMENTS   = os.path.join(ROOT, "documents.json")            # Sortie : documents enrichis

# Nom du modèle Hugging Face — téléchargé automatiquement au premier lancement
# et mis en cache dans ~/.cache/huggingface/ pour les prochaines exécutions
NOM_MODELE = "paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================
# ÉTAPE 1 : Charger le corpus
# ============================================================

def charger_corpus():
    """
    Charge le corpus original et, s'il existe, fusionne avec les nouveaux documents.
    Le fichier original n'est jamais modifié.
    """

    if not os.path.exists(FICHIER_JSON):
        print(f"[ERREUR] Fichier introuvable : '{FICHIER_JSON}'")
        sys.exit(1)

    with open(FICHIER_JSON, "r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"[OK] Corpus original — {len(documents)} documents")

    # Fusion désactivée — seul notion_rag_corpus.json est utilisé
    # Pour réactiver la fusion avec les nouveaux docs, décommenter le bloc ci-dessous :
    # if os.path.exists(FICHIER_NOUVEAUX):
    #     with open(FICHIER_NOUVEAUX, "r", encoding="utf-8") as f:
    #         nouveaux = json.load(f)
    #     documents = documents + nouveaux
    #     print(f"[OK] + {len(nouveaux)} nouveau(x) document(s) fusionné(s)")

    print(f"[OK] Total : {len(documents)} documents à vectoriser")
    return documents


# ============================================================
# ÉTAPE 2 : Charger le modèle d'embedding
# ============================================================

def charger_modele():
    """
    Charge le modèle sentence-transformers.

    Au premier lancement : télécharge ~117 Mo depuis Hugging Face Hub.
    Aux lancements suivants : charge depuis le cache local (instantané).
    """

    print(f"\n[...] Chargement du modèle '{NOM_MODELE}'...")
    print("      (premier lancement = téléchargement ~117 Mo, patientez)\n")

    # SentenceTransformer télécharge et met en cache le modèle automatiquement
    modele = SentenceTransformer(NOM_MODELE)

    # Afficher la dimension des vecteurs produits par ce modèle
    # get_sentence_embedding_dimension() retourne 384 pour MiniLM-L12
    dimension = modele.get_sentence_embedding_dimension()
    print(f"[OK] Modèle chargé — dimension des vecteurs : {dimension}")

    return modele


# ============================================================
# ÉTAPE 3 : Calculer les embeddings
# ============================================================

def creer_embeddings(modele, documents):
    """
    Transforme les textes en vecteurs numériques.

    POURQUOI encode uniquement le champ 'text' et pas 'title' ?
      Le titre seul manque de contexte sémantique. "Relations et rollups"
      ne dit pas grand-chose à la recherche vectorielle. Le texte complet
      contient les mots clés et les concepts que l'utilisateur va chercher.

    POURQUOI show_progress_bar=True ?
      Sur 13 documents c'est instantané, mais c'est une bonne habitude :
      sur des milliers de documents, la barre de progression est essentielle.

    POURQUOI convert_to_numpy=True ?
      FAISS travaille exclusivement avec des tableaux numpy float32.
      Cette option évite une conversion manuelle ultérieure.
    """

    print("\n[...] Calcul des embeddings en cours...")

    # Concaténer le titre et le texte pour la vectorisation
    # Inclure le titre améliore le rappel pour les requêtes courtes :
    # ex. "rollups" → "Relations et rollups\n[texte...]" score plus haut
    textes = [f"{doc['title']}\n{doc['text']}" for doc in documents]

    # encode() transforme la liste de textes en une matrice numpy
    # Forme de sortie : (nb_documents, dimension) = (13, 384)
    embeddings = modele.encode(
        textes,
        show_progress_bar=True,   # Affiche une barre de progression dans le terminal
        convert_to_numpy=True,    # Retourne un tableau numpy (requis par FAISS)
        batch_size=8,             # Traite 8 textes à la fois (optimal pour CPU)
    )

    # Vérification de la forme de la matrice produite
    print(f"\n[OK] Embeddings calculés")
    print(f"     Forme de la matrice : {embeddings.shape}")
    print(f"     → {embeddings.shape[0]} documents × {embeddings.shape[1]} dimensions")

    return embeddings


# ============================================================
# ÉTAPE 4 : Sauvegarder les résultats
# ============================================================

def sauvegarder_resultats(embeddings, documents):
    """
    Sauvegarde deux fichiers nécessaires aux étapes suivantes :

    1. embeddings.npy : la matrice numpy des vecteurs.
       Format .npy = format binaire numpy, lecture ultra-rapide.
       Alternative écartée : CSV → trop lent et perd la précision float32.

    2. documents.json : le corpus avec un index numérique ajouté.
       L'index permet à FAISS de retrouver le bon document à partir
       de sa position dans la matrice (FAISS retourne des indices, pas des textes).
    """

    # -- Sauvegarder la matrice d'embeddings --
    # np.save() écrit un fichier binaire .npy (efficace et fidèle)
    np.save(FICHIER_EMBEDDINGS, embeddings)
    taille_fichier = os.path.getsize(FICHIER_EMBEDDINGS) / 1024  # Taille en Ko
    print(f"\n[OK] Embeddings sauvegardés → '{FICHIER_EMBEDDINGS}' ({taille_fichier:.1f} Ko)")

    # -- Enrichir les documents avec leur indice numérique --
    # FAISS retourne un indice entier (0, 1, 2...) lors d'une recherche.
    # On ajoute cet indice à chaque document pour faire le lien plus facilement.
    documents_indexes = []
    for i, doc in enumerate(documents):
        doc_enrichi = dict(doc)   # Copie du dictionnaire original
        doc_enrichi["index"] = i  # Ajout de l'indice de position
        documents_indexes.append(doc_enrichi)

    # Sauvegarder les documents enrichis
    with open(FICHIER_DOCUMENTS, "w", encoding="utf-8") as f:
        # indent=2 : JSON indenté (lisible par un humain si besoin)
        # ensure_ascii=False : préserve les accents français (é, è, à, etc.)
        json.dump(documents_indexes, f, ensure_ascii=False, indent=2)

    print(f"[OK] Documents sauvegardés  → '{FICHIER_DOCUMENTS}'")


# ============================================================
# ÉTAPE 5 : Contrôle qualité (vérification visuelle)
# ============================================================

def afficher_verification(embeddings, documents):
    """
    Affiche un aperçu des embeddings pour vérifier visuellement
    que tout s'est bien passé.

    POURQUOI vérifier manuellement ?
      Un embedding entièrement à zéro ou avec des NaN (Not a Number) indiquerait
      un problème de calcul silencieux. Mieux vaut le détecter maintenant.
    """

    print(f"\n{'='*55}")
    print("  CONTRÔLE QUALITÉ DES EMBEDDINGS")
    print(f"{'='*55}")

    for i, (doc, vecteur) in enumerate(zip(documents, embeddings)):
        # Statistiques de base sur le vecteur
        val_min  = vecteur.min()
        val_max  = vecteur.max()
        val_mean = vecteur.mean()
        has_nan  = bool(np.isnan(vecteur).any())  # Vérifie les valeurs invalides

        statut = "[ALERTE NaN!]" if has_nan else "[OK]"

        print(f"  {statut} {doc['id']} | '{doc['title'][:35]:<35}' "
              f"| min={val_min:+.3f} max={val_max:+.3f} moy={val_mean:+.4f}")

    # Norme L2 moyenne — indique si les vecteurs sont bien distribués
    # Une norme ~1.0 est signe que les vecteurs sont bien formés
    normes = np.linalg.norm(embeddings, axis=1)
    print(f"\n  Norme L2 moyenne des vecteurs : {normes.mean():.4f}")
    print(f"  (idéalement proche de 1.0 pour une recherche cosinus précise)")

    print(f"\n{'='*55}")
    print("[SUCCÈS] Embeddings valides et prêts pour FAISS !")
    print(f"\n  → Vous pouvez lancer : python 03_build_faiss_index.py")
    print(f"{'='*55}\n")


# ============================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  CRÉATION DES EMBEDDINGS — NOTION RAG")
    print(f"{'='*55}")

    # On enchaîne les étapes dans l'ordre logique
    documents  = charger_corpus()
    modele     = charger_modele()
    embeddings = creer_embeddings(modele, documents)
    sauvegarder_resultats(embeddings, documents)
    afficher_verification(embeddings, documents)
