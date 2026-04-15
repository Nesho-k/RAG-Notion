"""
03_build_faiss_index.py
-----------------------
Étape 3 : Construire l'index FAISS à partir des embeddings.

POURQUOI cette étape ?
  On a 13 vecteurs dans embeddings.npy. Pour trouver les plus proches
  d'une question, on pourrait comparer un à un — mais sur des millions
  de documents ce serait trop lent. FAISS construit une structure de données
  optimisée pour cette recherche. Sur 13 documents la différence est invisible,
  mais on adopte dès maintenant les bonnes pratiques.

QUEL TYPE D'INDEX FAISS ?
  On utilise IndexFlatIP (Inner Product = produit scalaire).

  Deux mesures de similarité existent :
    - Distance euclidienne (L2) : mesure "à vol d'oiseau" entre deux points.
    - Similarité cosinus        : mesure l'angle entre deux vecteurs.
                                  Insensible à la longueur des vecteurs.

  La similarité cosinus est meilleure pour le texte car deux phrases
  peuvent parler du même sujet avec des longueurs très différentes.

  ASTUCE : IndexFlatIP avec des vecteurs normalisés (norme = 1)
  calcule exactement la similarité cosinus. C'est l'approche recommandée
  par la documentation FAISS pour la recherche sémantique.

CE QUE CE SCRIPT PRODUIT :
  - faiss_index.bin : l'index FAISS sérialisé sur disque

UTILISATION :
  python 03_build_faiss_index.py
"""

import os       # Vérification des fichiers et taille
import sys      # Arrêt propre en cas d'erreur
import numpy as np   # Chargement de la matrice d'embeddings
import faiss         # Moteur de recherche vectorielle

# Dossier racine du projet (parent du dossier pipeline/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIGURATION
# ============================================================

FICHIER_EMBEDDINGS = os.path.join(ROOT, "embeddings.npy")    # Produit par l'étape 2
FICHIER_INDEX      = os.path.join(ROOT, "faiss_index.bin")   # Index FAISS à créer


# ============================================================
# ÉTAPE 1 : Charger les embeddings
# ============================================================

def charger_embeddings():
    """Charge la matrice numpy produite par l'étape 2."""

    if not os.path.exists(FICHIER_EMBEDDINGS):
        print(f"[ERREUR] Fichier introuvable : '{FICHIER_EMBEDDINGS}'")
        print("  → Lancez d'abord : python 02_create_embeddings.py")
        sys.exit(1)

    # np.load() lit le fichier binaire .npy et retourne un tableau numpy
    embeddings = np.load(FICHIER_EMBEDDINGS)

    print(f"[OK] Embeddings chargés — forme : {embeddings.shape}")
    print(f"     {embeddings.shape[0]} documents × {embeddings.shape[1]} dimensions")

    # FAISS exige des vecteurs en float32 (32 bits)
    # L'étape 2 produit déjà du float32, mais on s'en assure explicitement
    embeddings = embeddings.astype(np.float32)

    return embeddings


# ============================================================
# ÉTAPE 2 : Normaliser les vecteurs (L2)
# ============================================================

def normaliser(embeddings):
    """
    Normalise chaque vecteur pour que sa norme L2 soit égale à 1.

    POURQUOI normaliser ?
      IndexFlatIP calcule le produit scalaire (dot product).
      Si les vecteurs sont normalisés (norme = 1), le produit scalaire
      est mathématiquement équivalent à la similarité cosinus :

        cos(θ) = (A · B) / (‖A‖ × ‖B‖)
                                            ↑ vaut 1 si normalisé

      Résultat : les scores de similarité seront entre -1 et +1,
      où +1 = textes identiques, 0 = aucun rapport, -1 = opposés.

    faiss.normalize_L2() modifie le tableau EN PLACE (pas de copie).
    """

    faiss.normalize_L2(embeddings)  # Modifie directement le tableau numpy

    # Vérification : toutes les normes doivent être ~1.0
    normes = np.linalg.norm(embeddings, axis=1)
    print(f"\n[OK] Vecteurs normalisés")
    print(f"     Norme min : {normes.min():.6f} | max : {normes.max():.6f}")
    print(f"     (toutes les normes doivent être proches de 1.000000)")

    return embeddings


# ============================================================
# ÉTAPE 3 : Construire l'index FAISS
# ============================================================

def construire_index(embeddings):
    """
    Crée et remplit l'index FAISS.

    IndexFlatIP :
      - "Flat"  = index "naïf" : compare la requête avec TOUS les vecteurs.
                  Donne les résultats exacts (pas d'approximation).
                  Parfait pour moins de ~100 000 documents.
      - "IP"    = Inner Product (produit scalaire).
                  Avec des vecteurs normalisés → similarité cosinus exacte.

    Alternatives écartées :
      - IndexIVFFlat  : plus rapide sur millions de docs, mais approximatif
                        et nécessite un entraînement préalable. Inutile ici.
      - IndexHNSWFlat : graphe hiérarchique, très rapide, mais utilise plus
                        de RAM. Surdimensionné pour 13 documents.
    """

    # Récupère le nombre de dimensions (384 pour notre modèle)
    dimension = embeddings.shape[1]

    # Crée l'index vide (il ne contient encore aucun vecteur)
    index = faiss.IndexFlatIP(dimension)

    # Ajoute tous les vecteurs dans l'index
    # add() accepte un tableau numpy float32 de forme (n, dimension)
    index.add(embeddings)

    print(f"\n[OK] Index FAISS construit")
    print(f"     Type   : IndexFlatIP (similarité cosinus exacte)")
    print(f"     Dimension : {dimension}")
    # ntotal = nombre de vecteurs stockés dans l'index
    print(f"     Vecteurs indexés : {index.ntotal}")

    return index


# ============================================================
# ÉTAPE 4 : Sauvegarder l'index sur disque
# ============================================================

def sauvegarder_index(index):
    """
    Écrit l'index FAISS dans un fichier binaire.

    POURQUOI sauvegarder ?
      Construire l'index prend du temps sur de grands corpus.
      En le sauvegardant, les étapes 4 et 5 le chargent en ~1ms
      sans avoir à le reconstruire.

    faiss.write_index() sérialise l'index dans un format binaire propriétaire.
    faiss.read_index()  le recharge (utilisé dans les étapes suivantes).
    """

    faiss.write_index(index, FICHIER_INDEX)

    taille = os.path.getsize(FICHIER_INDEX) / 1024  # Taille en Ko
    print(f"\n[OK] Index sauvegardé → '{FICHIER_INDEX}' ({taille:.1f} Ko)")


# ============================================================
# ÉTAPE 5 : Test de recherche rapide
# ============================================================

def tester_index(index, embeddings):
    """
    Effectue une recherche test pour vérifier que l'index fonctionne.

    On prend le premier vecteur (doc_001) et on cherche ses 3 voisins.
    Le résultat attendu : doc_001 lui-même en position 0 (score = 1.0).
    Si ce n'est pas le cas, l'index est corrompu.
    """

    print(f"\n{'='*55}")
    print("  TEST DE RECHERCHE (auto-vérification)")
    print(f"{'='*55}")
    print("  Requête : vecteur du doc_001 (Créer une page)")
    print("  Résultat attendu : doc_001 en position 1 avec score ≈ 1.0\n")

    # Prend le vecteur du premier document et lui donne la forme (1, 384)
    # FAISS attend toujours un tableau 2D même pour une seule requête
    vecteur_test = embeddings[0:1]  # slice [0:1] donne (1, 384), pas (384,)

    # search() retourne deux tableaux numpy :
    #   - scores  : les scores de similarité, forme (1, k)
    #   - indices : les positions dans l'index, forme (1, k)
    # k=3 : on demande les 3 voisins les plus proches
    scores, indices = index.search(vecteur_test, k=3)

    # scores[0] et indices[0] = résultats pour la première (et unique) requête
    for rang, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        print(f"  Rang {rang} : indice={idx}  score={score:.4f}")

    # Vérification : l'indice 0 doit être le premier résultat avec score=1.0
    if indices[0][0] == 0 and abs(scores[0][0] - 1.0) < 0.001:
        print(f"\n[OK] Test réussi — l'index retourne bien le document exact")
    else:
        print(f"\n[ALERTE] Résultat inattendu — vérifiez la normalisation")

    print(f"\n{'='*55}")
    print("[SUCCÈS] Index FAISS prêt !")
    print(f"\n  → Vous pouvez lancer : python 04_search.py")
    print(f"{'='*55}\n")


# ============================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  CONSTRUCTION DE L'INDEX FAISS — NOTION RAG")
    print(f"{'='*55}\n")

    embeddings = charger_embeddings()
    embeddings = normaliser(embeddings)
    index      = construire_index(embeddings)
    sauvegarder_index(index)
    tester_index(index, embeddings)
