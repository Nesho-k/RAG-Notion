"""
01_verify_dataset.py
--------------------
Étape 1 : Vérifier que le fichier JSON est bien formé et prêt pour le pipeline RAG.

POURQUOI cette étape ?
  Avant de lancer des calculs coûteux (embeddings, index FAISS), on s'assure que
  les données sont propres. Un problème détecté ici évite des erreurs cryptiques
  plus tard dans le pipeline.

CE QUE CE SCRIPT VÉRIFIE :
  1. Le fichier JSON existe et est lisible
  2. Chaque document possède les champs obligatoires : id, title, text, url
  3. Les champs ne sont pas vides
  4. La longueur des textes est compatible avec le modèle d'embedding
  5. Les identifiants sont uniques (pas de doublon)

UTILISATION :
  python 01_verify_dataset.py
"""

import json      # Module standard Python pour lire/écrire du JSON
import sys       # Module standard pour quitter le script proprement (sys.exit)
import os        # Module standard pour vérifier l'existence d'un fichier

# Dossier racine du projet (parent du dossier pipeline/)
# __file__ = chemin absolu de ce script
# dirname(__file__)        = pipeline/
# dirname(dirname(__file__)) = racine du projet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIGURATION
# ============================================================

# Chemin absolu vers le corpus — dans le dossier racine du projet
FICHIER_JSON = os.path.join(ROOT, "notion_rag_corpus.json")

# Champs obligatoires que chaque document doit posséder
CHAMPS_OBLIGATOIRES = ["id", "title", "section", "text", "source", "url"]

# Limites de longueur du texte en tokens (approximation : 1 mot ≈ 1.3 tokens)
# Le modèle paraphrase-multilingual-MiniLM-L12-v2 accepte max 512 tokens
TOKENS_MAX = 512
MOTS_MIN = 50    # Un chunk trop court manque de contexte
MOTS_MAX = 400   # Marge de sécurité avant la limite des 512 tokens


# ============================================================
# FONCTION PRINCIPALE DE VÉRIFICATION
# ============================================================

def verifier_dataset():
    """Charge le JSON et effectue toutes les vérifications."""

    # ---- 1. Vérifier que le fichier existe ----
    print(f"\n{'='*55}")
    print("  VÉRIFICATION DU DATASET NOTION RAG")
    print(f"{'='*55}\n")

    # os.path.exists retourne True si le fichier est trouvé sur le disque
    if not os.path.exists(FICHIER_JSON):
        # Message d'erreur explicite + arrêt propre du script
        print(f"[ERREUR] Fichier introuvable : '{FICHIER_JSON}'")
        print("  → Vérifiez que le fichier est dans le même dossier que ce script.")
        sys.exit(1)  # Code de sortie 1 = erreur (convention Unix/Windows)

    print(f"[OK] Fichier trouvé : {FICHIER_JSON}")

    # ---- 2. Charger et parser le JSON ----
    try:
        # open() ouvre le fichier ; encoding="utf-8" est crucial pour les accents français
        with open(FICHIER_JSON, "r", encoding="utf-8") as f:
            # json.load() lit le fichier et le convertit en liste Python
            documents = json.load(f)

    except json.JSONDecodeError as e:
        # json.JSONDecodeError est levée si le JSON est malformé (virgule manquante, etc.)
        print(f"[ERREUR] Le fichier JSON est malformé : {e}")
        print("  → Vérifiez la syntaxe du fichier (virgules, guillemets, accolades).")
        sys.exit(1)

    # Vérifier que le JSON contient bien une liste (et pas un dict ou autre)
    if not isinstance(documents, list):
        print("[ERREUR] Le JSON doit être une liste de documents (tableau JSON [...])")
        sys.exit(1)

    print(f"[OK] JSON valide — {len(documents)} documents chargés\n")

    # ---- 3. Vérifier chaque document ----
    erreurs = []       # Liste pour accumuler toutes les erreurs trouvées
    ids_vus = set()    # set() pour détecter les doublons d'identifiants
    stats_mots = []    # Pour afficher des statistiques sur les longueurs de texte

    for i, doc in enumerate(documents):
        # `i` est l'indice (0-based), `doc` est le dictionnaire du document courant
        doc_id = doc.get("id", f"[document #{i}]")  # Récupère l'id ou un label générique

        # -- Vérifier les champs obligatoires --
        for champ in CHAMPS_OBLIGATOIRES:
            if champ not in doc:
                erreurs.append(f"  Doc '{doc_id}' : champ '{champ}' MANQUANT")
            elif not doc[champ]:  # Vrai si la valeur est vide ("", None, 0...)
                erreurs.append(f"  Doc '{doc_id}' : champ '{champ}' est VIDE")

        # -- Vérifier l'unicité des identifiants --
        if doc_id in ids_vus:
            erreurs.append(f"  Identifiant en double : '{doc_id}'")
        ids_vus.add(doc_id)

        # -- Vérifier la longueur du texte --
        if "text" in doc and doc["text"]:
            # split() découpe sur les espaces → approximation du nombre de mots
            nb_mots = len(doc["text"].split())
            stats_mots.append((doc_id, nb_mots))

            if nb_mots < MOTS_MIN:
                erreurs.append(
                    f"  Doc '{doc_id}' : texte trop court ({nb_mots} mots, minimum {MOTS_MIN})"
                )
            elif nb_mots > MOTS_MAX:
                erreurs.append(
                    f"  Doc '{doc_id}' : texte trop long ({nb_mots} mots → risque de dépasser {TOKENS_MAX} tokens)"
                )

    # ---- 4. Afficher les résultats ----

    # Affichage des statistiques de longueur
    print("LONGUEUR DES TEXTES (en mots) :")
    print(f"  {'ID':<12} {'Mots':>6}  {'Jauge'}")
    print(f"  {'-'*12} {'-'*6}  {'-'*30}")
    for doc_id, nb_mots in stats_mots:
        # Barre de progression visuelle : 1 bloc = 10 mots
        barre = "█" * (nb_mots // 10)
        print(f"  {doc_id:<12} {nb_mots:>6}  {barre}")

    # Calcul min/max/moyenne si des textes ont été analysés
    if stats_mots:
        toutes_longueurs = [nb for _, nb in stats_mots]
        print(f"\n  → Min : {min(toutes_longueurs)} mots")
        print(f"  → Max : {max(toutes_longueurs)} mots")
        print(f"  → Moyenne : {sum(toutes_longueurs) // len(toutes_longueurs)} mots")

    # Affichage du bilan final
    print(f"\n{'='*55}")
    if erreurs:
        print(f"[ÉCHEC] {len(erreurs)} problème(s) détecté(s) :\n")
        for erreur in erreurs:
            print(erreur)
        print("\n  → Corrigez ces erreurs avant de continuer.")
        sys.exit(1)
    else:
        print("[SUCCÈS] Le dataset est valide et prêt pour les embeddings !")
        print(f"\n  {len(documents)} chunks détectés")
        print(f"  Taille compatible avec le modèle (max {TOKENS_MAX} tokens)")
        print(f"  Tous les champs obligatoires sont présents")
        print(f"\n  → Vous pouvez lancer : python 02_create_embeddings.py")
    print(f"{'='*55}\n")


# ============================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================

# Ce bloc s'exécute uniquement quand on lance le script directement
# (pas quand il est importé par un autre module — bonne pratique Python)
if __name__ == "__main__":
    verifier_dataset()
