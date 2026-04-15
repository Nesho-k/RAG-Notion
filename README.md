# Assistant RAG Notion — Pipeline NLP 

Projet de NLP appliqué à la documentation Notion : assistant question-réponse en français basé sur RAG (Retrieval-Augmented Generation), avec pipeline d'embeddings vectoriels FAISS, génération via Gemini 2.5 Flash, et interface web Streamlit.

---

## Contexte

**Point de départ** : la documentation officielle de Notion (help.notion.com), couvrant les fonctionnalités clés de l'outil : pages, blocs, bases de données, collaboration, navigation, templates, raccourcis clavier et fonctionnalités avancées.

**Problème** : les LLMs généralistes comme Gemini ne connaissent pas les spécificités de la documentation interne, ni les détails techniques d'un produit. Répondre directement à "comment faire un rollup dans Notion ?" sans contexte génère des hallucinations ou des réponses incomplètes.

**Solution RAG** : avant chaque génération, on interroge une base vectorielle FAISS pour retrouver les chunks les plus pertinents, puis on les injecte dans le prompt. Gemini ne répond qu'à partir de ce contexte — **aucune invention possible**.

---

## Compétences démontrées

| Compétence | Ce qui est fait dans ce projet |
|---|---|
| **NLP & Embeddings** | Vectorisation des documents avec `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, 50+ langues) |
| **Recherche vectorielle** | Index FAISS (IndexFlatIP) avec normalisation L2 pour similarité cosinus |
| **Pipeline RAG** | Recherche des k chunks les plus proches → injection dans le prompt → génération Gemini |
| **Scraping & enrichissement** | Scraping des pages Notion avec BeautifulSoup, résumés générés par GPT-4o-mini |
| **Interface web** | App Streamlit avec historique de session, suggestions cliquables, affichage des sources |
| **Déploiement Cloud** | Déploiement sur Render avec gestion des variables d'environnement |

---

## Architecture

Le pipeline part de la documentation Notion et suit cinq étapes séquentielles.

Les pages sont d'abord scrapées via `enrich_corpus.py` : chaque URL est téléchargée, le texte est extrait avec BeautifulSoup, puis GPT-4o-mini génère un résumé structuré de 150-250 mots en préservant les termes techniques (rollup, filtre, vue, propriété...).

Les documents sont ensuite vectorisés par `b_create_embeddings.py` : le titre et le texte de chaque document sont concaténés puis encodés par le modèle sentence-transformers en vecteurs de 384 dimensions. L'index FAISS est construit par `c_build_faiss_index.py` avec normalisation L2.

À chaque question, `e_rag_pipeline.py` encode la question, interroge FAISS pour récupérer les 5 chunks les plus proches, filtre ceux dont le score est inférieur à 0.30, assemble le contexte et appelle Gemini. Si aucun chunk ne dépasse le seuil, la réponse est refusée sans hallucination.

---

## NLP : détail des choix techniques

### Modèle d'embedding

Le modèle `paraphrase-multilingual-MiniLM-L12-v2` a été choisi pour trois raisons :
- **Multilingue** : entraîné sur 50+ langues dont le français — comprend les nuances sans traduction préalable
- **Léger** : 117 Mo, tourne sur CPU sans GPU requis
- **Optimisé pour la similarité sémantique** : variante "paraphrase" conçue exactement pour mesurer la proximité entre une question et un document

Alternative écartée : `text-embedding-ada-002` (OpenAI) — payant et dépendant d'une connexion à chaque requête.

### Similarité cosinus via FAISS

| Paramètre | Valeur | Pourquoi |
|---|---|---|
| Index | `IndexFlatIP` | Produit scalaire = cosinus après normalisation L2 |
| Normalisation | L2 sur questions et documents | Espace vectoriel cohérent |
| k chunks envoyés à Gemini | 5 | Robustesse aux reformulations |
| Chunks affichés à l'utilisateur | 3 | Clarté de l'interface |
| Seuil minimum | 0.30 | Filtre les questions hors corpus |

### Stratégie de vectorisation

Le texte vectorisé concatène le titre et le contenu du document :

```
Relations et rollups
Les relations permettent de connecter deux bases de données...
```

Inclure le titre améliore le rappel pour les requêtes courtes — sans cela, "comment utiliser les rollups ?" ne retrouve pas le bon document car le mot "rollups" n'apparaît qu'au deuxième paragraphe.

### Gestion des questions hors corpus

Un seuil `SCORE_MIN = 0.30` filtre les questions sans réponse dans le corpus. Exemple : "c'est quoi le prix de YouTube Music ?" → score max 0.15 → refus propre sans hallucination.

---

## Stack technique

| Couche | Technologie |
|---|---|
| Embeddings | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Recherche vectorielle | FAISS (faiss-cpu) |
| Génération | Google Gemini 2.5 Flash |
| Enrichissement corpus | OpenAI GPT-4o-mini |
| Scraping | requests + BeautifulSoup4 |
| Interface | Streamlit |
| Déploiement | Render |

---

## Corpus

- **Source** : documentation officielle Notion (help.notion.com)
- **Volume** : 13 documents couvrant les fonctionnalités clés
- **Thèmes** : pages & blocs, bases de données, relations & rollups, vues/filtres/tris, collaboration, navigation, templates, raccourcis clavier, fonctionnalités avancées
- **Format** : textes de 150-250 mots par document, encodés en vecteurs de 384 dimensions

---

## Application en ligne

Interface Streamlit déployée sur Render :
*(lien à venir après déploiement)*

---

## Installation locale

### Prérequis
- Python 3.11+
- Clé API Gemini (`GEMINI_API_KEY`) dans un fichier `.env`

### Lancement

```bash
# 1. Cloner le repo
git clone https://github.com/Nesho-k/RAG-Notion.git
cd RAG-Notion

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Construire les embeddings et l'index FAISS
python pipeline/b_create_embeddings.py
python pipeline/c_build_faiss_index.py

# 5. Lancer l'interface Streamlit
streamlit run app/app.py
```

### Variables d'environnement

Créez un fichier `.env` à la racine :

```
GEMINI_API_KEY=votre_clé_gemini
OPENAI_API_KEY=votre_clé_openai   # uniquement pour l'enrichissement du corpus
```

---

## Structure du projet

Le code du pipeline est dans `pipeline/`, avec un fichier par étape (a → e). L'interface est dans `app/app.py`. Les fichiers `documents.json`, `embeddings.npy` et `faiss_index.bin` sont générés localement après avoir lancé les étapes b et c — ils ne sont pas versionnés.

---

## Auteur

**Nesho Kanthakumar**,
Étudiant en Data Science
[GitHub](https://github.com/Nesho-k) · [LinkedIn](https://www.linkedin.com/in/nesho-kanthakumar-6354512a6/)
