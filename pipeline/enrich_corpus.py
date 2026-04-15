"""
enrich_corpus.py
----------------
Enrichit automatiquement le corpus RAG à partir d'URLs Notion.

COMMENT ÇA MARCHE :
  1. Vous renseignez une liste d'URLs Notion dans URLS_A_AJOUTER
  2. Le script télécharge chaque page et en extrait le texte brut
  3. Gemini résume le contenu en français (150-250 mots)
  4. Chaque nouveau document est affiché pour validation avant ajout
  5. Le corpus JSON est mis à jour automatiquement

UTILISATION :
  1. Remplissez la liste URLS_A_AJOUTER ci-dessous
  2. python enrich_corpus.py
  3. Relancez ensuite : a_verify_dataset.py → b_create_embeddings.py → c_build_faiss_index.py
"""

import json
import os
import sys
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# Dossier racine du projet (parent du dossier pipeline/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(dotenv_path=os.path.join(ROOT, ".env"), override=False)

# ============================================================
# URLS À AJOUTER — remplissez cette liste
# ============================================================

# Laissez cette liste vide pour lire automatiquement depuis notion_urls.json
# Ou remplissez-la manuellement pour n'ajouter que certaines URLs
URLS_A_AJOUTER = []

# Fichier contenant toutes les URLs scrapées (généré par scrape_notion_urls.py)
FICHIER_URLS = os.path.join(ROOT, "notion_urls.json")

# ============================================================
# CONFIGURATION
# ============================================================

FICHIER_CORPUS         = os.path.join(ROOT, "notion_rag_corpus.json")       # Original — jamais modifié
FICHIER_NOUVEAUX_DOCS  = os.path.join(ROOT, "notion_rag_new_docs.json")    # Nouveaux documents ajoutés ici
NOM_MODELE      = "gpt-4o-mini"   # Modèle OpenAI — rapide, peu coûteux, sans limite de quota stricte
MOTS_MIN        = 150
MOTS_MAX        = 250
DELAI_ENTRE_REQUETES = 2   # secondes entre chaque appel (évite le rate limiting)
AUTO_VALIDER         = True  # True = ajout automatique sans confirmation, False = demande pour chaque doc

# Prompt envoyé à OpenAI pour générer le résumé
PROMPT_RESUME = """Tu es un rédacteur technique expert sur Notion.
À partir du contenu d'une page d'aide Notion, génère un résumé en français.

CONTRAINTES STRICTES :
- Entre {min} et {max} mots exactement
- En français uniquement
- Factuel : uniquement ce qui est dans le contenu fourni
- Pas de titre, pas de bullet points — un seul bloc de texte continu
- Commence directement par le contenu (pas de "Voici un résumé...")
- IMPÉRATIF : conserve les termes techniques Notion tels quels, sans les traduire ni les paraphraser.
  Exemples à conserver : rollup, filtre, vue, propriété, bloc, base de données, relation,
  formule, template, webhook, workspace, teamspace, sidebar, toggle, synced block,
  kanban, timeline, galerie, tableau, calendrier, breadcrumb, backlink.

CONTENU DE LA PAGE :
{contenu}

RÉSUMÉ ({min}-{max} mots) :"""


# ============================================================
# INITIALISATION OPENAI
# ============================================================

def init_openai():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERREUR] OPENAI_API_KEY non définie dans le fichier .env")
        sys.exit(1)
    client = OpenAI(api_key=api_key)
    print(f"[OK] API OpenAI configurée\n")
    return client


# ============================================================
# CHARGEMENT DU CORPUS EXISTANT
# ============================================================

def charger_corpus():
    """
    Charge le fichier de nouveaux documents (notion_rag_new_docs.json).
    Si ce fichier n'existe pas encore, on le crée vide.
    Calcule le prochain ID en tenant compte des deux fichiers (original + nouveaux).
    """

    if not os.path.exists(FICHIER_CORPUS):
        print(f"[ERREUR] Corpus original introuvable : {FICHIER_CORPUS}")
        sys.exit(1)

    # Charge le corpus original pour calculer le prochain ID
    with open(FICHIER_CORPUS, "r", encoding="utf-8") as f:
        corpus_original = json.load(f)

    # Charge les nouveaux docs s'ils existent, sinon démarre avec une liste vide
    if os.path.exists(FICHIER_NOUVEAUX_DOCS):
        with open(FICHIER_NOUVEAUX_DOCS, "r", encoding="utf-8") as f:
            nouveaux_docs = json.load(f)
        print(f"[OK] {len(nouveaux_docs)} nouveau(x) document(s) existant(s) dans notion_rag_new_docs.json")
    else:
        nouveaux_docs = []
        print(f"[OK] notion_rag_new_docs.json sera créé")

    # Le prochain ID est calculé sur l'ensemble des deux fichiers
    tous_les_docs = corpus_original + nouveaux_docs
    numeros = []
    for doc in tous_les_docs:
        try:
            numeros.append(int(doc["id"].replace("doc_", "")))
        except (ValueError, KeyError):
            pass

    prochain_numero = max(numeros) + 1 if numeros else 1
    print(f"[OK] {len(corpus_original)} documents originaux + {len(nouveaux_docs)} nouveaux")
    print(f"     Prochain ID : doc_{prochain_numero:03d}\n")

    return nouveaux_docs, prochain_numero


# ============================================================
# SCRAPING DE LA PAGE
# ============================================================

def scraper_page(url):
    """
    Télécharge une page Notion Help et en extrait le texte principal.

    POURQUOI BeautifulSoup ?
      Les pages Notion Help sont rendues côté serveur (pas de JS requis),
      ce qui permet de les scraper simplement avec requests + BeautifulSoup.
      BeautifulSoup parse le HTML et permet d'extraire uniquement le texte
      utile en ignorant la navigation, les menus et les publicités.
    """

    headers = {
        # Un User-Agent réaliste évite d'être bloqué par certains serveurs
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        print(f"  [1/3] Téléchargement de la page...")
        reponse = requests.get(url, headers=headers, timeout=15)

        # Certaines pages n'ont pas de version française : Notion retourne 404
        # ou redirige vers une URL sans /fr/. Dans ce cas on retente en anglais.
        if reponse.status_code == 404 and "/fr/help/" in url:
            url_en = url.replace("/fr/help/", "/help/")
            print(f"  [INFO] Version française non trouvée — essai en anglais : {url_en}")
            reponse = requests.get(url_en, headers=headers, timeout=15)

        reponse.raise_for_status()  # Lève une exception si code HTTP ≥ 400

    except requests.exceptions.Timeout:
        print(f"  [ERREUR] La page n'a pas répondu dans les 15 secondes : {url}")
        return None, None
    except requests.exceptions.HTTPError as e:
        print(f"  [ERREUR] HTTP {e.response.status_code} pour : {url}")
        return None, None
    except requests.exceptions.ConnectionError:
        print(f"  [ERREUR] Impossible de joindre : {url}")
        return None, None

    # Parsing HTML avec BeautifulSoup
    soup = BeautifulSoup(reponse.text, "html.parser")

    # -- Extraction du titre --
    # On stocke le résultat de find() dans une variable avant d'appeler get_text()
    # afin que Pylance sache que l'objet n'est pas None au moment de l'appel
    titre = None
    h1 = soup.find("h1")
    title_tag = soup.find("title")

    if h1:
        titre = h1.get_text(strip=True)
    elif title_tag:
        # La balise <title> contient souvent " | Notion Help" qu'on supprime
        titre = title_tag.get_text(strip=True)
        titre = titre.replace(" – Notion Help", "").replace(" | Notion", "").strip()

    # -- Extraction du contenu principal --
    # On retire les balises inutiles avant d'extraire le texte
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "button", "iframe"]):
        tag.decompose()   # Supprime le tag et son contenu du DOM

    # On cherche le conteneur principal de l'article
    contenu = None
    for selecteur in ["article", "main", "[role='main']", ".help-article", ".content"]:
        element = soup.select_one(selecteur)
        if element:
            contenu = element.get_text(separator="\n", strip=True)
            break

    # Fallback : prend tout le body si aucun conteneur trouvé
    if not contenu and soup.body:
        contenu = soup.body.get_text(separator="\n", strip=True)

    if not contenu or len(contenu.strip()) < 100:
        print(f"  [ERREUR] Contenu trop court ou vide pour : {url}")
        return None, None

    # Limite le contenu à 3000 caractères pour ne pas dépasser les tokens Gemini
    contenu_tronque = contenu[:3000]

    print(f"  [OK] Page téléchargée — titre : '{titre or 'Non détecté'}'")
    print(f"       Contenu extrait : {len(contenu)} caractères")

    return titre, contenu_tronque


# ============================================================
# GÉNÉRATION DU RÉSUMÉ AVEC OPENAI
# ============================================================

def generer_resume(contenu, client):
    """
    Envoie le contenu à OpenAI et retourne un résumé de 150-250 mots.
    Retente jusqu'à 3 fois en cas d'erreur 429.
    """

    print(f"  [2/3] Génération du résumé par OpenAI...")

    prompt = (
        PROMPT_RESUME
        .replace("{min}", str(MOTS_MIN))
        .replace("{max}", str(MOTS_MAX))
        .replace("{contenu}", contenu)
    )

    for essai in range(1, 4):
        try:
            reponse = client.chat.completions.create(
                model=NOM_MODELE,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,   # Température basse = réponse factuelle et stable
            )
            resume = reponse.choices[0].message.content.strip()

            nb_mots = len(resume.split())
            print(f"  [OK] Résumé généré — {nb_mots} mots")

            return resume

        except Exception as e:
            if "429" in str(e) and essai < 3:
                print(f"  [QUOTA] Attente 10s avant tentative {essai + 1}/3...")
                time.sleep(10)
            else:
                print(f"  [ERREUR] OpenAI : {e}")
                return None

    return None


# ============================================================
# DÉTECTION DE LA SECTION
# ============================================================

def detecter_section(url, titre):
    """
    Devine la section à partir de l'URL ou du titre.
    Vous pouvez enrichir ce dictionnaire selon vos besoins.
    """

    correspondances = {
        "database":     "Bases de données",
        "relation":     "Bases de données avancées",
        "rollup":       "Bases de données avancées",
        "formula":      "Bases de données avancées",
        "filter":       "Bases de données",
        "template":     "Productivité",
        "page":         "Pages & contenu",
        "block":        "Pages & contenu",
        "share":        "Collaboration",
        "permission":   "Collaboration",
        "member":       "Collaboration",
        "sidebar":      "Navigation",
        "search":       "Navigation",
        "keyboard":     "Productivité",
        "shortcut":     "Productivité",
        "offline":      "Fonctionnalités avancées",
        "api":          "Fonctionnalités avancées",
        "ai":           "Fonctionnalités avancées",
        "agent":        "Fonctionnalités avancées",
        "import":       "Fonctionnalités avancées",
        "export":       "Fonctionnalités avancées",
        "refund":       "Facturation & abonnements",
        "billing":      "Facturation & abonnements",
        "plan":         "Facturation & abonnements",
        "pricing":      "Facturation & abonnements",
    }

    url_lower = url.lower()
    for mot_cle, section in correspondances.items():
        if mot_cle in url_lower:
            return section

    return "Fonctionnalités Notion"   # Section générique par défaut


# ============================================================
# VALIDATION ET AJOUT AU CORPUS
# ============================================================

def valider_et_ajouter(corpus, doc_id, titre, url, section, resume):
    """
    Affiche le document généré et demande confirmation avant de l'ajouter.
    """

    print(f"\n  [3/3] Document généré :")
    print(f"  {'─'*50}")
    print(f"  ID      : {doc_id}")
    print(f"  Titre   : {titre}")
    print(f"  Section : {section}")
    print(f"  URL     : {url}")
    print(f"  Mots    : {len(resume.split())}")
    print(f"  {'─'*50}")
    print(f"\n{resume}\n")
    print(f"  {'─'*50}")

    # En mode AUTO_VALIDER, on ajoute sans demander
    if AUTO_VALIDER:
        choix = "o"
        print("  [AUTO] Ajout automatique activé")
    else:
        choix = input("\n  Ajouter ce document au corpus ? [o/n] : ").strip().lower()

    if choix in ("o", "oui", "y", "yes"):
        nouveau_doc = {
            "id":      doc_id,
            "title":   titre,
            "section": section,
            "source":  "Notion Docs",
            "url":     url,
            "text":    resume,
        }
        corpus.append(nouveau_doc)
        print(f"  [OK] '{doc_id}' ajouté au corpus\n")
        return True
    else:
        print(f"  [IGNORÉ] Document non ajouté\n")
        return False


# ============================================================
# SAUVEGARDE DU CORPUS
# ============================================================

def sauvegarder_corpus(nouveaux_docs):
    """
    Sauvegarde uniquement dans notion_rag_new_docs.json.
    Le fichier original notion_rag_corpus.json n'est jamais touché.
    La fusion des deux se fait automatiquement dans b_create_embeddings.py.
    """

    with open(FICHIER_NOUVEAUX_DOCS, "w", encoding="utf-8") as f:
        json.dump(nouveaux_docs, f, ensure_ascii=False, indent=2)

    print(f"[OK] {len(nouveaux_docs)} document(s) sauvegardé(s) → notion_rag_new_docs.json")
    print(f"     notion_rag_corpus.json : inchangé ✓")


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

def main():
    print(f"\n{'='*55}")
    print("  ENRICHISSEMENT AUTOMATIQUE DU CORPUS")
    print(f"{'='*55}\n")

    # Vérification que la liste d'URLs n'est pas vide
    urls_valides = [u.strip() for u in URLS_A_AJOUTER if u.strip()]

    # Si la liste manuelle est vide, on charge depuis notion_urls.json
    if not urls_valides:
        if not os.path.exists(FICHIER_URLS):
            print("[ERREUR] notion_urls.json introuvable et URLS_A_AJOUTER est vide.")
            print("  → Lancez d'abord : python scrape_notion_urls.py")
            sys.exit(1)

        print(f"[INFO] URLS_A_AJOUTER vide — chargement depuis notion_urls.json...")
        with open(FICHIER_URLS, "r", encoding="utf-8") as f:
            urls_par_section = json.load(f)

        # Aplatir toutes les URLs de toutes les sections
        # Filtrer les pages d'index (/category/) et les guides (/guides/)
        # Convertir les URLs sans /fr/ en version française
        for urls in urls_par_section.values():
            for url in urls:
                if "/category/" not in url and "/guides" not in url:
                    if "/fr/help/" not in url:
                        url = url.replace("/help/", "/fr/help/")
                    urls_valides.append(url)

        # Dédoublonner (dict.fromkeys préserve l'ordre)
        urls_valides = list(dict.fromkeys(urls_valides))
        print(f"[OK] {len(urls_valides)} URLs chargées (catégories et guides filtrés)\n")

    client = init_openai()
    corpus, prochain_numero = charger_corpus()

    ajoutes = 0
    ignores = 0

    for i, url in enumerate(urls_valides, start=1):
        print(f"{'─'*55}")
        print(f"  [{i}/{len(urls_valides)}] Traitement : {url}\n")

        doc_id = f"doc_{prochain_numero:03d}"

        # Vérification doublon — URL déjà présente dans l'un des deux fichiers ?
        # On normalise les URLs pour que /fr/help/X et /help/X soient considérés identiques
        def normaliser_url(u):
            return u.replace("//www.notion.com/fr/help/", "//www.notion.com/help/")

        with open(FICHIER_CORPUS, "r", encoding="utf-8") as f:
            tous_les_docs = json.load(f) + corpus
        urls_existantes = {normaliser_url(doc.get("url", "")) for doc in tous_les_docs}
        if normaliser_url(url) in urls_existantes:
            print(f"  [IGNORÉ] URL déjà dans le corpus : {url}\n")
            ignores += 1
            continue

        # Étape 1 : Scraping
        titre, contenu = scraper_page(url)
        if not contenu:
            ignores += 1
            continue

        # Si le titre n'a pas été détecté, demander à l'utilisateur
        if not titre:
            titre = input("  Titre non détecté — entrez le titre manuellement : ").strip()
            if not titre:
                titre = f"Document {doc_id}"

        # Étape 2 : Résumé Gemini
        resume = generer_resume(contenu, client)
        if not resume:
            ignores += 1
            continue

        # Détection automatique de la section
        section = detecter_section(url, titre)

        # Étape 3 : Validation et ajout
        ajoute = valider_et_ajouter(corpus, doc_id, titre, url, section, resume)

        if ajoute:
            ajoutes += 1
            prochain_numero += 1

        # Pause entre les requêtes pour éviter le rate limiting
        if i < len(urls_valides):
            time.sleep(DELAI_ENTRE_REQUETES)

    # Sauvegarde finale
    if ajoutes > 0:
        print(f"\n{'='*55}")
        sauvegarder_corpus(corpus)
        print(f"\n  {ajoutes} document(s) ajouté(s), {ignores} ignoré(s)")
        print(f"\n  → Relancez maintenant dans cet ordre :")
        print(f"     python a_verify_dataset.py")
        print(f"     python b_create_embeddings.py")
        print(f"     python c_build_faiss_index.py")
    else:
        print(f"\n  Aucun document ajouté — corpus inchangé.")

    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
