"""
scrape_notion_urls.py
---------------------
Scrape la page d'aide Notion pour récupérer toutes les URLs d'articles,
organisées par section et sous-section.

RÉSULTATS :
  - notion_urls.json : URLs organisées par section { "Section": ["url1", ...] }
  - notion_urls.txt  : liste plate de toutes les URLs (une par ligne)

UTILISATION :
  python scrape_notion_urls.py
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIGURATION
# ============================================================

URL_BASE        = "https://www.notion.com"
URL_HELP        = "https://www.notion.com/fr/help"
FICHIER_JSON    = os.path.join(ROOT, "notion_urls.json")
FICHIER_TXT     = os.path.join(ROOT, "notion_urls.txt")
DELAI_REQUETES  = 1   # secondes entre chaque requête (respecter le serveur)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9",
}


# ============================================================
# SCRAPING DE LA PAGE PRINCIPALE
# ============================================================

def scraper_page_principale():
    """
    Récupère la page d'accueil de l'aide Notion et extrait
    tous les liens vers des articles, groupés par section.
    """

    print(f"[...] Téléchargement de {URL_HELP}...")

    try:
        reponse = requests.get(URL_HELP, headers=HEADERS, timeout=15)
        reponse.raise_for_status()
    except Exception as e:
        print(f"[ERREUR] Impossible de charger la page : {e}")
        return None

    soup = BeautifulSoup(reponse.text, "html.parser")

    # Vérifie si la page est bien rendue (pas juste un shell JS vide)
    texte_page = soup.get_text()
    if len(texte_page.strip()) < 500:
        print("[AVERTISSEMENT] La page semble vide — elle est peut-être rendue par JavaScript.")
        print("                Voir la note en bas du script pour la solution alternative.")
        return None

    print(f"[OK] Page chargée ({len(texte_page)} caractères)")
    return soup


# ============================================================
# EXTRACTION DES URLS PAR SECTION
# ============================================================

def extraire_urls_par_section(soup):
    """
    Parcourt le HTML et regroupe les URLs par section/sous-section.
    Stratégie : cherche les titres de sections (h2, h3) puis les liens
    qui les suivent dans le DOM.
    """

    sections = {}
    section_courante = "Général"

    # On cherche tous les éléments — titres et liens — dans l'ordre du DOM
    for element in soup.find_all(["h2", "h3", "h4", "a"]):

        # Si c'est un titre → nouvelle section courante
        if element.name in ("h2", "h3", "h4"):
            texte = element.get_text(strip=True)
            if texte:
                section_courante = texte
                if section_courante not in sections:
                    sections[section_courante] = []

        # Si c'est un lien → on vérifie si c'est un article d'aide
        elif element.name == "a":
            href = element.get("href", "")

            # On garde uniquement les liens vers des articles d'aide
            if "/help/" in href and href not in ("#", "/fr/help", "/help"):
                # Construire l'URL complète si le lien est relatif
                url_complete = urljoin(URL_BASE, href)

                # S'assurer que c'est bien une URL Notion
                if "notion.com" in url_complete:
                    if section_courante not in sections:
                        sections[section_courante] = []
                    if url_complete not in sections[section_courante]:
                        sections[section_courante].append(url_complete)

    return sections


# ============================================================
# FALLBACK : EXTRACTION SIMPLE (toutes les URLs sans section)
# ============================================================

def extraire_toutes_urls(soup):
    """
    Méthode de fallback : extrait toutes les URLs d'aide
    sans essayer de les grouper par section.
    Utile si la structure de la page ne permet pas le groupement.
    """

    urls = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/help/" in href and href not in ("#", "/fr/help", "/help"):
            url_complete = urljoin(URL_BASE, href)
            if "notion.com" in url_complete:
                urls.add(url_complete)

    return sorted(urls)


# ============================================================
# SAUVEGARDE DES RÉSULTATS
# ============================================================

def sauvegarder(sections, toutes_urls):
    """Sauvegarde les résultats dans deux formats."""

    # -- JSON organisé par section --
    with open(FICHIER_JSON, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] URLs par section → {FICHIER_JSON}")

    # -- TXT liste plate --
    with open(FICHIER_TXT, "w", encoding="utf-8") as f:
        for url in toutes_urls:
            f.write(url + "\n")
    print(f"[OK] Liste plate      → {FICHIER_TXT}")


# ============================================================
# AFFICHAGE DU RÉSUMÉ
# ============================================================

def afficher_resume(sections, toutes_urls):
    """Affiche un résumé lisible dans le terminal."""

    print(f"\n{'='*55}")
    print(f"  RÉSULTATS DU SCRAPING")
    print(f"{'='*55}\n")

    if sections:
        for section, urls in sections.items():
            if urls:
                print(f"  [{len(urls):>3} URLs] {section}")
    else:
        print("  Aucune section détectée — liste plate uniquement")

    print(f"\n  TOTAL : {len(toutes_urls)} URLs trouvées")
    print(f"\n{'='*55}")
    print(f"  → Copiez les URLs qui vous intéressent dans")
    print(f"    URLS_A_AJOUTER dans enrich_corpus.py")
    print(f"{'='*55}\n")


# ============================================================
# POINT D'ENTRÉE
# ============================================================

def main():
    print(f"\n{'='*55}")
    print(f"  SCRAPING DES URLs NOTION HELP")
    print(f"{'='*55}\n")

    soup = scraper_page_principale()

    if soup is None:
        print("\n  La page principale n'a pas pu être scrapée.")
        print("  Raison probable : page rendue par JavaScript (React/Next.js).")
        print()
        print("  SOLUTION : utilisez le navigateur manuellement —")
        print("  1. Ouvrez https://www.notion.com/fr/help dans Chrome")
        print("  2. Clic droit → Enregistrer sous → Page Web, complète")
        print("  3. Relancez ce script avec le fichier HTML local :")
        print("     Modifiez URL_HELP par le chemin du fichier sauvegardé")
        return

    # Extraction par section
    sections = extraire_urls_par_section(soup)

    # Liste plate (toutes URLs confondues)
    toutes_urls = extraire_toutes_urls(soup)

    if not toutes_urls:
        print("[AVERTISSEMENT] Aucune URL d'article trouvée.")
        print("  La page est probablement rendue par JavaScript.")
        return

    sauvegarder(sections, toutes_urls)
    afficher_resume(sections, toutes_urls)


if __name__ == "__main__":
    main()
