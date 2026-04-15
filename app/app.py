"""
app.py
------
Interface Streamlit pour l'assistant RAG Notion.

Importe la logique métier depuis e_rag_pipeline.py :
  - charger_ressources()   : charge documents, index FAISS, modèle
  - rechercher_chunks()    : recherche sémantique FAISS
  - construire_contexte()  : assemble les chunks pour le prompt
  - appeler_gemini()       : appelle l'API Gemini

UTILISATION :
  streamlit run app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Dossier racine du projet (parent du dossier app/)
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(ROOT, "pipeline")

# Ajoute pipeline/ au chemin Python pour pouvoir importer e_rag_pipeline
sys.path.insert(0, PIPELINE_DIR)

from e_rag_pipeline import (
    charger_ressources,
    rechercher_chunks,
    construire_contexte,
    appeler_gemini,
    PROMPT_TEMPLATE,
)

# Charge le fichier .env depuis la racine du projet
load_dotenv(dotenv_path=os.path.join(ROOT, ".env"), override=False)


# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="Assistant Notion",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS PERSONNALISÉ
# ============================================================

st.markdown("""
<style>
    /* Police générale */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Titre principal */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Badge de section */
    .section-badge {
        display: inline-block;
        background: #f3f4f6;
        color: #374151;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 999px;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Carte de réponse */
    .response-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #6366f1;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.7;
    }

    /* Carte source */
    .source-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
        transition: box-shadow 0.2s;
    }

    .source-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    .source-rank {
        font-size: 0.7rem;
        font-weight: 700;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .source-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827;
        margin: 0.3rem 0;
    }

    .source-score {
        font-size: 0.8rem;
        color: #9ca3af;
    }

    .source-url {
        font-size: 0.8rem;
        color: #6366f1;
        word-break: break-all;
    }

    /* Bouton exemple */
    .stButton > button {
        width: 100%;
        text-align: left !important;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        color: #374151;
        font-size: 0.85rem;
        padding: 0.5rem 0.8rem;
        transition: all 0.15s;
    }

    .stButton > button:hover {
        background: #eef2ff;
        border-color: #6366f1;
        color: #4f46e5;
    }

    /* Masquer le footer Streamlit */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# INITIALISATION (une seule fois grâce au cache)
# ============================================================

def init_gemini():
    """Configure l'API Gemini — arrête l'app si la clé est manquante."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("**GEMINI_API_KEY manquante.** Ajoutez votre clé dans le fichier `.env`.")
        st.stop()
    genai.configure(api_key=api_key)


@st.cache_resource(show_spinner="Chargement du modèle et de l'index...")
def load_resources():
    """
    Charge les ressources RAG une seule fois au démarrage.
    @st.cache_resource : Streamlit garde le résultat en mémoire entre
    les interactions — le modèle (~117 Mo) n'est pas rechargé à chaque question.
    """
    # Vérification des fichiers avant d'appeler charger_ressources()
    # (qui ferait sys.exit() si un fichier manque — incompatible avec Streamlit)
    for fichier in ["documents.json", "faiss_index.bin"]:
        if not os.path.exists(fichier):
            st.error(f"**Fichier manquant : `{fichier}`**  \nLancez d'abord `b_create_embeddings.py` puis `c_build_faiss_index.py`.")
            st.stop()

    documents, index, modele = charger_ressources()
    return documents, index, modele


# ============================================================
# SIDEBAR
# ============================================================

def afficher_sidebar():
    """Affiche le panneau latéral avec les guides et exemples."""

    with st.sidebar:
        st.markdown("## Assistant Notion")
        st.markdown("Posez vos questions sur Notion en français. L'assistant s'appuie uniquement sur la documentation officielle.")

        st.divider()

        # -- Questions qui fonctionnent --
        st.markdown("""
<div style="background:#f0fdf4; border-left:4px solid #16a34a; border-radius:6px; padding:0.6rem 0.9rem; margin-bottom:0.8rem;">
    <span style="color:#15803d; font-weight:700; font-size:0.9rem;">Ce qui fonctionne</span>
</div>
""", unsafe_allow_html=True)

        exemples = [
            "Comment créer une page ?",
            "Comment lier deux bases de données ?",
            "Comment utiliser les raccourcis clavier ?",
            "Comment partager une page avec quelqu'un ?",
            "Comment créer un template personnalisé ?",
            "Comment filtrer une base de données ?",
            "Comment utiliser les rollups ?",
            "Comment travailler hors ligne ?",
        ]

        # Chaque bouton injecte la question dans le champ de saisie via session_state
        for exemple in exemples:
            if st.button(exemple, key=f"btn_{exemple}"):
                st.session_state.question_input = exemple
                st.rerun()

        st.divider()

        # -- Ce qui ne fonctionne pas --
        st.markdown("""
<div style="background:#fef2f2; border-left:4px solid #dc2626; border-radius:6px; padding:0.6rem 0.9rem; margin-bottom:0.8rem;">
    <span style="color:#b91c1c; font-weight:700; font-size:0.9rem;">Hors corpus</span>
</div>
""", unsafe_allow_html=True)
        st.markdown("""
- Prix et plans tarifaires
- Intégrations (Slack, GitHub...)
- API Notion
- Comparaison avec d'autres outils
- Fonctionnalités mobiles spécifiques
""")

        st.divider()

        # -- Info corpus --
        st.markdown("### Corpus actuel")
        st.markdown("""
**13 documents** couvrant :
- Pages & blocs
- Bases de données
- Collaboration & permissions
- Navigation & organisation
- Productivité & templates
- Fonctionnalités avancées
""")


# ============================================================
# AFFICHAGE DE LA RÉPONSE
# ============================================================

def afficher_reponse(reponse, chunks):
    """Affiche la réponse Gemini et les sources de manière structurée."""

    # -- Réponse --
    st.markdown('<div class="section-badge">Réponse</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="response-card">{reponse}</div>',
        unsafe_allow_html=True
    )

    # -- Sources --
    st.markdown('<div class="section-badge" style="margin-top:1.5rem">Sources utilisées</div>', unsafe_allow_html=True)

    # On affiche uniquement les 3 premières sources (Gemini en a reçu 5)
    cols = st.columns(3)

    for i, (col, (doc, score)) in enumerate(zip(cols, chunks[:3]), start=1):
        with col:
            titre = doc.get("title", "Titre non disponible")
            url   = doc.get("url", None)
            score_pct = f"{score:.0%}"

            # Couleur du score selon la pertinence
            if score >= 0.4:
                couleur_score = "#10b981"   # vert
            elif score >= 0.25:
                couleur_score = "#f59e0b"   # orange
            else:
                couleur_score = "#ef4444"   # rouge

            url_html = (
                f'<a href="{url}" target="_blank" class="source-url">Voir la documentation</a>'
                if url else
                '<span class="source-url" style="color:#9ca3af">Source non disponible</span>'
            )

            st.markdown(f"""
<div class="source-card">
    <div class="source-rank">Source {i}</div>
    <div class="source-title">{titre}</div>
    <div class="source-score" style="color:{couleur_score}; font-weight:600;">
        Pertinence : {score_pct}
    </div>
    <div style="margin-top:0.5rem">{url_html}</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE PRINCIPALE
# ============================================================

def main():
    init_gemini()
    documents, index, modele = load_resources()

    # Initialisation des variables de session
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    if "historique" not in st.session_state:
        st.session_state.historique = []

    afficher_sidebar()

    # -- En-tête --
    st.markdown('<div class="main-title">Assistant Notion</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Posez vos questions sur Notion — réponses basées sur la documentation officielle.</div>', unsafe_allow_html=True)

    # -- Champ de saisie --
    # Le champ est lié à st.session_state["question_input"] via key=
    # Les boutons sidebar écrivent directement dans cette clé puis font st.rerun()
    # La valeur est ainsi préservée quand l'utilisateur clique "Envoyer →"
    question = st.text_input(
        label="Votre question",
        key="question_input",
        placeholder="Ex : Comment créer une base de données dans Notion ?",
        label_visibility="collapsed",
    )

    envoyer = st.button("Envoyer →", type="primary")

    # -- Traitement de la question --
    if envoyer and question.strip():
        with st.spinner("Recherche des documents pertinents et génération de la réponse..."):

            # Étape 1 : Recherche FAISS
            chunks = rechercher_chunks(question, documents, index, modele)

            # Aucun chunk au-dessus du seuil → question hors corpus
            if not chunks:
                st.warning("Aucun document pertinent trouvé pour cette question. Elle semble hors du corpus Notion.")
                st.stop()

            # Étape 2 : Construction du prompt
            contexte = construire_contexte(chunks)
            prompt   = PROMPT_TEMPLATE.format(contexte=contexte, question=question)

            # Étape 3 : Appel Gemini
            reponse = appeler_gemini(prompt)

        if reponse is None:
            st.error("Impossible de générer une réponse. Vérifiez votre clé API ou réessayez dans quelques secondes.")
        else:
            # Sauvegarde dans l'historique de session
            st.session_state.historique.insert(0, {
                "question": question,
                "reponse": reponse,
                "chunks": chunks,
            })

    elif envoyer and not question.strip():
        st.warning("Veuillez saisir une question avant d'envoyer.")

    # -- Affichage de l'historique --
    for i, item in enumerate(st.session_state.historique):
        if i == 0:
            # Dernière réponse : affichage complet
            st.markdown(f"**Question :** {item['question']}")
            afficher_reponse(item["reponse"], item["chunks"])
        else:
            # Réponses précédentes : repliées dans un expander
            with st.expander(f"Q : {item['question']}", expanded=False):
                afficher_reponse(item["reponse"], item["chunks"])

        if i < len(st.session_state.historique) - 1:
            st.divider()


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    main()
