# app.py
import streamlit as st
import requests
from textwrap import dedent

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="ClusterLens Documentation",
    page_icon="clusterlens_logo.png",
    layout="wide",
)

SHOW_GITHUB_BADGE = False

# ---------------------------------------------------------
# Tunables (no JS; fixed, safe top offset for Cloud)
# ---------------------------------------------------------
LEFT_SIDEBAR_PX = 240
RIGHT_SIDEBAR_PX = 220
LOGO_WIDTH = 220
TOP_OFFSET_REM = 6  # <- increase if you still see clipping on Cloud

# ---------------------------------------------------------
# Global CSS (sticky sidebars; no :has; no JS)
# ---------------------------------------------------------
st.markdown(
    f"""
    <style>
    :root {{
        --left-col: {LEFT_SIDEBAR_PX}px;
        --right-col: {RIGHT_SIDEBAR_PX}px;
        --top-offset: {TOP_OFFSET_REM}rem;
    }}

    /* App container */
    .block-container {{
        padding-top: 0.25rem;  /* tiny; real spacing handled below */
        padding-left: 0;
        padding-right: 0;
        max-width: 1900px;
    }}

    /* ===== Left sticky column (1) ===== */
    div[data-testid="column"]:nth-of-type(1) {{
        position: sticky;
        top: var(--top-offset);
        align-self: flex-start;
        width: var(--left-col);
        min-width: var(--left-col);
        max-width: var(--left-col);
        height: calc(100vh - var(--top-offset));
        padding: 0.9rem 1rem 1.25rem 1rem;
        border-right: 1px solid #e5e7eb;
        background-color: #f3f4f6;
        overflow-y: auto;
        box-sizing: border-box;
        z-index: 2;
    }}

    /* ===== Main column (2) ===== */
    div[data-testid="column"]:nth-of-type(2) {{
        margin-left: var(--left-col);
        margin-right: var(--right-col);
        padding: 0 1.5rem;
        padding-top: var(--top-offset);  /* prevents clipping under header */
        box-sizing: border-box;
    }}

    /* ===== Right sticky column (3) ===== */
    div[data-testid="column"]:nth-of-type(3) {{
        position: sticky;
        top: var(--top-offset);
        align-self: flex-start;
        width: var(--right-col);
        min-width: var(--right-col);
        max-width: var(--right-col);
        height: calc(100vh - var(--top-offset));
        padding: 0.75rem 1rem 1.25rem 1rem;
        border-left: 1px solid #e5e7eb;
        background-color: #ffffff;
        overflow-y: auto;
        box-sizing: border-box;
        z-index: 2;
    }}

    /* ===== Centered, larger logo ===== */
    .cl-logo-wrap {{
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.25rem 0 0.5rem 0;
    }}
    .cl-logo-wrap img {{
        width: {LOGO_WIDTH}px;
        max-width: 92%;
        height: auto;
        display: block;
    }}

    /* Search */
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] {{ position: relative; margin: 0.25rem 0 1rem 0; }}
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] > div {{ background: transparent !important; box-shadow: none !important; padding: 0 !important; }}
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] label {{ display:none; }}
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] input {{
        border-radius: 999px; border: 1px solid #d1d5db; padding: 0.35rem 0.9rem 0.35rem 2rem; font-size: 0.9rem; background: #fff;
    }}
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"]::before {{
        content: "🔍"; position: absolute; left: 0.6rem; top: 50%; transform: translateY(-50%); font-size: 0.85rem; color: #9ca3af; pointer-events: none;
    }}

    /* Radio-as-docs nav */
    div[data-testid="stRadio"] > label {{ display: none !important; }}
    div[data-testid="stRadio"] div[role="radiogroup"] {{ display: flex; flex-direction: column; gap: 0.15rem; }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label {{ padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.95rem; color: #374151; }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {{ display: none !important; }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {{ background-color: #eff6ff; border-left: 3px solid #2563eb; color: #111827; font-weight: 600; }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{ background-color: #e5e7eb; }}

    /* TOC */
    div[data-testid="column"]:nth-of-type(3) h6 {{ font-size: 0.85rem; font-weight: 600; margin-bottom: 0.15rem; color: #4b5563; }}
    div[data-testid="column"]:nth-of-type(3) ul {{ list-style-type: disc; padding-left: 1.1rem; margin: 0; }}
    div[data-testid="column"]:nth-of-type(3) li {{ margin: 0; padding: 0; line-height: 1.1; }}
    div[data-testid="column"]:nth-of-type(3) li a {{ font-size: 0.8rem; text-decoration: none; color: #2563eb; }}
    div[data-testid="column"]:nth-of-type(3) li a:hover {{ text-decoration: underline; }}

    /* Anchor offset for in-page links */
    h1, h2, h3, h4, h5, h6 {{ scroll-margin-top: calc(var(--top-offset) + 8px); }}

    pre, code {{ font-size: 0.9rem !important; }}

    /* Responsive fallbacks */
    @media (max-width: 1100px) {{
        :root {{ --left-col: 200px; --right-col: 200px; }}
    }}
    @media (max-width: 900px) {{
        div[data-testid="column"] {{
            position: static !important; width: auto !important; min-width: 0 !important; max-width: none !important;
            margin: 0 !important; height: auto !important; overflow: visible !important;
        }}
        h1, h2, h3, h4, h5, h6 {{ scroll-margin-top: 1rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_github_stars():
    """Fetch GitHub stars for ClusterLens once per hour."""
    url = "https://api.github.com/repos/akthammomani/ClusterLens"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("stargazers_count", 0)
    except Exception:
        pass
    return None

def subheader_with_anchor(text: str, anchor: str):
    st.markdown(f'<div id="{anchor}"></div>', unsafe_allow_html=True)
    st.subheader(text)

# ---------------------------------------------------------
# Nav model
# ---------------------------------------------------------
SECTIONS = [
    {"id": "home", "label": "Home"},
    {"id": "quickstart", "label": "Quickstart"},
    {"id": "data_requirements", "label": "Data requirements"},
    {"id": "api_init_fit", "label": "API: init & fit"},
    {"id": "api_importance_shap", "label": "API: importance & SHAP"},
    {"id": "api_contrastive", "label": "API: contrastive importance"},
    {"id": "api_distributions", "label": "API: distributions"},
    {"id": "api_narratives_summaries", "label": "API: narratives & summaries"},
    {"id": "api_splits_exports", "label": "API: splits & exports"},
    {"id": "under_the_hood", "label": "Under the hood"},
]
SECTION_SEARCH = {"home": "overview", "quickstart": "quickstart", "data_requirements": "data", "api_init_fit": "init fit",
                  "api_importance_shap": "importance shap", "api_contrastive": "contrastive", "api_distributions": "dists",
                  "api_narratives_summaries": "narratives summaries", "api_splits_exports": "splits exports", "under_the_hood": "internals"}

TOC_ITEMS = {
    "api_init_fit": [
        {"label": "ClusterAnalyzer.__init__", "anchor": "api_init_fit_init"},
        {"label": "ClusterAnalyzer.fit", "anchor": "api_init_fit_fit"},
    ],
    "api_importance_shap": [
        {"label": "plot_cluster_shap", "anchor": "api_importance_plot"},
        {"label": "get_cluster_classification_stats", "anchor": "api_importance_stats"},
        {"label": "get_top_shap_features", "anchor": "api_importance_top_feats"},
    ],
}

if "active_section" not in st.session_state:
    st.session_state["active_section"] = "home"

# ---------------------------------------------------------
# 3 columns
# ---------------------------------------------------------
col_nav, col_main, col_toc = st.columns([0.22, 0.6, 0.18], gap="small")

# ---------------------- NAV COLUMN -----------------------
with col_nav:
    st.markdown(
        """
        <div class="cl-logo-wrap">
            <img src="clusterlens_logo.png" alt="ClusterLens logo"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if SHOW_GITHUB_BADGE:
        stars = get_github_stars()
        stars_text = f"{stars:,}" if stars is not None else "Repo"
        st.markdown(
            f"""
            <a href="https://github.com/akthammomani/ClusterLens" target="_blank"
               style="display:inline-flex;align-items:stretch;margin:.25rem auto 1rem;border-radius:4px;overflow:hidden;border:1px solid #d0d7de;font-size:.8rem;text-decoration:none;color:#111827;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <span style="display:inline-flex;align-items:center;gap:.35rem;padding:.25rem .6rem;background:#f6f8fa;"> GitHub</span>
                <span style="padding:.25rem .6rem;border-left:1px solid #d0d7de;font-variant-numeric:tabular-nums;background:#fff;">{stars_text}</span>
            </a>
            """,
            unsafe_allow_html=True,
        )

    query = st.text_input("", placeholder="Search", label_visibility="collapsed")
    filtered_sections = (
        [s for s in SECTIONS if query.lower() in (s["label"] + " " + SECTION_SEARCH.get(s["id"], "")).lower()] or SECTIONS
        if query else SECTIONS
    )
    selected_label = st.radio("", options=[s["label"] for s in filtered_sections], label_visibility="collapsed", key="nav_radio")
    st.session_state["active_section"] = next(s["id"] for s in filtered_sections if s["label"] == selected_label)

section_id = st.session_state["active_section"]

# ---------------------- MAIN COLUMN ----------------------
with col_main:
    if section_id == "api_init_fit":
        st.header("API: init & fit")
        subheader_with_anchor("ClusterAnalyzer.__init__", "api_init_fit_init")
        st.markdown(
            dedent(
                """
                ```python
                ClusterAnalyzer(
                    df: pd.DataFrame,
                    num_features: Optional[List[str]] = None,
                    cat_features: Optional[List[str]] = None,
                    cluster_col: str = "Cluster",
                    random_state: int = 1981,
                    encoder: str = "onehot",
                    model_type: str = "rf",
                    model_params: Optional[dict] = None,
                    eval_max_n: Optional[int] = None,
                )
                ```
                """
            )
        )
        subheader_with_anchor("ClusterAnalyzer.fit", "api_init_fit_fit")
        st.markdown(
            dedent(
                """
                ```python
                ca.fit(
                    test_size: float = 0.2,
                    sample_n: Optional[int] = None,
                    sample_frac: Optional[float] = None,
                    stratify_sample: bool = True,
                )
                ```
                """
            )
        )
    else:
        st.header("ClusterLens")
        st.write("Select a section from the left.")

# ---------------------- RIGHT TOC COLUMN -----------------
with col_toc:
    items = TOC_ITEMS.get(section_id, [])
    if items:
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        st.markdown("###### On this page")
        for item in items:
            st.markdown(f"- [{item['label']}](#{item['anchor']})")
