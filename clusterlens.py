# app.py
import streamlit as st
import requests
from textwrap import dedent
import streamlit.components.v1 as components  # for header-height fix

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
# Robust header height -> CSS var to avoid "clipped top"
# (Why: Streamlit Cloud header/toolbars differ from local; we measure once and on resize)
# ---------------------------------------------------------
components.html(
    """
    <script>
    (function() {
      const doc = parent.document;
      const header = doc.querySelector('header[data-testid="stHeader"]');
      const root = doc.documentElement;

      function setOffset() {
        const h = header ? header.offsetHeight : 64;
        // extra 8px breathing room
        root.style.setProperty('--top-offset', (h + 8) + 'px');
      }
      setOffset();
      if (header && 'ResizeObserver' in window) {
        try { new ResizeObserver(setOffset).observe(header); } catch (e) { setOffset(); }
      } else {
        // fallback update on load + after a tick
        setTimeout(setOffset, 500);
      }
    })();
    </script>
    """,
    height=0,
)

# ---------------------------------------------------------
# Layout + visual constants (tunable)
# ---------------------------------------------------------
LEFT_SIDEBAR_PX = 240
RIGHT_SIDEBAR_PX = 220
LOGO_WIDTH = 140

# ---------------------------------------------------------
# Global CSS (sticky sidebars, margins, logo sizing, etc.)
# ---------------------------------------------------------
st.markdown(
    f"""
    <style>
    :root {{
        --left-col: {LEFT_SIDEBAR_PX}px;
        --right-col: {RIGHT_SIDEBAR_PX}px;
        --top-offset: 72px; /* default; JS above will update this */
    }}

    /* ===== App container spacing ===== */
    .block-container {{
        /* keep a small top pad; true top spacing handled by columns below */
        padding-top: 0.5rem;
        padding-left: 0;
        padding-right: 0;
        max-width: 1900px;
    }}

    /* ===== Left column: sticky nav ===== */
    /* Use high specificity and attribute selectors (more robust across Streamlit versions) */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="column"]) > div[data-testid="column"]:nth-of-type(1) {{
        position: sticky;
        top: var(--top-offset);
        align-self: flex-start;             /* ensure sticky works in flex contexts */
        width: var(--left-col);
        min-width: var(--left-col);
        max-width: var(--left-col);
        height: calc(100vh - var(--top-offset));
        padding: 0.75rem 1rem 1.25rem 1rem;
        border-right: 1px solid #e5e7eb;
        background-color: #f3f4f6;
        overflow-y: auto;
        z-index: 2;
    }}

    /* ===== Main column ===== */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="column"]) > div[data-testid="column"]:nth-of-type(2) {{
        /* leave space for the sticky columns */
        margin-left: var(--left-col);
        margin-right: var(--right-col);
        padding: 0 1.5rem;
        padding-top: var(--top-offset);     /* avoid clipping under Streamlit header */
    }}

    /* ===== Right column: sticky TOC ===== */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="column"]) > div[data-testid="column"]:nth-of-type(3) {{
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
        z-index: 2;
    }}

    /* ===== Minor polish ===== */
    /* Cap any image in left column (logo) */
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="column"]) > div[data-testid="column"]:nth-of-type(1) img {{
        display: block;
        margin: 0 auto;
        max-width: {LOGO_WIDTH}px;
        height: auto;
    }}
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="column"]) > div[data-testid="column"]:nth-of-type(1) div[data-testid="stImage"] {{
        margin-bottom: 0.5rem;
    }}

    /* radio-as-docs nav */
    div[data-testid="stRadio"] > label {{ display: none !important; }}
    div[data-testid="stRadio"] div[role="radiogroup"] {{
        display: flex; flex-direction: column; gap: 0.15rem;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label {{
        padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.95rem; font-weight: 400; color: #374151;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {{ display: none !important; }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {{
        background-color: #eff6ff; border-left: 3px solid #2563eb; color: #111827; font-weight: 600;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{ background-color: #e5e7eb; }}

    /* search box */
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] {{
        position: relative; margin: 0.25rem 0 1rem 0;
    }}
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] > div {{
        background: transparent !important; box-shadow: none !important; padding: 0 !important;
    }}
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] label {{ display:none; }}
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] input {{
        border-radius: 999px; border: 1px solid #d1d5db; padding: 0.35rem 0.9rem 0.35rem 2rem; font-size: 0.9rem; background: #fff;
    }}
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"]::before {{
        content: "🔍"; position: absolute; left: 0.6rem; top: 50%; transform: translateY(-50%); font-size: 0.85rem; color: #9ca3af; pointer-events: none;
    }}

    /* TOC */
    div[data-testid="column"]:nth-of-type(3) h6 {{
        font-size: 0.85rem; font-weight: 600; margin-bottom: 0.15rem; color: #4b5563;
    }}
    div[data-testid="column"]:nth-of-type(3) ul {{ list-style-type: disc; padding-left: 1.1rem; margin: 0; }}
    div[data-testid="column"]:nth-of-type(3) li {{ margin: 0; padding: 0; line-height: 1.1; }}
    div[data-testid="column"]:nth-of-type(3) li a {{ font-size: 0.8rem; text-decoration: none; color: #2563eb; }}
    div[data-testid="column"]:nth-of-type(3) li a:hover {{ text-decoration: underline; }}

    /* Headings offset to not hide under header when jumping */
    h1, h2, h3, h4, h5, h6 {{ scroll-margin-top: calc(var(--top-offset) + 8px); }}

    pre, code {{ font-size: 0.9rem !important; }}

    /* ===== Responsive fallback ===== */
    @media (max-width: 1100px) {{
        :root {{
            --left-col: 200px;
            --right-col: 200px;
        }}
    }}
    @media (max-width: 900px) {{
        /* collapse to 1col on small screens */
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="column"]) > div[data-testid="column"] {{
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
# Small helpers
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
    """Render a subheader with an HTML anchor so the TOC can link to it."""
    st.markdown(f'<div id="{anchor}"></div>', unsafe_allow_html=True)
    st.subheader(text)

# ---------------------------------------------------------
# Navigation model
# ---------------------------------------------------------
SECTIONS = [
    {"id": "home",                     "label": "Home"},
    {"id": "quickstart",               "label": "Quickstart"},
    {"id": "data_requirements",        "label": "Data requirements"},
    {"id": "api_init_fit",             "label": "API: init & fit"},
    {"id": "api_importance_shap",      "label": "API: importance & SHAP"},
    {"id": "api_contrastive",          "label": "API: contrastive importance"},
    {"id": "api_distributions",        "label": "API: distributions"},
    {"id": "api_narratives_summaries", "label": "API: narratives & summaries"},
    {"id": "api_splits_exports",       "label": "API: splits & exports"},
    {"id": "under_the_hood",           "label": "Under the hood"},
]

SECTION_SEARCH = {
    "home": """
        overview introduction clusters clusterlens segmentation interpretability
        one-vs-rest classifiers shap narratives exports
    """,
    "quickstart": """
        quickstart example ClusterAnalyzer ca.fit
        get_cluster_classification_stats get_top_shap_features
        generate_cluster_narratives
    """,
    "data_requirements": """
        pandas DataFrame numeric features categorical features
        is_numeric_dtype OneHotEncoder LeaveOneOutEncoder CatBoostEncoder
        num_features cat_features cluster_col
    """,
    "api_init_fit": """
        ClusterAnalyzer.__init__ ClusterAnalyzer.fit
        cluster_col encoder onehot loo catboost model_type rf lgbm xgb
        eval_max_n test_size sample_n sample_frac stratify_sample
    """,
    "api_importance_shap": """
        plot_cluster_shap importance_scope positive negative all
        get_cluster_classification_stats get_top_shap_features Abs_SHAP
    """,
    "api_contrastive": """
        contrastive_importance shap effect hybrid weight_shap weight_effect
        min_support Cohen d Cramér V
    """,
    "api_distributions": """
        compare_feature_across_clusters histograms stacked bar auto_log_skew
        log1p skewness
    """,
    "api_narratives_summaries": """
        generate_cluster_narratives get_cluster_summary cluster size share
        nearest cluster Mann-Whitney Cohen d Cramér V
    """,
    "api_splits_exports": """
        get_split_table export_summary save_shap_figs shap_cluster_0
        train test splits summary csv
    """,
    "under_the_hood": """
        nearest_cluster_centroid medians IQR shap.Explainer TreeExplainer
        effect sizes Cramér V
    """,
}

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
    "api_contrastive": [
        {"label": "contrastive_importance", "anchor": "api_contrastive_main"},
        {"label": "Modes", "anchor": "api_contrastive_modes"},
        {"label": "Weights", "anchor": "api_contrastive_weights"},
    ],
    "api_distributions": [
        {"label": "compare_feature_across_clusters", "anchor": "api_distributions_compare"},
    ],
    "api_narratives_summaries": [
        {"label": "generate_cluster_narratives", "anchor": "api_narratives_generate"},
        {"label": "get_cluster_summary", "anchor": "api_narratives_summary"},
    ],
    "api_splits_exports": [
        {"label": "get_split_table", "anchor": "api_splits_table"},
        {"label": "export_summary", "anchor": "api_splits_export"},
        {"label": "save_shap_figs", "anchor": "api_splits_save_shap"},
    ],
}

if "active_section" not in st.session_state:
    st.session_state["active_section"] = "home"

# ---------------------------------------------------------
# Layout: three columns (sticky nav + main content + right TOC)
# ---------------------------------------------------------
col_nav, col_main, col_toc = st.columns([0.22, 0.6, 0.18], gap="small")

# ---------------------- NAV COLUMN -----------------------
with col_nav:
    st.image("clusterlens_logo.png", width=LOGO_WIDTH)

    if SHOW_GITHUB_BADGE:
        stars = get_github_stars()
        stars_text = f"{stars:,}" if stars is not None else "Repo"
        st.markdown(
            f"""
            <a class="gh-btn" href="https://github.com/akthammomani/ClusterLens" target="_blank" style="display:inline-flex;align-items:stretch;margin:.25rem auto 1rem;border-radius:4px;overflow:hidden;border:1px solid #d0d7de;font-size:.8rem;text-decoration:none;color:#111827;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <span style="display:inline-flex;align-items:center;gap:.35rem;padding:.25rem .6rem;background:#f6f8fa;">
                    <span style="font-size:.9rem;"></span>
                    <span>GitHub</span>
                </span>
                <span style="padding:.25rem .6rem;border-left:1px solid #d0d7de;font-variant-numeric:tabular-nums;background:#fff;">{stars_text}</span>
            </a>
            """,
            unsafe_allow_html=True,
        )

    query = st.text_input("", placeholder="Search", label_visibility="collapsed")

    if query:
        q = query.lower()
        def matches(section):
            text = section["label"] + " " + SECTION_SEARCH.get(section["id"], "")
            return q in text.lower()
        filtered_sections = [s for s in SECTIONS if matches(s)] or SECTIONS
    else:
        filtered_sections = SECTIONS

    nav_labels = [s["label"] for s in filtered_sections]
    selected_label = st.radio("", options=nav_labels, label_visibility="collapsed", key="nav_radio")
    selected_id = next(s["id"] for s in filtered_sections if s["label"] == selected_label)
    st.session_state["active_section"] = selected_id

section_id = st.session_state["active_section"]

# ---------------------- MAIN COLUMN ----------------------
with col_main:
    if section_id == "home":
        st.title("ClusterLens")
        st.write(
            "ClusterLens is an interpretability engine for **clustered / segmented data**.\n"
            "You already have clusters - customer segments, user personas, product tiers, risk bands.\n"
            "ClusterLens answers the harder questions:"
        )
        st.markdown(
            """
            - What actually *drives* each cluster?
            - How is Cluster 1 different from Cluster 3 in a statistically meaningful way?
            - Which features make Cluster A “high value” or “high risk” compared to others?
            - How can I turn a big table into cluster narratives that non-ML stakeholders can read?
            """
        )
        st.markdown(
            """
            ClusterLens sits on top of **any clustering method** (k-means, GMM, HDBSCAN,
            rule-based labels, etc.). All it needs is a `DataFrame` with a column that
            holds the cluster labels.
            """
        )

        st.header("Core idea")
        st.markdown(
            """
            At its core, ClusterLens:

            1. Takes your labeled data (`df` with a cluster column).
            2. Builds **one-vs-rest classifiers (OVR)** for each cluster.
            3. Uses **SHAP + effect sizes** to understand what separates clusters.
            4. Exposes a small, opinionated API to:
               - inspect **per-cluster metrics**,
               - rank **feature importance**,
               - compute **contrastive importance** between clusters,
               - plot **distributions**,
               - generate **markdown narratives**,
               - export **summary tables** and **SHAP PNGs**.
            """
        )

    elif section_id == "quickstart":
        st.header("Quickstart")
        st.markdown(
            dedent(
                """
                ```python
                import pandas as pd
                from clusterlens import ClusterAnalyzer

                df = pd.read_csv("my_clustered_table.csv")

                ca = ClusterAnalyzer(
                    df=df,
                    cluster_col="Cluster",
                    # optional: override auto-detected lists
                    # num_features=[...],
                    # cat_features=[...],
                    encoder="onehot",
                    model_type="rf",
                    eval_max_n=5000,
                )

                ca.fit(
                    test_size=0.2,
                    sample_n=None,
                    sample_frac=None,
                    stratify_sample=True,
                )

                metrics = ca.get_cluster_classification_stats()
                top_feats = ca.get_top_shap_features(top_n=5)
                narratives = ca.generate_cluster_narratives(top_n=5)
                ```
                """
            )
        )
        st.markdown(
            """
            Minimal **happy path** steps shown above.
            """
        )

    elif section_id == "data_requirements":
        st.header("Data requirements")
        st.markdown(
            """
            ClusterLens expects a DataFrame with a cluster label column and any mix of numeric/categorical features.
            """
        )

    elif section_id == "api_init_fit":
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

    elif section_id == "api_importance_shap":
        st.header("API: importance & SHAP")

        subheader_with_anchor("plot_cluster_shap", "api_importance_plot")
        st.markdown(
            dedent(
                """
                ```python
                ca.plot_cluster_shap(
                    top_n: Optional[int] = None,
                    importance_scope: str = "positive",  # "positive" | "negative" | "all"
                    show: bool = True,
                )
                ```
                """
            )
        )

        subheader_with_anchor("get_cluster_classification_stats", "api_importance_stats")
        st.markdown("`stats = ca.get_cluster_classification_stats()`")

        subheader_with_anchor("get_top_shap_features", "api_importance_top_feats")
        st.markdown(
            dedent(
                """
                ```python
                top_feats = ca.get_top_shap_features(
                    top_n: Optional[int] = None,
                    importance_scope: str = "positive",
                )
                ```
                """
            )
        )

    elif section_id == "api_contrastive":
        st.header("API: contrastive importance")

        subheader_with_anchor("contrastive_importance", "api_contrastive_main")
        st.markdown(
            dedent(
                """
                ```python
                ca.contrastive_importance(
                    cluster_a,
                    cluster_b,
                    top_n: Optional[int] = None,
                    importance_scope: str = "positive",
                    mode: str = "hybrid",     # "shap" | "effect" | "hybrid"
                    weight_shap: float = 1.0,
                    weight_effect: float = 1.0,
                    min_support: float = 0.0,
                )
                ```
                """
            )
        )

        subheader_with_anchor("Modes", "api_contrastive_modes")
        st.markdown("Use SHAP, effect sizes, or a hybrid score.")

        subheader_with_anchor("Weights", "api_contrastive_weights")
        st.markdown("Adjust weights to bias towards model or raw distribution contrasts.")

    elif section_id == "api_distributions":
        st.header("API: distributions")

        subheader_with_anchor(
            "compare_feature_across_clusters", "api_distributions_compare"
        )
        st.markdown(
            dedent(
                """
                ```python
                ca.compare_feature_across_clusters(
                    feature: Optional[str] = None,
                    bins: int = 30,
                    auto_log_skew: Optional[float] = None,
                    linewidth: float = 1.5,
                    alpha: float = 0.9,
                )
                ```
                """
            )
        )

    elif section_id == "api_narratives_summaries":
        st.header("API: narratives & summaries")

        subheader_with_anchor(
            "generate_cluster_narratives", "api_narratives_generate"
        )
        st.markdown(
            dedent(
                """
                ```python
                narratives = ca.generate_cluster_narratives(
                    top_n: Optional[int] = None,
                    min_support: float = 0.05,
                    output: str = "markdown",   # "markdown" | "dict"
                )
                ```
                """
            )
        )

        subheader_with_anchor("get_cluster_summary", "api_narratives_summary")
        st.markdown(
            dedent(
                """
                ```python
                summary = ca.get_cluster_summary(
                    sample_size: Optional[int] = None,
                    top_n_contrast: Optional[int] = None,
                    min_support: float = 0.05,
                )
                ```
                """
            )
        )

    elif section_id == "api_splits_exports":
        st.header("API: splits & exports")

        subheader_with_anchor("get_split_table", "api_splits_table")
        st.markdown("`split_tbl = ca.get_split_table()`")

        subheader_with_anchor("export_summary", "api_splits_export")
        st.markdown('`ca.export_summary("cluster_summary.csv")`')

        subheader_with_anchor("save_shap_figs", "api_splits_save_shap")
        st.markdown(
            dedent(
                """
                ```python
                ca.plot_cluster_shap(top_n=10, importance_scope="positive")
                ca.save_shap_figs("./shap_figs")
                ```
                """
            )
        )

    elif section_id == "under_the_hood":
        st.header("Under the hood")
        st.markdown(
            """
            Medians + IQR scaling for numeric contrasts, lifts + Cramér's V for categorical, OVR SHAP extraction.
            """
        )

        st.markdown("---\nQuestions or ideas? Open an issue in the ClusterLens repo. 🚀")

# ---------------------- RIGHT TOC COLUMN -----------------
with col_toc:
    items = TOC_ITEMS.get(section_id, [])
    if items:
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        st.markdown("###### On this page")
        for item in items:
            st.markdown(f"- [{item['label']}](#{item['anchor']})")
