import streamlit as st
from textwrap import dedent
import requests

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="ClusterLens Documentation",
    page_icon="clusterlens_logo.png",
    layout="wide",
)

SHOW_GITHUB_BADGE = False  # toggle GitHub button

# ---------------------------------------------------------
# Global CSS: layout + styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* ===== Overall page width & padding ===== */
    .block-container {
        padding-top: 0;
        padding-left: 0;
        padding-right: 0;
        max-width: 1800px;     /* wider, closer to local look */
        margin: 0 auto;
    }

    /* Fixed left sidebar (first column) */
    div[data-testid="column"]:nth-of-type(1) {
        position: fixed;
        top: 3.5rem;                 /* just under Streamlit header */
        left: 0;
        width: 260px;                /* left nav width */
        height: calc(100vh - 3.5rem);
        padding: 1rem 1.5rem 2rem 1.5rem;
        border-right: 1px solid #e5e7eb;
        background-color: #f3f4f6;
        overflow-y: auto;
        z-index: 100;
    }

    /* Main content (second column) */
    div[data-testid="column"]:nth-of-type(2) {
        margin-left: 260px;          /* reserve space for left nav */
        margin-right: 230px;         /* reserve space for TOC */
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 3.5rem;
    }

    /* Fixed right TOC sidebar (third column) */
    div[data-testid="column"]:nth-of-type(3) {
        position: fixed;
        top: 3.5rem;
        right: 0;
        width: 230px;                /* right sidebar width */
        height: calc(100vh - 3.5rem);
        padding: 1rem 1.5rem 2rem 1.5rem;
        border-left: 1px solid #e5e7eb;
        background-color: #ffffff;
        overflow-y: auto;
        z-index: 100;
    }

    /* Center logo in left column */
    div[data-testid="column"]:nth-of-type(1) img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stImage"] {
        margin-bottom: 0.75rem;
    }

    /* ===== GitHub button (optional) ===== */
    .gh-btn {
        display: inline-flex;
        align-items: stretch;
        margin: 0.25rem auto 1rem auto;
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid #d0d7de;
        font-size: 0.8rem;
        text-decoration: none;
        color: #111827;
        background-color: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .gh-btn:hover { background-color: #f6f8fa; }
    .gh-left {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.6rem;
        background-color: #f6f8fa;
    }
    .gh-right {
        padding: 0.25rem 0.6rem;
        border-left: 1px solid #d0d7de;
        font-variant-numeric: tabular-nums;
        background-color: #ffffff;
    }
    .gh-icon { font-size: 0.9rem; }

    /* ===== Search box styling (left column) ===== */
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] {
        position: relative;
        margin: 0.25rem 0 1.25rem 0;
    }
    /* Remove outer pill */
    div[data-testid="column"]:nth-of-type(1)
      div[data-testid="stTextInput"] > div {
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    /* Hide visual label (we keep it non-empty for accessibility) */
    div[data-testid="column"]:nth-of-type(1)
      div[data-testid="stTextInput"] label {
        display: none;
    }
    /* Input itself */
    div[data-testid="column"]:nth-of-type(1)
      div[data-testid="stTextInput"] input {
        border-radius: 999px;
        border: 1px solid #d1d5db;
        padding: 0.35rem 0.9rem 0.35rem 2rem;
        font-size: 0.9rem;
        background-color: #ffffff;
    }
    /* Magnifying glass icon */
    div[data-testid="column"]:nth-of-type(1)
      div[data-testid="stTextInput"]::before {
        content: "🔍";
        position: absolute;
        left: 0.6rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 0.85rem;
        color: #9ca3af;
        pointer-events: none;
    }

    /* ===== Radio nav: make it look like docs menu ===== */

    /* Hide the radio group label visually (again, keep it non-empty) */
    div[data-testid="stRadio"] > label {
        display: none !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        padding: 4px 10px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.95rem;
        font-weight: 400;
        color: #374151;
    }

    /* Hide circular radio icon */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Selected state */
    div[data-testid="stRadio"] div[role="radiogroup"]
      > label[data-baseweb="radio"]:has(input:checked) {
        background-color: #eff6ff;
        border-left: 3px solid #2563eb;
        color: #111827;
        font-weight: 600;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #e5e7eb;
    }

    /* ===== Right TOC ===== */
    div[data-testid="column"]:nth-of-type(3) h6 {
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
        color: #4b5563;
    }
    div[data-testid="column"]:nth-of-type(3) ul {
        list-style-type: disc;
        padding-left: 1.1rem;
        margin: 0;
    }
    div[data-testid="column"]:nth-of-type(3) li {
        margin: 0;
        padding: 0;
        line-height: 1.1;
    }
    div[data-testid="column"]:nth-of-type(3) li a {
        font-size: 0.8rem;
        text-decoration: none;
        color: #2563eb;
    }
    div[data-testid="column"]:nth-of-type(3) li a:hover {
        text-decoration: underline;
    }

    /* Headings scroll offset so anchors are visible */
    h1, h2, h3, h4, h5, h6 {
        scroll-margin-top: 1.5rem;
    }

    pre, code {
        font-size: 0.9rem !important;
    }

    /* ===== Simple responsive fallback for narrower screens ===== */
    @media (max-width: 1200px) {
        div[data-testid="column"]:nth-of-type(1),
        div[data-testid="column"]:nth-of-type(3) {
            position: static;
            width: 100%;
            height: auto;
            border: none;
        }
        div[data-testid="column"]:nth-of-type(2) {
            margin-left: 0;
            margin-right: 0;
            padding-top: 1.5rem;
        }
    }
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
# Layout: three columns
# ---------------------------------------------------------
col_nav, col_main, col_toc = st.columns([0.2, 0.6, 0.2])

# ---------------------- NAV COLUMN -----------------------
with col_nav:
    st.image("clusterlens_logo.png")

    if SHOW_GITHUB_BADGE:
        stars = get_github_stars()
        stars_text = f"{stars:,}" if stars is not None else "Repo"
        st.markdown(
            f"""
            <a class="gh-btn" href="https://github.com/akthammomani/ClusterLens" target="_blank">
                <span class="gh-left">
                    <span class="gh-icon"></span>
                    <span>GitHub</span>
                </span>
                <span class="gh-right">{stars_text}</span>
            </a>
            """,
            unsafe_allow_html=True,
        )

    query = st.text_input(
        "Search sections",
        placeholder="Search",
        label_visibility="collapsed",   # text is still non-empty -> no warning
    )

    if query:
        q = query.lower()

        def matches(section):
            text = section["label"] + " " + SECTION_SEARCH.get(section["id"], "")
            return q in text.lower()

        filtered_sections = [s for s in SECTIONS if matches(s)] or SECTIONS
    else:
        filtered_sections = SECTIONS

    nav_labels = [s["label"] for s in filtered_sections]

    selected_label = st.radio(
        "Sections",
        options=nav_labels,
        label_visibility="collapsed",   # again, keep non-empty but hide visually
        key="nav_radio",
    )

    selected_id = next(
        s["id"] for s in filtered_sections if s["label"] == selected_label
    )
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
                    encoder="onehot",      # or "loo" / "catboost"
                    model_type="rf",       # or "lgbm" / "xgb"
                    eval_max_n=5000,       # cap SHAP eval rows per cluster
                )

                ca.fit(
                    test_size=0.2,
                    sample_n=None,         # or e.g. 50_000 for big tables
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
            Minimal happy path:

            - Instantiate `ClusterAnalyzer` with your `DataFrame`.
            - Call `.fit()` once to train all OVR models and cache SHAP.
            - Use the small API surface to inspect metrics, features, narratives,
              and exports.
            """
        )

    elif section_id == "data_requirements":
        st.header("Data requirements")
        st.markdown(
            """
            ClusterLens expects:

            - A **pandas DataFrame**.
            - A column with cluster labels (default name `"Cluster"`).
            - Any mix of numeric and categorical features.

            **Numeric features**

            - Auto-detected via `pandas.api.types.is_numeric_dtype`.
            - Used for SHAP, effect sizes, distributions, contrastive stats.

            **Categorical features**

            - Everything that is *not* numeric is treated as categorical
              (unless you override `num_features` / `cat_features`).
            - Encoded via:
              - `OneHotEncoder` (default, safe and interpretable),
              - `LeaveOneOutEncoder` (`encoder="loo"`),
              - `CatBoostEncoder` (`encoder="catboost"`).

            You can always pass explicit lists:

            ```python
            ca = ClusterAnalyzer(
                df,
                num_features=["age", "tenure_days", "spend"],
                cat_features=["region", "channel"],
            )
            ```
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
                    encoder: str = "onehot",      # "onehot" | "loo" | "catboost"
                    model_type: str = "rf",        # "rf" | "lgbm" | "xgb"
                    model_params: Optional[dict] = None,
                    eval_max_n: Optional[int] = None,
                )
                ```
                """
            )
        )
        st.markdown(
            """
            - **`cluster_col`**: column with cluster labels.
            - **`encoder`**:
              - `"onehot"`: fast, robust default.
              - `"loo"` / `"catboost"`: target-style encoders for high-cardinality cats.
            - **`model_type`**:
              - `"rf"`: `RandomForestClassifier` (default).
              - `"lgbm"`: LightGBM (install `lightgbm`).
              - `"xgb"`: XGBoost (install `xgboost`).
            - **`eval_max_n`**: cap SHAP evaluation rows per cluster.

            If you don't pass `num_features` / `cat_features`, ClusterLens infers them
            from dtypes and avoids double-counting any column.
            """
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
        st.markdown(
            """
            - **`test_size`**: fraction of rows reserved for evaluation.
            - **`sample_n` / `sample_frac`**:
              - `None`: use full dataframe.
              - Set one of them to subsample before splitting.
            - **`stratify_sample`**:
              - `True`: keeps cluster proportions when sampling.

            `.fit()`:

            1. Optional subsample.
            2. Fit encoder.
            3. One train/test split.
            4. Train OVR models per cluster.
            5. Cache SHAP values.
            """
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
        st.markdown(
            """
            **`importance_scope`** controls which rows feed SHAP:

            - `"positive"` (default): only rows belonging to the cluster.
            - `"negative"`: rows from all *other* clusters.
            - `"all"`: full evaluation set.

            Use `"positive"` for narratives, `"negative"` / `"all"` for debugging.
            """
        )

        subheader_with_anchor("get_cluster_classification_stats", "api_importance_stats")
        st.markdown(
            dedent(
                """
                ```python
                stats = ca.get_cluster_classification_stats()
                ```
                Returns one row per cluster:

                - Accuracy, Precision, Recall, F1, ROC_AUC
                - TN, FP, FN, TP
                """
            )
        )

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
        st.markdown(
            """
            Aggregates SHAP from encoded columns back to original feature names and
            returns `Cluster`, `Feature`, `Abs_SHAP`.
            """
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
        st.markdown(
            """
            Compares **Cluster A vs Cluster B** and scores features by how strongly
            they separate the two.
            """
        )

        subheader_with_anchor("Modes", "api_contrastive_modes")
        st.markdown(
            """
            - `"shap"`: only normalized SHAP magnitudes.
            - `"effect"`: only statistical contrasts  
              (numeric gaps in IQR units + |Cohen's d|; categorical lift + Cramér's V).
            - `"hybrid"`: weighted sum of both.
            """
        )

        subheader_with_anchor("Weights", "api_contrastive_weights")
        st.markdown(
            """
            - Increase `weight_shap` for a more model-centric view.
            - Increase `weight_effect` for a more distribution/statistics-centric view.
            - `min_support` drops very rare categories when computing lifts.
            """
        )

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
        st.markdown(
            """
            - `feature=None`: faceted histograms for all numeric features.
            - Numeric feature: overlaid step histograms per cluster.
            - Categorical feature: stacked bar chart (counts).

            `auto_log_skew` = threshold for applying `log1p` to skewed non-negative
            features (e.g. `1.5`).
            """
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
        st.markdown(
            """
            Produces markdown bullets per cluster: size, high/low numeric drivers,
            dominant categories, and differences vs nearest cluster.
            """
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
        st.markdown(
            """
            Returns a compact table per cluster:

            - `N`, `%` of dataset,
            - per-feature summaries,
            - short contrastive strings vs nearest cluster.
            """
        )

    elif section_id == "api_splits_exports":
        st.header("API: splits & exports")

        subheader_with_anchor("get_split_table", "api_splits_table")
        st.markdown(
            dedent(
                """
                ```python
                split_tbl = ca.get_split_table()
                ```
                """
            )
        )
        st.markdown(
            """
            Shows how train/test split looks for each cluster: one vs rest counts and
            train share.
            """
        )

        subheader_with_anchor("export_summary", "api_splits_export")
        st.markdown(
            dedent(
                """
                ```python
                ca.export_summary("cluster_summary.csv")
                ```
                """
            )
        )
        st.markdown("Convenience wrapper around `get_cluster_summary()` that writes CSV.")

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
        st.markdown(
            """
            Saves one SHAP PNG per cluster (e.g. `shap_cluster_0.png`) – handy for decks
            and emails.
            """
        )

    elif section_id == "under_the_hood":
        st.header("Under the hood")
        st.markdown(
            """
            - `_nearest_cluster_centroid` uses median numeric profiles.
            - Numeric contrasts use medians + IQR-based scaling.
            - Categorical contrasts combine lift + Cramér's V.
            - SHAP extractor tries `shap.Explainer` then falls back to `TreeExplainer`.

            The goal is a small, stable API on top of numerically honest internals.
            """
        )
        st.markdown(
            "---\nQuestions or ideas for new knobs? Open an issue in the ClusterLens repo. 🚀"
        )

# ---------------------- RIGHT TOC COLUMN -----------------
with col_toc:
    items = TOC_ITEMS.get(section_id, [])
    if items:
        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown("###### On this page")
        for item in items:
            st.markdown(f"- [{item['label']}](#{item['anchor']})")
