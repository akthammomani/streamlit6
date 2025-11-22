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

# Toggle for GitHub badge
SHOW_GITHUB_BADGE = False

# ---------------------------------------------------------
# Global CSS: Layout, fixed sidebars, logo size, etc.
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Streamlit's native header is 3.5rem tall. Using a variable for cleaner code. */
    :root {
        --header-height: 3.5rem; /* ~56px */
        --left-nav-width: 250px; /* New fixed width for left sidebar */
        --right-toc-width: 200px; /* New fixed width for right sidebar */
    }

    /* 1. FIX CLIPPING & BASE LAYOUT */
    /* Override Streamlit's default padding on the top-level main content wrapper */
    .main > div {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Ensure the main container of your three columns is full width and has no internal padding */
    .block-container {
        padding-top: 0;
        padding-bottom: 0;
        padding-left: 0;
        padding-right: 0;
        max-width: 1900px;
    }

    /* 2. Fixed Left Sidebar (column 1) */
    div[data-testid="column"]:nth-of-type(1) {
        position: fixed;
        top: var(--header-height);
        left: 0;
        width: var(--left-nav-width); /* Reduced width */
        height: calc(100vh - var(--header-height));
        padding: 1rem 1rem 2rem 1rem; /* Reduced horizontal padding */
        border-right: 1px solid #e5e7eb;
        background-color: #f3f4f6;
        overflow-y: auto;
        z-index: 100;
    }

    /* 3. Fixed Right "On this page" sidebar (column 3) */
    div[data-testid="column"]:nth-of-type(3) {
        position: fixed;
        top: var(--header-height);
        right: 0;
        width: var(--right-toc-width); /* Reduced width */
        height: calc(100vh - var(--header-height));
        padding: 1rem 1rem 2rem 1rem; /* Reduced horizontal padding */
        border-left: 1px solid #e5e7eb;
        background-color: #ffffff;
        overflow-y: auto;
        z-index: 100;
    }

    /* 4. Main content (column 2) - Adjust margins for fixed sidebars' full width */
    div[data-testid="column"]:nth-of-type(2) {
        /* Left margin = left-nav-width + left-nav-padding (1rem) + right-nav-padding (1rem) */
        margin-left: calc(var(--left-nav-width) + 2rem);
        /* Right margin = right-toc-width + left-toc-padding (1rem) + right-toc-padding (1rem) */
        margin-right: calc(var(--right-toc-width) + 2rem);
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: var(--header-height); /* Push content down past fixed header */
    }

    /* 5. Reduced Logo Size (left column) */
    div[data-testid="column"]:nth-of-type(1) img {
        max-width: 150px; /* Reduced logo size */
        display: block;
        margin-left: auto;
        margin-right: auto;
        height: auto;
    }

    div[data-testid="column"]:nth-of-type(1) div[data-testid="stImage"] {
        margin-bottom: 0.75rem;
    }

    /* The rest of your existing CSS (GitHub, Search, Radio, TOC tweaks) goes here, unchanged. */

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

    .gh-btn:hover {
        background-color: #f6f8fa;
    }

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

    .gh-icon {
        font-size: 0.9rem;
    }

    /* ===== Search box styling (left column) ===== */
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] {
        position: relative;
        margin: 0.25rem 0 1.25rem 0;
    }

    /* Remove grey outer pill around the input wrapper */
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] > div {
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* Hide the label text of the search input */
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] label {
        display: none;
    }

    /* Search input itself */
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"] input {
        border-radius: 999px;
        border: 1px solid #d1d5db;
        padding: 0.35rem 0.9rem 0.35rem 2rem;  /* left space for icon */
        font-size: 0.9rem;
        background-color: #ffffff;  /* white pill */
    }

    /* Magnifying glass icon */
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stTextInput"]::before {
        content: "🔍";
        position: absolute;
        left: 0.6rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 0.85rem;
        color: #9ca3af;
        pointer-events: none;
    }

    /* ===== Docs-style nav (radio but no round dots) ===== */

    /* Completely hide the built-in radio label text ("Sections") */
    div[data-testid="stRadio"] > label {
        display: none !important;
    }

    /* Container that holds all options */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
    }

    /* Each option row */
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        padding: 4px 10px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.95rem;
        font-weight: 400;
        color: #374151;
    }

    /* Hide the circular radio icon */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Text container inside label */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:last-child {
        width: 100%;
    }

    /* Selected state */
    div[data-testid="stRadio"] div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
        background-color: #eff6ff;
        border-left: 3px solid #2563eb;
        color: #111827;
        font-weight: 600;
    }

    /* Hover state */
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #e5e7eb;
    }

    /* ===== Right "On this page" sidebar text tweaks ===== */

    /* Your "On this page" title (you use ###### => h6) */
    div[data-testid="column"]:nth-of-type(3) h6 {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: none;
        margin-bottom: 0.15rem;
        color: #4b5563;
    }

    /* Bullet list inside right column */
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

    /* Headings scroll offset so anchors are not hidden under header */
    h1, h2, h3, h4, h5, h6 {
        scroll-margin-top: 1.5rem;
    }

    pre, code {
        font-size: 0.9rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Small helpers (omitted for brevity, assume they are correct)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_github_stars():
    # ... (implementation omitted)
    return None

def subheader_with_anchor(text: str, anchor: str):
    # ... (implementation omitted)
    st.markdown(f'<div id="{anchor}"></div>', unsafe_allow_html=True)
    st.subheader(text)


# ---------------------------------------------------------
# Navigation model (omitted for brevity, assume it is correct)
# ---------------------------------------------------------
SECTIONS = [
    {"id": "home", "label": "Home"},
    {"id": "quickstart", "label": "Quickstart"},
    # ... (rest of sections)
]
SECTION_SEARCH = {
    # ... (search content)
}
TOC_ITEMS = {
    # ... (TOC content)
}

if "active_section" not in st.session_state:
    st.session_state["active_section"] = "home"

# ---------------------------------------------------------
# Layout: three columns (fixed nav + main content + right TOC)
# ---------------------------------------------------------
# The column ratios are now less important as width is controlled by CSS,
# but it's good practice to keep them roughly aligned.
col_nav, col_main, col_toc = st.columns([0.22, 0.6, 0.18])

# ---------------------- NAV COLUMN -----------------------
with col_nav:
    st.image("clusterlens_logo.png")
    # ... (rest of left column content)
    # GitHub stars button (toggle with SHOW_GITHUB_BADGE)
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

    # Search box
    query = st.text_input(
        label="",
        placeholder="Search",
        label_visibility="collapsed",
    )

    # Filter sections by query across label + indexed content
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
        label="",
        options=nav_labels,
        label_visibility="collapsed",
        key="nav_radio",
    )

    selected_id = next(
        s["id"] for s in filtered_sections if s["label"] == selected_label
    )
    st.session_state["active_section"] = selected_id

# Current section id
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
        # ... (Quickstart content)
        st.header("Quickstart")
        st.markdown(
            dedent(
                """
                ```python
                import pandas as pd
                from clusterlens import ClusterAnalyzer
                
                # ... (rest of quickstart code)
                ```
                """
            )
        )
        st.markdown(
            """
            That is the minimal **happy path**:
            
            - Instantiate `ClusterAnalyzer` with your `DataFrame`.
            - Call `.fit()` once to train all OVR models and cache SHAP.
            - Use the small API surface to inspect metrics, features, narratives,
              and exports.
            """
        )
    
    # ... (rest of main column content for other sections)
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
                    encoder: str = "onehot",       # "onehot" | "loo" | "catboost"
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
            - **`cluster_col`**: Name of the column with your cluster labels.
            - **`encoder`**:
              - `"onehot"`: Fast, robust, good default.
              - `"loo"`: Target-encoding style; can be nicer for high-cardinality cats.
              - `"catboost"`: Similar idea, different smoothing; needs `category_encoders`.
            - **`model_type`**:
              - `"rf"`: `RandomForestClassifier` (default, zero-config).
              - `"lgbm"`: LightGBM, if you install `lightgbm`.
              - `"xgb"`: XGBoost, if you install `xgboost`.
            - **`eval_max_n`**: Cap the number of rows used for SHAP evaluation
              per cluster. Use this when your test set is huge and SHAP is slow.

            If you don't pass `num_features` / `cat_features`, ClusterLens will
            infer them from dtypes and avoid double-counting any column.
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
            - **`test_size`**: Fraction of rows reserved for evaluation for
              each OVR classifier.
            - **`sample_n` / `sample_frac`**:
              - If `None`: train on the full `df`.
              - If set: ClusterLens first **subsamples** the dataframe
                (optionally stratified by cluster), *then* splits into train/test.
              - Use this when your table is huge and you want fast iterations.
            - **`stratify_sample`**:
              - If `True` (default): preserves cluster proportions in the sample.
              - If `False`: draws a simple random subset.

            `.fit()`:

            1. Optionally subsamples the DataFrame.
            2. Fits the chosen categorical encoder.
            3. Builds a single stratified train/test split.
            4. Trains **one-vs-rest** classifiers for each cluster.
            5. Caches SHAP values for each OVR model.
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
            **`importance_scope`: Which rows feed SHAP**

            ClusterLens stores SHAP values for each OVR model along with the
            binary label `y_eval_bin` (1 = belongs to the target cluster;
            0 = all other rows).

            - `"positive"` (default):
              - Uses only rows where `y_eval_bin == 1`.
              - Focuses on **why points inside the cluster** look the way they do.
              - Best when you want to describe the **internal signature** of a cluster.
            - `"negative"`:
              - Uses only rows where `y_eval_bin == 0` (all other clusters).
              - Reads as: features that **keep points out of this cluster**.
              - Good for debugging: "what repels points from Cluster A?”.
            - `"all"`:
              - Uses the full evaluation set.
              - More of a **global discriminative view** for that OVR classifier
                (positive vs. the rest combined).

            If you are unsure: keep `"positive"` for interpretability decks, and
            try `"negative"` / `"all"` as diagnostic lenses when something looks off.
            """
        )

        subheader_with_anchor("get_cluster_classification_stats", "api_importance_stats")
        st.markdown(
            dedent(
                """
                ```python
                stats = ca.get_cluster_classification_stats()
                ```
                Returns one row per cluster with:

                - Accuracy, Precision, Recall, F1, ROC_AUC
                - Confusion matrix counts: TN, FP, FN, TP
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
            - Aggregates SHAP from encoded columns back to **original features**.
            - Returns a long DataFrame with columns:
              `Cluster`, `Feature` and `Abs_SHAP`.
            - Use `top_n` to keep only the top features per cluster; or `None`
              to keep all of them.
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
                    mode: str = "hybrid",       # "shap" | "effect" | "hybrid"
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
            This compares **Cluster A vs Cluster B** and scores features by how strongly they separate the two.
            """
        )

        subheader_with_anchor("Modes", "api_contrastive_modes")
        st.markdown(
            """
            - `"shap"`: Uses only normalized SHAP magnitudes for each feature.
            - `"effect"`: Uses only statistical contrasts:
              - numeric: Standardized median gaps (in IQR units) + |Cohen's d|.
              - categorical: Best lift + Cramér's V.
            - `"hybrid"` (default): Adds both pieces:

            `score = weight_shap * SHAP_norm + weight_effect * EFFECT_norm`
            """
        )

        subheader_with_anchor("Weights", "api_contrastive_weights")
        st.markdown(
            """
            - Increase `weight_shap` if you trust the OVR models and want a model-centric view.
            - Increase `weight_effect` if you care more about distribution shifts in the raw data and want something closer to pure stats.

            **`min_support` (categorical)**

            - Ignores categories whose cluster share is below `min_support`.
            - Use e.g. `min_support=0.05` to focus on categories that cover at least 5% of the cluster.

            The result is a DataFrame sorted by `Score`, with extra columns exposing each normalized component so you can debug the ranking.
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
            - With `feature=None`:
              - Plots faceted histograms for **all numeric features**.
            - With `feature="col"`:
              - If `col` is numeric: overlaid **step histograms** (raw counts)
                per cluster.
              - If `col` is categorical: a **stacked bar chart** with counts
                per cluster.

            **`auto_log_skew`**

            - If `None`: keeps the raw scale.
            - If a value (e.g. `1.5`):
              - For skewed non-negative numeric features (|skew| > 1.5),
                applies `log1p` before plotting.
              - This helps make long-tailed metrics visually comparable.
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
                    output: str = "markdown",  # "markdown" | "dict"
                )
                ```
                """
            )
        )
        st.markdown(
            """
            Produces **human-readable bullets** per cluster:

            - Cluster size & share of the dataset.
            - High/low numeric drivers:
              - median gaps in IQR units.
              - Cohen's d.
              - Mann-Whitney p-values.
            - Dominant categories with lifts & Cramér's V.
            - Key differences vs the **nearest cluster** in numeric space.
            - `top_n`: Limits how many numeric & categorical bullets you keep
              per cluster.
            - `min_support`: Drops rare categories when building categorical
              bullets.
            - `output="markdown"`: Returns ready-to-paste markdown strings.
            - `output="dict"`: Returns structured dicts if you want to build
              your own UI.

            These are designed to go straight into **slide decks or reports**
            without extra massaging.
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
            Returns a compact **table** with one row per cluster:

            - `N`, `%` of dataset,
            - Compact per-feature summaries (mean/median with deltas vs global),
            - Short contrastive strings vs the nearest cluster (numeric + cat).

            Use this when you want an **overview table** next to plots.
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
            Shows how the train/test split looks per cluster:

            - How many positives & “rest” rows ended up in train vs test;
            """
        )

    # The rest of the main content...

# ---------------------- TOC COLUMN ----------------------
with col_toc:
    # Only render TOC if the current section has items defined
    if section_id in TOC_ITEMS:
        st.markdown("###### On this page")
        toc_list = "\n".join(
            f'- <a href="#{item["anchor"]}">{item["label"]}</a>'
            for item in TOC_ITEMS[section_id]
        )
        st.markdown(toc_list, unsafe_allow_html=True)
