import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from timeseries_utils import (
    load_and_prepare, filter_df, build_series, evaluate_models
)
import os
# from statsmodels.graphics.tsaplots import plot_acf

DATA_PATHS = [
    os.path.join(os.path.dirname(__file__), "bayer_final_1.csv"),
    os.path.join(os.path.dirname(__file__), "bayer_final_2.csv"),
    os.path.join(os.path.dirname(__file__), "bayer_final_3.csv"),
]
LOGO_PATH = os.path.join(os.path.dirname(__file__), "logo.svg")

st.set_page_config(page_title="Time Series Forecasting", layout="wide")

# Use a modern, readable style and deterministic color palette
plt.style.use('seaborn-v0_8')
PALETTE = plt.cm.tab10.colors  # 10 distinct, accessible colors

# Style the "Run models" button
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #89D329 !important;
        color: #ffffff !important;
        border: none !important;
    }
    /* Button hover */
    div.stButton > button:hover {
        background-color: #89D329 !important;
    }
    
    /* Header bottom border */
    .block-container h1 {
        border-bottom: 2px solid #F1F2F6 !important;
        padding-bottom: 8px !important;
        margin-bottom: 12px !important;
    }

    /* Global font color across the app */
    html, body, .stApp, .block-container {
        color: #10384F !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, .stText, .stCaption {
        color: #10384F !important;
    }

    /* Selected option background + font in dropdown (Select/Multiselect) */
    /* Selected item in the listbox */
    div[data-baseweb="select"] [role="listbox"] > div[aria-selected="true"] {
        background-color: #10384F !important;
        color: #ffffff !important;
    }
    /* Fallback: selected nested element */
    div[data-baseweb="select"] [role="listbox"] [aria-selected="true"] {
        background-color: #10384F !important;
        color: #ffffff !important;
    }
    /* Fallback: BaseWeb may use data-selected */
    div[data-baseweb="select"] [role="listbox"] [data-selected="true"] {
        background-color: #10384F !important;
        color: #ffffff !important;
    }
    /* Keyboard focus highlight */
    div[data-baseweb="select"] [role="listbox"] > div:focus {
        background-color: #10384F !important;
        color: #ffffff !important;
        outline: none !important;
    }

    /* Hover state for dropdown options */
    div[data-baseweb="select"] [role="listbox"] > div:hover {
        background-color: #89D329 !important;
        color: #ffffff !important;
    }

    /* Selected tags/chips shown in Multiselect input */
    /* Use attribute-only selector to cover any element type */
    [data-baseweb="tag"] {
        background-color: #10384F !important;
        color: #ffffff !important;
        border: none !important;
    }
    [data-baseweb="tag"] span {
        color: #ffffff !important;
    }
    /* Close icon on tags */
    [data-baseweb="tag"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    /* Ensure tags inside Select get same styling */
    div[data-baseweb="select"] [data-baseweb="tag"] {
        background-color: #10384F !important;
        color: #ffffff !important;
        border: none !important;
    }
    /* Icons/checkmarks inside options: use brand green */
    div[data-baseweb="select"] [role="listbox"] svg {
        fill: #10384F !important;
        color: #10384F !important;
        stroke: #10384F !important;
    }
    div[data-baseweb="select"] [role="listbox"] svg path,
    div[data-baseweb="select"] [role="listbox"] svg circle,
    div[data-baseweb="select"] [role="listbox"] svg polyline {
        fill: #10384F !important;
        stroke: #10384F !important;
    }

    /* Control border: neutral by default; brand green on focus/open */
    div[data-baseweb="select"] > div {
        border-color: #DDE3EA !important; /* neutral */
        border-width: 1px !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-baseweb="select"] > div:focus,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="select"][aria-expanded="true"] > div {
        border-color: #10384F !important;
        border-width: 1px !important;
        box-shadow: inset 0 0 0 1px #10384F !important; /* thinner active border */
        outline: none !important;
    }
    .green-header {
        color: #89D329 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    dfs = [load_and_prepare(path) for path in DATA_PATHS]
    return pd.concat(dfs, ignore_index=True)

df = load_data()
# Inline logo before the main heading

def format_indian_number(n):
    s = str(abs(int(n)))
    if len(s) <= 3:
        return s
    else:
        # Last 3 digits
        last3 = s[-3:]
        rest = s[:-3]
        # Split rest into groups of 2
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        return ','.join(parts + [last3])

try:
    with open(LOGO_PATH, "r", encoding="utf-8") as _f:
        _svg = _f.read()
    # Constrain logo size for header row
    _svg = _svg.replace("<svg ", "<svg style=\"height:36px;width:36px;display:block;\" ")
    header_html = (
        "<div style=\"display:flex; align-items:center; gap:10px;\">"
        + _svg
        + "<h1 style=\"margin:0;\">Bayer Time Series Forecasting</h1>"
        + "</div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)
except Exception:
    # Fallback to standard title if logo not available
    st.title("Bayer Time Series Forecasting")


countries = sorted(df["Country"].dropna().unique().tolist())
cats_all = sorted(df["Global_CAT"].dropna().unique().tolist())
has_bch = "BCH" in df.columns
bchs = (sorted(df["BCH"].dropna().unique().tolist(), key=lambda x: 0 if str(x).strip().lower() == "yes" else 1) if has_bch else [])

has_seg = "Global_Segment" in df.columns
has_prod = "Product" in df.columns

ALL = "Select All"

col1, col2, col3, col4, col5 = st.columns(5)


# 1) Countries
with col1:
    country_options = [ALL] + countries
    sel_countries = st.multiselect(
        "Countries",
        options=country_options,
        default=[ALL],
        key="sel_countries"
    )
eff_countries = countries if ALL in sel_countries or len(sel_countries) == 0 else sel_countries

# 2) Bayer (BCH) selection
with col4:
    if has_bch:
        bch_options = [ALL] + bchs
        sel_bchs = st.multiselect("Bayer", options=bch_options, default=[ALL])
    else:
        sel_bchs = []
eff_bchs = (bchs if has_bch and (ALL in sel_bchs or len(sel_bchs) == 0) else sel_bchs)

# 3) Global CAT depends on Countries + Bayer
with col2:
    df_for_cats = df[df["Country"].isin(eff_countries)]
    if has_bch and len(eff_bchs) > 0:
        df_for_cats = df_for_cats[df_for_cats["BCH"].isin(eff_bchs)]
    cats_filtered = sorted(
        df_for_cats["Global_CAT"].dropna().unique().tolist()
    )
    cat_options = [ALL] + cats_filtered
    sel_cats = st.multiselect(
        "Global CAT",
        options=cat_options,
        default=[ALL],
        key="sel_cats"
    )
eff_cats = cats_filtered if ALL in sel_cats or len(sel_cats) == 0 else sel_cats

# 4) Global Segment depends on Countries + Bayer + Global CAT
with col3:
    if has_seg:
        df_for_segments = df[df["Country"].isin(eff_countries)]
        if has_bch and len(eff_bchs) > 0:
            df_for_segments = df_for_segments[df_for_segments["BCH"].isin(eff_bchs)]
        if len(eff_cats) > 0:
            df_for_segments = df_for_segments[df_for_segments["Global_CAT"].isin(eff_cats)]
        segments_filtered = sorted(df_for_segments["Global_Segment"].dropna().unique().tolist())
        seg_options = [ALL] + segments_filtered
        sel_segments = st.multiselect("Global Segment", options=seg_options, default=[ALL])
    else:
        sel_segments = []
        segments_filtered = []
eff_segments = segments_filtered if has_seg and (ALL in sel_segments or len(sel_segments) == 0) else sel_segments

# Determine if the Product filter should be shown
# It should appear if 'Yes' (BCH) or 'Select All' is chosen in the Bayer filter.
# It should be hidden if only 'No' (Others) is selected.
show_product_filter = False
if has_prod:
    if not has_bch:
        show_product_filter = True  # No Bayer filter, so always show Product
    elif ALL in sel_bchs or any(str(x).strip().lower() in ["yes", "bch"] for x in sel_bchs):
        show_product_filter = True

# 5) Product (depends on all above)
with col5:
    if show_product_filter:
        df_for_prods = df[df["Country"].isin(eff_countries)]
        if len(eff_cats) > 0:
            df_for_prods = df_for_prods[df_for_prods["Global_CAT"].isin(eff_cats)]
        if has_seg and len(eff_segments) > 0:
            df_for_prods = df_for_prods[df_for_prods["Global_Segment"].isin(eff_segments)]
        if has_bch and len(eff_bchs) > 0:
            df_for_prods = df_for_prods[df_for_prods["BCH"].isin(eff_bchs)]
        
        products_filtered = sorted(df_for_prods["Product"].dropna().unique().tolist())
        prod_options = [ALL] + products_filtered
        sel_products = st.multiselect("Product", options=prod_options, default=[ALL])
    else:
        sel_products = []
        products_filtered = []

eff_products = products_filtered if has_prod and (ALL in sel_products or len(sel_products) == 0) else sel_products

st.write("Selected filters:", {
    "Countries": f"{len(eff_countries)} selected",
    "Global CAT": f"{len(eff_cats)} selected",
    "Global Segment": f"{len(eff_segments)} selected" if has_seg else "N/A",
    "Bayer": f"{len(eff_bchs)} selected" if has_bch else "N/A",
    "Product": f"{len(eff_products)} selected" if has_prod else "N/A",
})

target_options = ["Units", "Euro Value"]
sel_targets = st.multiselect("Target(s)", options=target_options, default=["Units"])

df_filt = filter_df(df, eff_countries, eff_cats, eff_bchs, eff_segments, eff_products)
if df_filt.empty:
    st.warning("No data for the selected filters.")
    st.stop()

series = build_series(df_filt)
st.write(f"Series length: {len(series)}, date range: {series.index.min().date()} → {series.index.max().date()}")

# Controls
use_tsfresh = st.checkbox("Enable feature extraction and selection")
use_tuning = st.checkbox("Enable hyperparameter tuning")
run = st.button("Run models")

# Initialize session state for storing results
if 'results_cache' not in st.session_state:
    st.session_state.results_cache = {}

# Create a unique key for current filters
filter_key = f"{eff_countries}_{eff_cats}_{eff_segments}_{eff_bchs}_{eff_products}_{sel_targets}_{use_tsfresh}_{use_tuning}"



if run or filter_key in st.session_state.results_cache:
    if run:
        # Clear old cache and compute new results
        st.session_state.results_cache = {}

        # Collect results for all targets
        all_results = {}

        for target in (sel_targets or ["Units"]):
            series = build_series(df_filt, target_col=target)

            enable = {
                "pmdarima": True,
                "prophet": True,
                "skforecast_xgb": True,
                "sktime_es": True,
                "darts_es": True,
                "pydlm": True,
                "tsfresh_xgb": bool(use_tsfresh),
            }
            results_df, best_model, test_compare, all_forecasts, model_name_mapping = evaluate_models(series, enable, target_name=target, tune=use_tuning)

            # Store results for this target
            all_results[target] = {
                'series': series,
                'results_df': results_df,
                'best_model': best_model,
                'test_compare': test_compare,
                'all_forecasts': all_forecasts,
                'model_name_mapping': model_name_mapping
            }

        # Cache the results
        st.session_state.results_cache[filter_key] = {
            'all_results': all_results,
            'targets': sel_targets
        }

    # Retrieve cached results
    cached = st.session_state.results_cache[filter_key]
    all_results = cached['all_results']

    # Only keep the export summary section
    fcst_outputs = {}
    for target in (cached['targets'] or ["Units"]):
        result = all_results[target]
        all_forecasts = result['all_forecasts']
        best_model = result['best_model']
        if best_model and best_model in all_forecasts:
            fcst_outputs[target] = all_forecasts[best_model]
        elif all_forecasts:
            # fallback to first model if best not found
            fcst_outputs[target] = next(iter(all_forecasts.values()))

    if fcst_outputs:
        # 1) Merge forecasts for all selected targets on Month
        fcst_months = None
        for tgt, df_out in fcst_outputs.items():
            col_name = f"Forecast_{tgt}"
            df_tmp = df_out[["Month", col_name]]
            fcst_months = df_tmp if fcst_months is None else fcst_months.merge(df_tmp, on="Month", how="outer")

        # 2) Build a single summary row per month with selected filters
        display_country = (eff_countries[0] if len(eff_countries) == 1 else "All")
        display_cat = ("All" if set(eff_cats) == set(cats_filtered) else ", ".join(eff_cats or ["All"]))
        display_segment = (
            eff_segments[0] if has_seg and len(eff_segments) == 1
            else "All" if has_seg and (set(eff_segments) == set(segments_filtered) or len(eff_segments) == 0)
            else ", ".join(eff_segments) if has_seg
            else "All"
        )
        display_bch = ("All" if has_bch and set(eff_bchs) == set(bchs) else (", ".join(eff_bchs) if has_bch and len(eff_bchs) > 0 else "All"))
        display_product = (
            eff_products[0] if has_prod and len(eff_products) == 1
            else "All" if has_prod and (set(eff_products) == set(products_filtered) or len(eff_products) == 0)
            else ", ".join(eff_products) if has_prod
            else "All"
        )

        export_df = fcst_months.copy()
        export_df["Country"] = display_country
        export_df["Global_CAT"] = display_cat
        export_df["Global_Segment"] = display_segment
        export_df["BCH"] = display_bch
        export_df["Product"] = display_product

        # 3) Order columns for readability
        forecast_cols = [c for c in export_df.columns if c.startswith("Forecast_")]
        export_df = export_df[["Country", "Global_CAT", "Global_Segment", "BCH", "Product", "Month"] + forecast_cols]

        # 4) Sort by Month and show
        export_df = export_df.sort_values(["Month"]).reset_index(drop=True)

        # Format Month as date only, and round Forecast columns to int and Indian number format
        if "Month" in export_df.columns:
            export_df["Month"] = pd.to_datetime(export_df["Month"]).dt.strftime("%Y-%m-%d")
        for col in forecast_cols:
            export_df[col] = export_df[col].round(0).astype(int)
            export_df[col] = export_df[col].apply(format_indian_number)

        st.subheader("Summary of filters with forecasts ready for export")
        st.dataframe(export_df, use_container_width=True, hide_index=True)
    else:
        pass
 
# -----------------------------