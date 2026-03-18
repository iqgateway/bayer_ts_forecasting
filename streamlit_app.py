import os
# Limit thread usage for numpy/pandas/scipy multi-threaded libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from timeseries_utils import (
    load_and_prepare, filter_df, build_series, evaluate_models
)
# Add itertools for combinations
import os
import itertools
import concurrent.futures
import datetime
import pickle
from file_cache_utils import load_cache, save_cache, get_cached_valid_combinations
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



col1, col2, col3, col4, col5 = st.columns(5)




# 1) Countries (single selection)
with col1:
    country_options = countries
    sel_country = st.selectbox(
        "Country",
        options=country_options,
        key="sel_country"
    )
eff_countries = [sel_country]



# 2) Bayer (BCH) selection (multi-select with Select All)
with col4:
    if has_bch:
        bch_options = ["Select All"] + bchs
        sel_bchs = st.multiselect("Bayer", options=bch_options, default=[], key="sel_bchs")
        if "Select All" in sel_bchs:
            sel_bchs = bchs
    else:
        sel_bchs = []
eff_bchs = sel_bchs



# 3) Global CAT depends on Countries + Bayer
with col2:
    df_for_cats = df[df["Country"].isin(eff_countries)]
    if has_bch and len(eff_bchs) > 0:
        df_for_cats = df_for_cats[df_for_cats["BCH"].isin(eff_bchs)]
    cats_filtered = sorted(
        df_for_cats["Global_CAT"].dropna().unique().tolist()
    )
    cat_options = ["Select All"] + cats_filtered
    sel_cats = st.multiselect(
        "Global CAT",
        options=cat_options,
        default=[],
        key="sel_cats"
    )
    if "Select All" in sel_cats:
        sel_cats = cats_filtered
eff_cats = sel_cats



# 4) Global Segment depends on Countries + Bayer + Global CAT
with col3:
    if has_seg:
        df_for_segments = df[df["Country"].isin(eff_countries)]
        if has_bch and len(eff_bchs) > 0:
            df_for_segments = df_for_segments[df_for_segments["BCH"].isin(eff_bchs)]
        if len(eff_cats) > 0:
            df_for_segments = df_for_segments[df_for_segments["Global_CAT"].isin(eff_cats)]
        segments_filtered = sorted(df_for_segments["Global_Segment"].dropna().unique().tolist())
        seg_options = ["Select All"] + segments_filtered
        sel_segments = st.multiselect(
            "Global Segment",
            options=seg_options,
            default=[],
            key="sel_segments"
        )
        if "Select All" in sel_segments:
            sel_segments = segments_filtered
    else:
        sel_segments = []
        segments_filtered = []
eff_segments = sel_segments


# Determine if the Product filter should be shown
# Only show if Bayer is BCH; if Bayer is Other, auto-select all products (not shown in UI)
show_product_filter = False
if has_prod:
    if not has_bch:
        show_product_filter = True  # No Bayer filter, so always show Product
    elif any(str(x).strip().lower() in ["yes", "bch"] for x in sel_bchs):
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
        prod_options = ["Select All"] + products_filtered
        sel_products = st.multiselect(
            "Product",
            options=prod_options,
            default=[],
            key="sel_products"
        )
        if "Select All" in sel_products:
            sel_products = products_filtered
        eff_products = sel_products
    else:
        # If Bayer is Other, auto-select all products for the filtered data, but do not show in UI
        if has_prod and has_bch and all(str(x).strip().lower() not in ["yes", "bch"] for x in sel_bchs):
            # Filter products based on all current filters (Country, Global CAT, Global Segment, Bayer)
            df_for_prods = df[df["Country"].isin(eff_countries)]
            if len(eff_cats) > 0:
                df_for_prods = df_for_prods[df_for_prods["Global_CAT"].isin(eff_cats)]
            if has_seg and len(eff_segments) > 0:
                df_for_prods = df_for_prods[df_for_prods["Global_Segment"].isin(eff_segments)]
            if has_bch and len(eff_bchs) > 0:
                df_for_prods = df_for_prods[df_for_prods["BCH"].isin(eff_bchs)]
            # Only use products present in the filtered data
            products_filtered = sorted(df_for_prods["Product"].dropna().unique().tolist())
            eff_products = products_filtered
        else:
            eff_products = []

st.write("Selected filters:", {
    "Countries": f"{len(eff_countries)} selected",
    "Global CAT": f"{len(eff_cats)} selected",
    "Global Segment": f"{len(eff_segments)} selected" if has_seg else "N/A",
    "Bayer": f"{len(eff_bchs)} selected" if has_bch else "N/A",
    "Product": f"{len(eff_products)} selected" if has_prod else "N/A",
})

target_options = ["Units", "Euro Value"]
sel_targets = st.multiselect("Target(s)", options=target_options, default=["Units"])


# --- Generate all combinations for modeling ---
combination_lists = [
    eff_countries or [None],
    eff_cats or [None],
    eff_segments or [None],
    eff_bchs or [None],
    eff_products or [None],
]

combinations = list(itertools.product(*combination_lists))



# --- Efficient valid combination generation ---
def get_valid_combinations(df, eff_countries, eff_cats, eff_segments, eff_bchs, eff_products):
    # Filter DataFrame by selected filters (skip empty lists)
    d = df.copy()
    if eff_countries:
        d = d[d["Country"].isin(eff_countries)]
    if eff_cats:
        d = d[d["Global_CAT"].isin(eff_cats)]
    if eff_segments:
        d = d[d["Global_Segment"].isin(eff_segments)]
    if eff_bchs:
        d = d[d["BCH"].isin(eff_bchs)]
    if eff_products:
        d = d[d["Product"].isin(eff_products)]
    # Get unique combinations of the relevant columns
    combo_cols = ["Country", "Global_CAT", "Global_Segment", "BCH", "Product"]
    valid_combos = d[combo_cols].drop_duplicates().values.tolist()
    return [tuple(row) for row in valid_combos]



# File-based cache for valid_combinations
COMBO_CACHE_FILE = os.path.join(os.path.dirname(__file__), "combo_cache.pkl")
if 'combination_cache' not in st.session_state:
    st.session_state.combination_cache = load_cache(COMBO_CACHE_FILE)

combo_cache_key = (
    tuple(eff_countries),
    tuple(eff_cats),
    tuple(eff_segments),
    tuple(eff_bchs),
    tuple(eff_products),
    tuple(sel_targets)
)
run_combinations = st.button("Run combinations")

valid_combinations = None
if run_combinations or combo_cache_key in st.session_state.combination_cache:
    if combo_cache_key in st.session_state.combination_cache:
        valid_combinations = st.session_state.combination_cache[combo_cache_key]
    else:
        combo_start_time = datetime.datetime.now()
        st.write(f"Combinations start time: {combo_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        progress_bar_combo = st.progress(0, text="Finding valid combinations...")
        valid_combinations = get_valid_combinations(df, eff_countries, eff_cats, eff_segments, eff_bchs, eff_products)
        progress_bar_combo.progress(100, text="Valid combinations found.")
        progress_bar_combo.empty()
        combo_end_time = datetime.datetime.now()
        st.write(f"Combinations end time: {combo_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        combo_total_time = (combo_end_time - combo_start_time).total_seconds()
        st.write(f"Total time taken for combinations: {combo_total_time:.2f} seconds")
        st.session_state.combination_cache[combo_cache_key] = valid_combinations
        save_cache(COMBO_CACHE_FILE, st.session_state.combination_cache)

if valid_combinations is not None:
    num_targets = len(sel_targets or ["Units"])
    enabled_model_keys = [k for k, v in {
        # "pmdarima": True,
        "skforecast_xgb": True,
        "sktime_es": True,
        "darts_es": True,
        "pydlm": True,
        "tsfresh_xgb": True
    }.items() if v]
    st.info(f"Models to run: Combinations found = {len(valid_combinations)}, Models to run: {len(valid_combinations) * len(enabled_model_keys) * num_targets}")

# Controls
use_tsfresh = True  # Always enabled
use_tuning = st.checkbox("Enable hyperparameter tuning")

run = st.button("Run models")

# Initialize session state for storing results
if 'results_cache' not in st.session_state:
    st.session_state.results_cache = {}
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = None
if 'summary_placeholder' not in st.session_state:
    st.session_state.summary_placeholder = None

# Create a unique key for current filters
filter_key = f"{eff_countries}_{eff_cats}_{eff_segments}_{eff_bchs}_{eff_products}_{sel_targets}_{use_tsfresh}_{use_tuning}"





if run or (filter_key in st.session_state.results_cache):
    
    if run:
        # Clear old cache and compute new results
        st.session_state.results_cache = {}

        enabled_model_keys = [k for k, v in {
            "skforecast_xgb": True,
            "sktime_es": True,
            "darts_es": True,
            "pydlm": True,
            "tsfresh_xgb": True
        }.items() if v]
        
        runtime_start = datetime.datetime.now()
        
        all_targets_results = []

        # --- Process target by target ---
        for target in (sel_targets or ["Units"]):
            st.markdown(f"### Processing Target: {target}")
            target_progress_bar = st.progress(0, text=f"Running combinations for {target}...")
            target_summary_placeholder = st.empty()
            
            target_export_df = pd.DataFrame()

            for i, combo in enumerate(valid_combinations):
                # For each combination, run all models for the current target
                combo_tasks = []
                for model_name in enabled_model_keys:
                    combo_tasks.append((combo, target, model_name, use_tuning))

                combo_results = {}
                
                def run_model_task(args):
                    combo, target, model_name, use_tuning = args
                    country, cat, segment, bch, product = combo
                    df_filt = filter_df(
                        df,
                        [country] if country else [], [cat] if cat else [],
                        [bch] if bch else [], [segment] if segment else [],
                        [product] if product else [],
                    )
                    if df_filt.empty: return (combo, target, model_name, None)
                    series = build_series(df_filt, target_col=target)
                    single_enable = {k: (k == model_name) for k in enabled_model_keys}
                    results_df, best_model, test_compare, all_forecasts, model_name_mapping = evaluate_models(
                        series, single_enable, target_name=target, tune=use_tuning
                    )
                    return (combo, target, model_name, {
                        'results_df': results_df, 'best_model': best_model,
                        'test_compare': test_compare, 'all_forecasts': all_forecasts,
                        'model_name_mapping': model_name_mapping
                    })

                with concurrent.futures.ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                    futures = [executor.submit(run_model_task, task) for task in combo_tasks]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            f_combo, f_target, f_model_name, f_result = future.result()
                            if f_result:
                                combo_key = (f_combo[0], f_combo[1], f_combo[2], f_combo[3], f_combo[4], f_target)
                                if combo_key not in combo_results:
                                    combo_results[combo_key] = {'all_forecasts': {}}
                                for model, forecast in f_result['all_forecasts'].items():
                                    combo_results[combo_key]['all_forecasts'][model] = forecast
                                if f_result.get('best_model'):
                                    combo_results[combo_key]['best_model'] = f_result['best_model']
                        except Exception as e:
                            st.warning(f"A model run for combination {combo} failed: {e}")
                            continue
                
                # Process results for the current combination
                combo_export_rows = []
                for combo_key, result in combo_results.items():
                    country, cat, segment, bch, product, res_target = combo_key
                    if not result: continue
                    
                    all_forecasts = result.get('all_forecasts', {})
                    best_model = result.get('best_model')

                    df_out = all_forecasts.get(best_model)
                    if df_out is None:
                        df_out = next(iter(all_forecasts.values()), None)
                    
                    if df_out is None: continue

                    col_name = f"Forecast_{res_target}"
                    for _, row in df_out.iterrows():
                        combo_export_rows.append({
                            "Country": country, "Global_CAT": cat, "Global_Segment": segment,
                            "BCH": bch, "Product": product, "Month": row["Month"],
                            col_name: row[col_name],
                        })
                
                if combo_export_rows:
                    new_rows_df = pd.DataFrame(combo_export_rows)
                    target_export_df = pd.concat([target_export_df, new_rows_df], ignore_index=True)

                    # Display the updated dataframe for the current target
                    display_df = target_export_df.copy()
                    display_df["Month"] = pd.to_datetime(display_df["Month"]).dt.strftime("%Y-%m-%d")
                    forecast_col = f"Forecast_{target}"
                    if forecast_col in display_df.columns:
                        display_df[forecast_col] = display_df[forecast_col].round(0).fillna(0).astype(int).apply(format_indian_number)
                    
                    with target_summary_placeholder.container():
                        st.subheader(f"Summary of filters with forecasts ready for export - {target}")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Update progress bar for the current target
                progress = min((i + 1) / len(valid_combinations), 1.0)
                target_progress_bar.progress(progress, text=f"Running combinations for {target}... ({i + 1}/{len(valid_combinations)})")

            all_targets_results.append(target_export_df)
            target_progress_bar.empty() # Clear progress bar for the completed target

        # --- Final Merging and Display ---
        if not all_targets_results:
            st.warning("No results were generated.")
        else:
            # Merge all target dataframes
            final_export_df = all_targets_results[0]
            for i in range(1, len(all_targets_results)):
                merge_cols = ["Country", "Global_CAT", "Global_Segment", "BCH", "Product", "Month"]
                final_export_df = pd.merge(final_export_df, all_targets_results[i], on=merge_cols, how="outer")

            runtime_end = datetime.datetime.now()
            total_time = runtime_end - runtime_start
            
            st.markdown(f"**Runtime start:** {runtime_start.strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Runtime end:** {runtime_end.strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Total time taken:** {str(total_time).split('.')[0]}")

            # Display final combined table at the bottom
            st.subheader("Final Combined Forecast Summary")
            display_df = final_export_df.copy()
            display_df["Month"] = pd.to_datetime(display_df["Month"]).dt.strftime("%Y-%m-%d")
            forecast_cols = [c for c in display_df.columns if c.startswith("Forecast_")]
            for col in forecast_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(0).fillna(0).astype(int).apply(format_indian_number)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Final results storage
            st.session_state.results_cache[filter_key] = {
                'export_df': final_export_df.to_dict('records'),
                'targets': sel_targets,
                'combinations': valid_combinations,
                'runtime_start': runtime_start,
                'runtime_end': runtime_end,
                'total_time': total_time,
            }


    # This part of the code is now mostly for reloading from cache
    cached = st.session_state.results_cache.get(filter_key)
    if cached and not run: # Only display from cache if not a new run
        runtime_start = cached.get('runtime_start')
        runtime_end = cached.get('runtime_end')
        total_time = cached.get('total_time')

        if runtime_start and runtime_end and total_time:
            st.markdown(f"**Runtime start:** {runtime_start.strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Runtime end:** {runtime_end.strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Total time taken:** {str(total_time).split('.')[0]}")

        export_df_records = cached.get('export_df')
        if export_df_records:
            export_df = pd.DataFrame(export_df_records)
            if "Month" in export_df.columns:
                export_df["Month"] = pd.to_datetime(export_df["Month"]).dt.strftime("%Y-%m-%d")
            forecast_cols = [c for c in export_df.columns if c.startswith("Forecast_")]
            for col in forecast_cols:
                export_df[col] = pd.to_numeric(export_df[col], errors='coerce').round(0).fillna(0).astype(int)
                export_df[col] = export_df[col].apply(format_indian_number)
            
            summary_placeholder = st.empty()
            with summary_placeholder.container():
                st.subheader("Summary of filters with forecasts ready for export")
                st.dataframe(export_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No data for the selected combinations.")
elif not run:
    st.info("Select filters and click 'Run models' to start.")
 
# -----------------------------