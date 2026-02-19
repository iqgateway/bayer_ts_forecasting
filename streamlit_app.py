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

# 2) Global CAT depends on Countries
with col2:
    cats_filtered = sorted(
        df[df["Country"].isin(eff_countries)]["Global_CAT"].dropna().unique().tolist()
    )
    cat_options = [ALL] + cats_filtered
    sel_cats = st.multiselect(
    "Global CAT",
    options=cat_options,
    default=[ALL],
    key="sel_cats"
)

eff_cats = cats_filtered if ALL in sel_cats or len(sel_cats) == 0 else sel_cats

# 3) Global Segment depends on Countries + Global CAT
with col3:
    if has_seg:
        df_for_segments = df[df["Country"].isin(eff_countries)]
        if len(eff_cats) > 0:
            df_for_segments = df_for_segments[df_for_segments["Global_CAT"].isin(eff_cats)]
        segments_filtered = sorted(df_for_segments["Global_Segment"].dropna().unique().tolist())
        seg_options = [ALL] + segments_filtered
        sel_segments = st.multiselect("Global Segment", options=seg_options, default=[ALL])
    else:
        sel_segments = []
        segments_filtered = []

eff_segments = segments_filtered if has_seg and (ALL in sel_segments or len(sel_segments) == 0) else sel_segments

# 4) Bayer (unchanged)
with col4:
    if has_bch:
        bch_options = [ALL] + bchs
        sel_bchs = st.multiselect("Bayer", options=bch_options, default=[ALL])
    else:
        sel_bchs = []

eff_bchs = (bchs if has_bch and (ALL in sel_bchs or len(sel_bchs) == 0) else sel_bchs)

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
            results_df, best_model, test_compare, all_forecasts = evaluate_models(series, enable, target_name=target, tune=use_tuning)
            
            # Store results for this target
            all_results[target] = {
                'series': series,
                'results_df': results_df,
                'best_model': best_model,
                'test_compare': test_compare,
                'all_forecasts': all_forecasts
            }
        
        # Cache the results
        st.session_state.results_cache[filter_key] = {
            'all_results': all_results,
            'targets': sel_targets
        }
    
    # Retrieve cached results
    cached = st.session_state.results_cache[filter_key]
    all_results = cached['all_results']
    
    # Now display results (this part won't re-compute on dropdown change)
    yearly_totals_by_target = {}
    fcst_outputs = {}
    best_model_per_target = {}
    
    for target in (cached['targets'] or ["Units"]):
        st.subheader(f"--------------- Report ({target}) -----------------")
        
        result = all_results[target]
        series = result['series']
        results_df = result['results_df']
        best_model = result['best_model']
        test_compare = result['test_compare']
        all_forecasts = result['all_forecasts']
        
        st.write(f"{target} series length: {len(series)}, date range: {series.index.min().date()} → {series.index.max().date()}")

        st.subheader(f"Test Data Analysis: Actual vs Predictions — {target}")
        styled_df = test_compare.style.set_properties(**{'text-align': 'center'}).set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}]
        )
        st.dataframe(
            styled_df,
            use_container_width=True, 
            hide_index=True
        )

        # Plot: test actual vs each model
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(test_compare["Month"], test_compare["Actual"], label="Actual", marker="o", linestyle=":", linewidth=2)
        for col in test_compare.columns:
            if col.startswith("Pred_"):
                ax.plot(test_compare["Month"], test_compare[col], label=col.replace("Pred_", "Pred: "), marker="o")
        ax.set_title(f"Test comparison ({target})")
        ax.set_xlabel("Month"); ax.set_ylabel(target); ax.grid(True); ax.legend(fontsize=8)
        st.pyplot(fig)

        st.subheader("Model metrics")
        st.dataframe(results_df, use_container_width=True, hide_index = True)
        st.info(f"Best model: {best_model or 'None'} — based on the current filter selection for {target}.")

        if best_model:
            pred_col = f"Pred_{best_model}"
            if pred_col in test_compare.columns:
                st.subheader(f"Best Model Visualization: Actual vs Predicted — {best_model} ({target})")
                fig_best, ax_best = plt.subplots(figsize=(10, 4))
                ax_best.plot(test_compare["Month"], test_compare["Actual"], label="Actual", marker="o", linestyle=":", linewidth=2)
                ax_best.plot(test_compare["Month"], test_compare[pred_col], label=f"Predicted ({best_model})", marker="o", linestyle="-", linewidth=2)
                ax_best.set_xlabel("Month"); ax_best.set_ylabel(target); ax_best.set_title(f"Actual vs Predicted for best model ({target})")
                ax_best.grid(True); ax_best.legend(fontsize=8)
                st.pyplot(fig_best)
            else:
                st.warning(f"Predictions for best model '{best_model}' not found ({target}).")

        # Model selection dropdown for forecast - All forecasts are already computed
        available_models = list(all_forecasts.keys())
        if available_models:
            st.markdown(f"<h3 class='green-header'>Select model for forecast ({target}). Current Selection is best model: {best_model}</h3>", unsafe_allow_html=True)
            default_index = available_models.index(best_model) if best_model in available_models else 0
            selected_model = st.selectbox(
                "Model selection",
                options=available_models,
                index=default_index,
                key=f"model_select_{target}",
                label_visibility="collapsed"
            )
            fcst_df = all_forecasts[selected_model]
        else:
            selected_model = None
            fcst_df = pd.DataFrame(columns=["Month", f"Forecast_{target}"])
        
        st.subheader(f"Forecast: 2026 — {target} ({selected_model})")
        fc_col = f"Forecast_{target}"
        if not fcst_df.empty and fc_col in fcst_df.columns:
            display_df = fcst_df.copy()
            display_df["Month"] = display_df["Month"].dt.strftime("%Y-%m-%d %H:%M:%S")
            total_val = float(display_df[fc_col].sum())
            total_row = pd.DataFrame({"Month": ["Total"], fc_col: [total_val]})
            display_df = pd.concat([display_df, total_row], ignore_index=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(fcst_df, use_container_width=True, hide_index=True)

        # Plot forecast vs history
        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.plot(series.index, series.values, label="History", marker="o")
        if not fcst_df.empty and fc_col in fcst_df.columns:
            ax2.plot(fcst_df["Month"], fcst_df[fc_col], label=f"Forecast ({selected_model or 'N/A'})", marker="o")
        ax2.set_title(f"2026 Forecast ({target})")
        ax2.set_xlabel("Month"); ax2.set_ylabel(target); ax2.grid(True); ax2.legend()
        st.pyplot(fig2)

        # Yearly analysis (history totals + forecasted year)
        st.subheader(f"Yearly Totals — {target}")
        hist_totals = series.groupby(series.index.year).sum()
        if not fcst_df.empty and fc_col in fcst_df.columns:
            forecast_year = int(fcst_df["Month"].dt.year.iloc[0])
            hist_totals.loc[forecast_year] = float(fcst_df[fc_col].sum())
        yearly_totals_by_target[target] = hist_totals.sort_index()

        yearly_df = pd.DataFrame({
        "Year": yearly_totals_by_target[target].index.astype(str),
        f"Total_{target}": yearly_totals_by_target[target].values
        })

        # Bar chart per target
        fig_y, ax_y = plt.subplots(figsize=(8,4))
        rects = ax_y.bar(
            yearly_df["Year"].astype(str),
            yearly_df[f"Total_{target}"],
            color=PALETTE[0]
        )
        ax_y.set_title(f"Yearly totals ({target})")
        ax_y.set_xlabel("Year")
        ax_y.set_ylabel(target)
        ax_y.grid(axis="y", alpha=0.3)
        # Add value labels on bars
        for r in rects:
            h = r.get_height()
            ax_y.annotate(f"{h:.0f}", (r.get_x() + r.get_width() / 2, h),
                          ha="center", va="bottom", fontsize=8)
        st.pyplot(fig_y)

        # Keep per-target forecast for export (use selected model)
        fcst_outputs[target] = fcst_df.copy()
        best_model_per_target[target] = selected_model


    # Combined yearly analysis when multiple targets are selected
    if len(sel_targets or []) > 1:
        st.subheader("Yearly Totals — Combined")
        all_years = sorted(set().union(*[s.index.tolist() for s in yearly_totals_by_target.values()]))
        combined_df = pd.DataFrame({"Year": [str(y) for y in all_years]})
        for target in sel_targets:
            s = yearly_totals_by_target.get(target, pd.Series(dtype=float))
            combined_df[f"Total_{target}"] = [float(s.get(y, 0.0)) for y in all_years]

        # Combined table
        #st.dataframe(combined_df, use_container_width=True, hide_index = True)

        # Grouped bar chart
        x = np.arange(len(all_years))
        width = min(0.8 / max(1, len(sel_targets)), 0.35)
        fig_c, ax_c = plt.subplots(figsize=(10,4))
        offsets = np.linspace(-width*(len(sel_targets)-1)/2, width*(len(sel_targets)-1)/2, len(sel_targets))
        for idx, target in enumerate(sel_targets):
            #ax_c.bar(x + offsets[idx], combined_df[f"Total_{target}"], width, label=target)
            totals = combined_df[f"Total_{target}"]
            bars = ax_c.bar(
                x + offsets[idx],
                totals,
                width,
                label=target,
                color=PALETTE[idx % len(PALETTE)]
            )
            # Add value labels
            for r in bars:
                h = r.get_height()
                ax_c.annotate(f"{h:.0f}", (r.get_x() + r.get_width()/2, h),
                              ha="center", va="bottom", fontsize=8)
                
        ax_c.set_xticks(x)
        ax_c.set_xticklabels([str(y) for y in all_years])
        ax_c.set_title("Yearly totals (combined)")
        ax_c.set_xlabel("Year")
        ax_c.set_ylabel("Totals")
        ax_c.grid(axis="y", alpha=0.3)
        ax_c.legend()
        st.pyplot(fig_c)

    # Build export report and download button (entity-level rows)
    if fcst_outputs:
        # 1) Merge forecasts for all selected targets on Month
        fcst_months = None
        for tgt, df_out in fcst_outputs.items():
            col_name = f"Forecast_{tgt}"
            df_tmp = df_out[["Month", col_name]]
            fcst_months = df_tmp if fcst_months is None else fcst_months.merge(df_tmp, on="Month", how="outer")

        # 2) Build a single summary row per month with selected filters
        # Display labels: show one selected value or "All" when selection is effectively all
        display_country = (eff_countries[0] if len(eff_countries) == 1 else "All")
        display_cat = ("All" if set(eff_cats) == set(cats_filtered) else ", ".join(eff_cats or ["All"]))
        # display_segment = ("All" if has_seg and set(eff_segments) == set(segments_filtered) else (", ".join(eff_segments) if has_seg and len(eff_segments) > 0 else "All"))
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

        st.subheader("Summary of filters with forecasts ready for export")
        st.dataframe(export_df, use_container_width=True, hide_index=True)

    else:
        pass
 
# -----------------------------