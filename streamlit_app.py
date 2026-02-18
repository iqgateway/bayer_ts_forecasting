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

DATA_PATH = os.path.join(os.path.dirname(__file__), "Bayer_2024_long.csv")

st.set_page_config(page_title="Time Series Forecasting", layout="wide")

# Use a modern, readable style and deterministic color palette
plt.style.use('seaborn-v0_8')
PALETTE = plt.cm.tab10.colors  # 10 distinct, accessible colors

@st.cache_data
def load_data():
    return load_and_prepare(DATA_PATH)

df = load_data()
st.title("Bayer Time Series Forecasting")


countries = sorted(df["Country"].dropna().unique().tolist())
cats_all = sorted(df["Global_CAT"].dropna().unique().tolist())
has_bch = "BCH" in df.columns
bchs = (sorted(df["BCH"].dropna().unique().tolist(), key=lambda x: 0 if str(x).strip().lower() == "yes" else 1) if has_bch else [])

has_seg = "Global_Segment" in df.columns

ALL = "Select All"

col1, col2, col3, col4 = st.columns(4)

# 1) Countries
with col1:
    country_options = [ALL] + countries
    sel_countries = st.multiselect("Countries", options=country_options, default=[ALL])

eff_countries = countries if ALL in sel_countries or len(sel_countries) == 0 else sel_countries

# 2) Global CAT depends on Countries
with col2:
    cats_filtered = sorted(
        df[df["Country"].isin(eff_countries)]["Global_CAT"].dropna().unique().tolist()
    )
    cat_options = [ALL] + cats_filtered
    sel_cats = st.multiselect("Global CAT", options=cat_options, default=[ALL])

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

st.write("Selected filters:", {
    "Countries": f"{len(eff_countries)} selected",
    "Global CAT": f"{len(eff_cats)} selected",
    "Global Segment": f"{len(eff_segments)} selected" if has_seg else "N/A",
    "Bayer": f"{len(eff_bchs)} selected" if has_bch else "N/A",
})

target_options = ["Units", "Euro Value"]
sel_targets = st.multiselect("Target(s)", options=target_options, default=["Units"])

df_filt = filter_df(df, eff_countries, eff_cats, eff_bchs, eff_segments)
if df_filt.empty:
    st.warning("No data for the selected filters.")
    st.stop()

series = build_series(df_filt)
st.write(f"Series length: {len(series)}, date range: {series.index.min().date()} → {series.index.max().date()}")

# Controls
use_tsfresh = st.checkbox("Enable feature extraction and selection")
use_tuning = st.checkbox("Enable hyperparameter tuning")
run = st.button("Run models")



if run:
    # Collect yearly totals for an optional combined view
    yearly_totals_by_target = {}

    # Collect per-target forecasts and best model for export
    fcst_outputs = {}
    best_model_per_target = {}

    for target in (sel_targets or ["Units"]):
        st.subheader(f"--------------- Report ({target}) -----------------")

        series = build_series(df_filt, target_col=target)
        st.write(f"{target} series length: {len(series)}, date range: {series.index.min().date()} → {series.index.max().date()}")

        ##################################################
        ##################################################
        ##################################################
        ##################################################
        ##################################################
        # # ACF for training set
        # from timeseries_utils import split_train_test  # local import to keep top imports minimal
        # y_train, y_test = split_train_test(series)
        # st.subheader(f"Autocorrelation (ACF) — {target} (train)")
        # if len(y_train) > 2:
        #     fig, ax = plt.subplots(figsize=(8, 3))
        #     plot_acf(y_train, ax=ax, lags=min(36, len(y_train) - 1))
        #     ax.set_title(f"ACF ({target}) - Train")
        #     st.pyplot(fig)
        # else:
        #     st.info("Not enough training data to compute ACF.")
        ##################################################
        ##################################################
        ##################################################
        ##################################################
        

        enable = {
            "pmdarima": True,
            "prophet": True,
            "skforecast_xgb": True,
            "sktime_es": True,
            "darts_es": True,
            "pydlm": True,
            "tsfresh_xgb": bool(use_tsfresh),
        }
        results_df, best_model, test_compare, fcst_df = evaluate_models(series, enable, target_name=target, tune=use_tuning)  #####################

        st.subheader(f"Test Data Analysis: Actual vs Predictions — {target}")
        st.dataframe(test_compare, use_container_width=True, hide_index = True)

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

        st.subheader(f"Forecast: 2025 — {target}")
        #st.dataframe(fcst_df, use_container_width=True, hide_index = True)
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
        #fc_col = f"Forecast_{target}"
        if not fcst_df.empty and fc_col in fcst_df.columns:
            ax2.plot(fcst_df["Month"], fcst_df[fc_col], label=f"Forecast ({best_model or 'N/A'})", marker="o")
        ax2.set_title(f"2025 Forecast ({target})")
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

        # Table per target
        #st.dataframe(yearly_df, use_container_width=True, hide_index = True)

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

        # Keep per-target forecast for export
        fcst_outputs[target] = fcst_df.copy()
        best_model_per_target[target] = best_model


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
    
    # # Build export report and download button (entity-level rows)
    # if fcst_outputs:
    #     # 1) Merge forecasts for all selected targets on Month
    #     fcst_months = None
    #     for tgt, df_out in fcst_outputs.items():
    #         col_name = f"Forecast_{tgt}"
    #         df_tmp = df_out[["Month", col_name]]
    #         fcst_months = df_tmp if fcst_months is None else fcst_months.merge(df_tmp, on="Month", how="outer")

    #     # 2) Build entity grid from filtered data (unique combos)
    #     entity_cols = ["Country", "Global_CAT", "Global_Segment", "BCH"]
    #     entities = df_filt.copy()

    #     # Ensure all required columns exist and are filled
    #     for c in entity_cols:
    #         if c not in entities.columns:
    #             entities[c] = "N/A"

    #     entities = entities[entity_cols].drop_duplicates()

    #     # 3) Cross-join entities with forecast months to get one row per entity per month
    #     export_df = (
    #         entities.assign(_key=1)
    #         .merge(fcst_months.assign(_key=1), on="_key", how="outer")
    #         .drop(columns="_key")
    #     )

    #     # 4) Order columns for readability (exclude best-model columns)
    #     forecast_cols = [c for c in export_df.columns if c.startswith("Forecast_")]
    #     export_df = export_df[["Country", "Global_CAT", "Global_Segment", "BCH", "Month"] + forecast_cols]

    #     # NEW: Show final tables
    #     st.subheader("Summary of filters with forecasts ready for export")
    #     st.dataframe(
    #         export_df.sort_values(["Country", "Global_CAT", "Global_Segment", "BCH", "Month"]),
    #         use_container_width=True,
    #         hide_index=True
    #     )
    # else:
    #     pass
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

        export_df = fcst_months.copy()
        export_df["Country"] = display_country
        export_df["Global_CAT"] = display_cat
        export_df["Global_Segment"] = display_segment
        export_df["BCH"] = display_bch

        # 3) Order columns for readability
        forecast_cols = [c for c in export_df.columns if c.startswith("Forecast_")]
        export_df = export_df[["Country", "Global_CAT", "Global_Segment", "BCH", "Month"] + forecast_cols]

        # 4) Sort by Month and show
        export_df = export_df.sort_values(["Month"]).reset_index(drop=True)

        st.subheader("Summary of filters with forecasts ready for export")
        st.dataframe(export_df, use_container_width=True, hide_index=True)

    else:
        pass
 
# -----------------------------