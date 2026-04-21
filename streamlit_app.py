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
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq
import shutil

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
import re
# from statsmodels.graphics.tsaplots import plot_acf

# Check if bayer_final.csv exists, otherwise DATA_PATHS will be empty (requires upload)
bayer_final_path = os.path.join(os.path.dirname(__file__), "bayer_final.csv")
DATA_PATHS = [bayer_final_path] if os.path.exists(bayer_final_path) else []
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
    /* Disabled button styling */
    div.stButton > button:disabled {
        background-color: #cccccc !important;
        color: #666666 !important;
        cursor: not-allowed !important;
        opacity: 0.6 !important;
    }
    div.stButton > button:disabled:hover {
        background-color: #cccccc !important;
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
    /* Hide uploaded file preview and cancel button, but keep instructions visible */
    div[data-testid="stFileUploader"] div[aria-live] ul {
        display: none !important;
    }
    div[data-testid="stFileUploader"] button[title="Remove"] {
        display: none !important;clear upload
    }

    /* Center the clear uploaded file button vertically with the file uploader and set width to auto */
    div[data-testid="stHorizontalBlock"] .stButton {
        margin-top: 12px !important;
    }
        div[data-testid="stFileUploaderFile"] {
        display: none !important;
    }
div[data-testid="stFileUploaderDropzoneInstructions"] div span:nth-child(2) {
    font-size: 10px !important;
}


    div[data-testid="stFileUploaderDropzoneInstructions"] div  span: nth-child(2) {
    data-testid="stFileUploaderFile
    div[data-testid="column"]:nth-of-type(2) button[kind="secondary"],
    div[data-testid="column"]:nth-of-type(2) button[data-testid="baseButton-secondary"] {
        width: auto !important;
        min-width: 160px;
        max-width: 100%;
        margin-top: -20px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    /* Add vertical space above the upload row using a pseudo-element on the flex container */
    div[data-testid="stHorizontalBlock"]::before {
        content: "";
        display: block;
        height: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def convert_wide_to_long(wide_df):
    """
    Convert wide-format DataFrame to long-format.
    Expects columns like 'Country', 'Global_CAT', etc., and multi-line headers like:
    'Jan 2021\nUnits', 'Jan 2021\nEuro Value', 'Feb 2021\nUnits', etc.
    """
    # Strip whitespace from column names
    wide_df = wide_df.rename(columns=lambda c: str(c).strip())
    
    # Predefined ID columns
    id_cols = [
        'Country',
        'Global CAT',
        'Global Sub-Cat',
        'Global Segment',
        'Global Brand BAM',
        'Brand',
        'Unified Corp',
        'Product',
        'BCH',
    ]
    
    # Headers like "Jan 2021\nUnits" or "Jan 2021\nEuro Value"
    value_cols = [c for c in wide_df.columns if '\n' in c]
    rx = re.compile(r'^(?P<Month>[A-Za-z]{3} \d{4})\n(?P<Measure>Units|Euro Value)$')
    parsed = [rx.match(c) for c in value_cols]
    
    if not all(parsed):
        raise ValueError("Unexpected header format. Expected 'Mon YYYY\\nUnits' or 'Mon YYYY\\nEuro Value'.")
    
    # Build MultiIndex (Month, Measure) for the wide columns
    multi_cols = pd.MultiIndex.from_tuples(
        [(m.group('Month'), m.group('Measure')) for m in parsed],
        names=['Month', 'Measure']
    )
    
    wide = wide_df.set_index(id_cols)[value_cols]
    wide.columns = multi_cols
    
    # Stack to convert to long format
    df_long_measure = (
        wide.stack(level=['Month', 'Measure'])
            .reset_index(name='Value')
    )
    
    # Convert to numeric
    df_long_measure['Value'] = pd.to_numeric(df_long_measure['Value'], errors='coerce')
    
    # Sort by Month (chronological ascending)
    df_long_measure['Month_dt'] = pd.to_datetime(df_long_measure['Month'], format='%b %Y', errors='coerce')
    df_long_measure = (
        df_long_measure
        .sort_values('Month_dt')
        .drop(columns='Month_dt')
        .reset_index(drop=True)
    )
    
    return df_long_measure


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

@st.cache_data
def load_data():
    if not DATA_PATHS:
        return None
    dfs = [load_and_prepare(path) for path in DATA_PATHS]
    return pd.concat(dfs, ignore_index=True)

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


# ==================== DATASET UPLOAD ====================

# Initialize session state for tracking processed files
if 'processed_file_id' not in st.session_state:
    st.session_state.processed_file_id = None
if 'uploader_key_counter' not in st.session_state:
    st.session_state.uploader_key_counter = 0

st.subheader("Upload File")
upload_col1, upload_col2 = st.columns([0.29, 1])
with upload_col1:
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        accept_multiple_files=False,
        key=f"dataset_uploader_{st.session_state.uploader_key_counter}",
        label_visibility="collapsed"
    )

with upload_col2:
    if st.button("Start Over",
                 key=f"start_over_{st.session_state.uploader_key_counter}",
                 disabled=uploaded_file is None):
        st.session_state.uploader_key_counter += 1
        st.session_state.processed_file_id = None
        # Delete all .pkl files in the workspace (including checkpoints and job state)
        import glob
        import os
        # Remove all .pkl files in the current directory
        for pkl_file in glob.glob(os.path.join(os.path.dirname(__file__), '*.pkl')):
            try:
                os.remove(pkl_file)
            except Exception:
                pass
        # Remove all .pkl files in the checkpoints directory
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        if os.path.exists(checkpoint_dir):
            for pkl_file in glob.glob(os.path.join(checkpoint_dir, '*.pkl')):
                try:
                    os.remove(pkl_file)
                except Exception:
                    pass
        st.rerun()

if uploaded_file is not None:
    # Create a unique identifier for this file
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    # Only process if this is a new file
    if st.session_state.processed_file_id != file_id:
        st.info("Converting dataset from wide to long format...")
        
        try:
            # Read the uploaded file
            wide_df = pd.read_csv(uploaded_file)
            
            # Convert to long format
            long_df = convert_wide_to_long(wide_df)
            
            # Save as bayer_final.csv
            output_path = os.path.join(os.path.dirname(__file__), "bayer_final.csv")
            long_df.to_csv(output_path, index=False)
            
            st.success(f"✅ Dataset converted and saved successfully! ({len(long_df)} rows)")
            
            # Mark this file as processed
            st.session_state.processed_file_id = file_id
            
            # Update DATA_PATHS
            DATA_PATHS.clear()
            DATA_PATHS.append(output_path)
            
            # Clear the cache to reload data
            load_data.clear()
            
            # Rerun to load the new data
            st.rerun()
            
        except Exception as e:
            st.error(f"Error converting dataset: {e}")
            st.session_state.processed_file_id = None
    else:
        # File already processed, show completion message
        st.success("Upload file complete. Please proceed with applying filter and running models.")
elif not DATA_PATHS:
    st.warning("No dataset found. Please upload a CSV file to continue.")
    st.stop()

st.markdown("---")

# ==================== END DATASET UPLOAD ====================


df = load_data()

if df is None:
    st.warning("⚠️ No dataset available. Please upload a CSV file above.")
    st.stop()


# ==================== CHECKPOINT/RESUME SETUP ====================

# Checkpoint and job state file paths
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
JOB_STATE_FILE = os.path.join(os.path.dirname(__file__), "job_state.pkl")

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_job_state(filter_config, valid_combinations):
    """Save the current job configuration for auto-resume"""
    try:
        state = {
            'filter_config': filter_config,
            'valid_combinations': valid_combinations,
            'timestamp': datetime.datetime.now().isoformat()
        }
        with open(JOB_STATE_FILE, 'wb') as f:
            pickle.dump(state, f)
    except Exception as e:
        st.warning(f"Failed to save job state: {e}")


def load_job_state():
    """Load saved job state if exists"""
    if os.path.exists(JOB_STATE_FILE):
        try:
            with open(JOB_STATE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def clear_job_state():
    """Clear saved job state"""
    if os.path.exists(JOB_STATE_FILE):
        try:
            os.remove(JOB_STATE_FILE)
        except Exception:
            pass


def get_checkpoint_file(target):
    """Get checkpoint file path for a target"""
    safe_target = target.replace(' ', '_').replace('/', '_')
    return os.path.join(CHECKPOINT_DIR, f"checkpoint_{safe_target}.pkl")


def load_checkpoint(target):
    """Load checkpoint - returns set of completed combination indices"""
    checkpoint_file = get_checkpoint_file(target)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            return data.get('completed_indices', set())
        except Exception as e:
            st.warning(f"Failed to load checkpoint for {target}: {e}")
            return set()
    return set()


def save_checkpoint(target, completed_indices, total):
    """Save checkpoint"""
    checkpoint_file = get_checkpoint_file(target)
    try:
        data = {
            'completed_indices': completed_indices,
            'total_combinations': total,
            'last_updated': datetime.datetime.now().isoformat()
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.warning(f"Failed to save checkpoint for {target}: {e}")


def clear_checkpoint(target):
    """Clear checkpoint file when target completes"""
    checkpoint_file = get_checkpoint_file(target)
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
        except Exception:
            pass


def has_active_checkpoint():
    """Check if there's an active checkpoint"""
    if not os.path.exists(CHECKPOINT_DIR):
        return False
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_')]
    return len(checkpoint_files) > 0


# Check for active checkpoint on app startup
if 'auto_resume_checked' not in st.session_state:
    st.session_state.auto_resume_checked = True
    
    if has_active_checkpoint():
        saved_state = load_job_state()
        
        if saved_state:
            st.session_state.auto_resume_state = saved_state
            st.session_state.should_auto_resume = True
            # Automatically trigger resume instead of waiting for button click
            st.session_state.trigger_auto_resume = True
        else:
            # Checkpoint exists but no job state - clear orphaned checkpoints
            shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    else:
        st.session_state.should_auto_resume = False


# ==================== END CHECKPOINT SETUP ====================


countries = sorted(df["Country"].dropna().unique().tolist())
cats_all = sorted(df["Global_CAT"].dropna().unique().tolist())
has_bch = "BCH" in df.columns
bchs = (sorted(df["BCH"].dropna().unique().tolist(), key=lambda x: 0 if str(x).strip().lower() == "yes" else 1) if has_bch else [])

has_seg = "Global_Segment" in df.columns
has_prod = "Product" in df.columns



col1, col2, col3, col4, col5 = st.columns(5)




# 1) Countries (single selection)
with col1:
    country_options = ["-- Select a Country --"] + countries
    if 'sel_country' not in st.session_state:
        st.session_state.sel_country = "-- Select a Country --"
    sel_country = st.selectbox(
        "Country",
        options=country_options,
        key="sel_country"
    )
eff_countries = [sel_country] if sel_country != "-- Select a Country --" else []


# 2) Global CAT depends on Countries
with col2:
    if 'sel_cats' not in st.session_state:
        st.session_state.sel_cats = []
    df_for_cats = df[df["Country"].isin(eff_countries)] if eff_countries else df
    cats_filtered = sorted(
        df_for_cats["Global_CAT"].dropna().unique().tolist()
    )
    cat_options = ["Select All"] + cats_filtered
    sel_cats = st.multiselect(
        "Global CAT",
        options=cat_options,
        key="sel_cats"
    )
    if "Select All" in sel_cats:
        sel_cats = cats_filtered
eff_cats = sel_cats


# 3) Global Segment depends on Countries + Global CAT
with col3:
    if has_seg:
        if 'sel_segments' not in st.session_state:
            st.session_state.sel_segments = []
        df_for_segments = df[df["Country"].isin(eff_countries)] if eff_countries else df
        if len(eff_cats) > 0:
            df_for_segments = df_for_segments[df_for_segments["Global_CAT"].isin(eff_cats)]
        segments_filtered = sorted(df_for_segments["Global_Segment"].dropna().unique().tolist())
        seg_options = ["Select All"] + segments_filtered
        sel_segments = st.multiselect(
            "Global Segment",
            options=seg_options,
            key="sel_segments"
        )
        if "Select All" in sel_segments:
            sel_segments = segments_filtered
    else:
        sel_segments = []
        segments_filtered = []
eff_segments = sel_segments


# 4) Bayer (BCH) selection (multi-select with Select All) - depends on previous filters
with col4:
    if has_bch:
        if 'sel_bchs' not in st.session_state:
            st.session_state.sel_bchs = []
        
        df_for_bchs = df.copy()
        if eff_countries:
            df_for_bchs = df_for_bchs[df_for_bchs["Country"].isin(eff_countries)]
        if eff_cats:
            df_for_bchs = df_for_bchs[df_for_bchs["Global_CAT"].isin(eff_cats)]
        if has_seg and eff_segments:
            df_for_bchs = df_for_bchs[df_for_bchs["Global_Segment"].isin(eff_segments)]

        bchs_filtered = sorted(df_for_bchs["BCH"].dropna().unique().tolist(), key=lambda x: 0 if str(x).strip().lower() == "yes" else 1)
        bch_options = ["Select All"] + bchs_filtered
        
        sel_bchs = st.multiselect("Bayer", options=bch_options, key="sel_bchs")
        
        if "Select All" in sel_bchs:
            sel_bchs = bchs_filtered
    else:
        sel_bchs = []
eff_bchs = sel_bchs


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
        if 'sel_products' not in st.session_state:
            st.session_state.sel_products = []
        df_for_prods = df[df["Country"].isin(eff_countries)] if eff_countries else df
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
            key="sel_products"
        )
        if "Select All" in sel_products:
            sel_products = products_filtered
        eff_products = sel_products
    else:
        # If Bayer is Other, auto-select all products for the filtered data, but do not show in UI
        if has_prod and has_bch and all(str(x).strip().lower() not in ["yes", "bch"] for x in sel_bchs):
            # Filter products based on all current filters (Country, Global CAT, Global Segment, Bayer)
            df_for_prods = df[df["Country"].isin(eff_countries)] if eff_countries else df
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




# Calculate product count for summary
if has_prod:
    # If all filters are empty (no selection), show 0
    if (
        not eff_countries and not eff_cats and not eff_segments and not eff_bchs and (
            (not show_product_filter) or (show_product_filter and not st.session_state.get('sel_products'))
        )
    ):
        product_count = 0
    elif not eff_products:
        product_count = 0
    else:
        product_count = len(eff_products)
else:
    product_count = "N/A"

st.write("Selected filters:", {
    "Countries": f"{len(eff_countries)} selected",
    "Global CAT": f"{len(eff_cats)} selected",
    "Global Segment": f"{len(eff_segments)} selected" if has_seg else "N/A",
    "Bayer": f"{len(eff_bchs)} selected" if has_bch else "N/A",
    "Product": f"{product_count} selected" if has_prod else "N/A",
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
    
    # Get unique combinations, handling BCH vs Other differently
    combo_cols = ["Country", "Global_CAT", "Global_Segment", "BCH", "Product"]
    valid_combos_raw = d[combo_cols].drop_duplicates().values.tolist()
    
    # Process combinations: aggregate products for non-BCH entries
    valid_combos = []
    seen_non_bch = set()  # Track non-BCH combinations (without product)
    
    for row in valid_combos_raw:
        country, cat, segment, bch, product = row
        
        # Check if BCH is "Yes" or "BCH" (case-insensitive)
        is_bch = str(bch).strip().lower() in ["yes", "bch"]
        
        if is_bch:
            # For BCH, keep individual products
            valid_combos.append(tuple(row))
        else:
            # For non-BCH (Other), aggregate all products into one combination
            # Use None as product to indicate aggregation
            combo_key = (country, cat, segment, bch)
            if combo_key not in seen_non_bch:
                seen_non_bch.add(combo_key)
                valid_combos.append((country, cat, segment, bch, None))
    
    return valid_combos



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
        progress_bar_combo = st.progress(0, text="Finding valid combinations...")
        valid_combinations = get_valid_combinations(df, eff_countries, eff_cats, eff_segments, eff_bchs, eff_products)
        progress_bar_combo.progress(100, text="Valid combinations found.")
        progress_bar_combo.empty()
        combo_end_time = datetime.datetime.now()
        combo_total_time = (combo_end_time - combo_start_time).total_seconds()
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

# Create buttons side by side with minimal gap
button_col1, button_col2 = st.columns([0.09, 1.2])
with button_col1:
    run = st.button("Run Model", width=120)
with button_col2:
    clear = st.button("Clear Selection", width=140)

# Handle clear selection
if clear:
    # Clear job state and checkpoints
    clear_job_state()
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Delete all filter session state keys to force complete refresh
    keys_to_delete = [
        'sel_country', 'sel_bchs', 'sel_cats', 'sel_segments', 'sel_products',
        'results_cache', 'auto_resume_state', 'should_auto_resume', 'trigger_auto_resume'
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# ==================== AUTO-RESUME NOTIFICATION ====================

# Check if there's a job to auto-resume - show informational message
if st.session_state.get('should_auto_resume', False) and st.session_state.get('auto_resume_state'):
    saved_state = st.session_state.auto_resume_state
    
    # st.info("🔄 **Automatically Resuming Incomplete Job from Previous Session**")
    
    config = saved_state['filter_config']
    
    col_info, col_actions = st.columns([3, 1])
    
    with col_info:
        for target in config['targets']:
            completed = load_checkpoint(target)
            if completed:
                progress_pct = (len(completed) / len(saved_state['valid_combinations'])) * 100
                # if len(completed) == len(saved_state['valid_combinations']):
                #     st.write(f"- ✅ '{target}': Complete")
                # else:
                #     st.write(f"- ⏸️ '{target}': {len(completed)}/{len(saved_state['valid_combinations'])} ({progress_pct:.1f}%)")
            else:
                temp_file = f"temp_results_{target}.parquet"
                if os.path.exists(temp_file):
                    st.write(f"- ✅ '{target}': Complete")
                else:
                    st.write(f"- 🆕 '{target}': Not started")
    
    # with col_actions:
    #     if st.button("🗑️ Cancel & Start Fresh", use_container_width=True):
    #         clear_job_state()
    #         import shutil
    #         shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    #         os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    #         # Clean up any temp parquet files
    #         import glob
    #         for temp_file in glob.glob("temp_results_*.parquet"):
    #             if os.path.exists(temp_file):
    #                 os.remove(temp_file)
    #         st.session_state.should_auto_resume = False
    #         st.session_state.trigger_auto_resume = False
    #         st.session_state.pop('auto_resume_state', None)
    #         st.success("✅ Cleared previous job. Select new filters below.")
    #         st.rerun()
    
    st.markdown("---")


# Initialize session state for storing results
if 'results_cache' not in st.session_state:
    st.session_state.results_cache = {}
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = None
if 'summary_placeholder' not in st.session_state:
    st.session_state.summary_placeholder = None

# Create a unique key for current filters
filter_key = f"{eff_countries}_{eff_cats}_{eff_segments}_{eff_bchs}_{eff_products}_{sel_targets}_{use_tsfresh}_{use_tuning}"


# ==================== MODEL TASK FUNCTION ====================

# Define run_model_task at module level with robust error handling
def run_model_task(args):
    try:
        combo, target, model_name, use_tuning = args
        country, cat, segment, bch, product = combo
        
        enabled_model_keys = [k for k, v in {
            "skforecast_xgb": True,
            "sktime_es": True,
            "darts_es": True,
            "pydlm": True,
            "tsfresh_xgb": True
        }.items() if v]
        
        # Handle product filtering: if product is None, don't filter by product (aggregate all products)
        df_filt = filter_df(
            df,
            [country] if country else [], 
            [cat] if cat else [],
            [bch] if bch else [], 
            [segment] if segment else [],
            [product] if product else [],  # Empty list means don't filter by product
        )
        if df_filt.empty: 
            return (combo, target, model_name, None)
        
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
    except MemoryError as e:
        return (combo, target, model_name, None)
    except Exception as e:
        # Return None on any error to prevent worker crash
        return (combo, target, model_name, None)


if run or (filter_key in st.session_state.results_cache) or st.session_state.get('trigger_auto_resume', False):
    
    # Check if this is an auto-resume
    is_auto_resume = st.session_state.get('trigger_auto_resume', False)
    
    if is_auto_resume:
        st.session_state.trigger_auto_resume = False  # Reset trigger
        
        # Load saved state
        saved_state = st.session_state.get('auto_resume_state')
        if saved_state:
            filter_config = saved_state['filter_config']
            valid_combinations = saved_state['valid_combinations']
            
            # Restore filter values
            eff_countries = filter_config.get('countries', eff_countries)
            eff_cats = filter_config.get('cats', eff_cats)
            eff_segments = filter_config.get('segments', eff_segments)
            eff_bchs = filter_config.get('bchs', eff_bchs)
            eff_products = filter_config.get('products', eff_products)
            sel_targets = filter_config.get('targets', sel_targets)
            use_tuning = filter_config.get('use_tuning', use_tuning)
            
            # st.info(f"🔄 Resuming job with {len(valid_combinations)} combinations...")
            run = True  # Trigger the run
        else:
            st.error("Failed to load saved state")
            is_auto_resume = False
            run = False
    
    if run:
        # Save job state for auto-resume capability
        if not is_auto_resume:
            # Use the valid_combinations that were already computed by "Run combinations" button
            # They are available in the outer scope from the combinations finding step
            if valid_combinations is None:
                st.error("❌ Please click 'Run combinations' first to find valid combinations!")
                st.stop()
            
            # # Clear all previous cache before starting a new run (safety measure)
            # st.info("🗑️ Clearing previous cache...")
            
            # Clear job state
            clear_job_state()
            
            # Clear checkpoints
            import shutil
            shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            
            # Clear temp parquet files
            import glob
            for temp_file in glob.glob("temp_results_*.parquet"):
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Clear combination cache
            st.session_state.combination_cache = {}
            if os.path.exists(COMBO_CACHE_FILE):
                os.remove(COMBO_CACHE_FILE)
            
            filter_config = {
                'countries': eff_countries,
                'cats': eff_cats,
                'segments': eff_segments,
                'bchs': eff_bchs,
                'products': eff_products,
                'targets': sel_targets,
                'use_tuning': use_tuning
            }
            
            save_job_state(filter_config, valid_combinations)
        
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
        completed_targets = set()  # Track which targets have completed
        total_targets = sel_targets or ["Units"]

        # --- Process target by target ---
        for target in total_targets:
            # Skip targets that have already been completed (no checkpoint exists but temp file does)
            if is_auto_resume:
                checkpoint_exists = os.path.exists(get_checkpoint_file(target))
                temp_parquet_file = f"temp_results_{target}.parquet"
                temp_file_exists = os.path.exists(temp_parquet_file)
                
                # Only skip if no checkpoint AND temp file exists (meaning it completed)
                if not checkpoint_exists and temp_file_exists:
                    # Target already completed - load existing results and display
                    completed_targets.add(target)  # Track that this target is done
                    st.markdown(f"### Processing Target: {target}")
                    st.success(f"✅ Target '{target}' already completed - loading existing results")
                    target_export_df = pq.read_table(temp_parquet_file).to_pandas()
                    
                    # Display the summary for the completed target
                    display_df = target_export_df.copy()
                    display_df["Month"] = pd.to_datetime(display_df["Month"]).dt.strftime("%Y-%m-%d")
                    forecast_col = f"Forecast_{target}"
                    if forecast_col in display_df.columns:
                        display_df[forecast_col] = display_df[forecast_col].round(0).fillna(0).astype(int).apply(format_indian_number)
                    
                    st.subheader(f"Summary of filters with forecasts ready for export - {target}")
                    st.dataframe(display_df, width='stretch', hide_index=True)
                    
                    all_targets_results.append(target_export_df)
                    continue
            
            st.markdown(f"### Processing Target: {target}")
            target_progress_bar = st.progress(0, text=f"Running combinations for {target}...")
            target_summary_placeholder = st.empty()
            
            # Define a temporary parquet file for the current target
            temp_parquet_file = f"temp_results_{target}.parquet"
            
            # Load checkpoint to resume from where we left off
            completed_indices = load_checkpoint(target)
            
            # Check if we're resuming
            if completed_indices:
                pass
                # st.info(f"📌 Resuming from checkpoint: {len(completed_indices)}/{len(valid_combinations)} combinations already completed for '{target}'")
            else:
                # Clean up old temp file only if starting fresh
                if os.path.exists(temp_parquet_file):
                    os.remove(temp_parquet_file)

            for i, combo in enumerate(valid_combinations):
                # Skip already-completed combinations
                if i in completed_indices:
                    progress = min((i + 1) / len(valid_combinations), 1.0)
                    target_progress_bar.progress(progress, text=f"Skipping completed combination {i + 1}/{len(valid_combinations)} for {target}")
                    continue
                
                # For each combination, run all models for the current target
                combo_tasks = []
                for model_name in enabled_model_keys:
                    combo_tasks.append((combo, target, model_name, use_tuning))

                combo_results = {}

                # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid memory issues
                # ThreadPoolExecutor shares memory, avoiding DataFrame pickling overhead
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(run_model_task, task) for task in combo_tasks]
                    for future in concurrent.futures.as_completed(futures, timeout=300):  # 5 min timeout per task
                        try:
                            f_combo, f_target, f_model_name, f_result = future.result(timeout=10)  # 10 sec to get result
                            if f_result:
                                combo_key = (f_combo[0], f_combo[1], f_combo[2], f_combo[3], f_combo[4], f_target)
                                if combo_key not in combo_results:
                                    combo_results[combo_key] = {'all_forecasts': {}}
                                for model, forecast in f_result['all_forecasts'].items():
                                    combo_results[combo_key]['all_forecasts'][model] = forecast
                                if f_result.get('best_model'):
                                    combo_results[combo_key]['best_model'] = f_result['best_model']
                        except concurrent.futures.TimeoutError:
                            st.warning(f"Model task timed out for combination {combo}")
                            continue
                        except MemoryError as e:
                            st.error(f"Out of memory for combination {combo}: {e}")
                            continue
                        except Exception as e:
                            st.warning(f"Model run failed for combination {combo}, model {f_model_name if 'f_model_name' in locals() else 'unknown'}: {str(e)[:100]}")
                            continue
                
                # Process results for the current combination and append to Parquet
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
                            "Country": country, 
                            "Global_CAT": cat, 
                            "Global_Segment": segment,
                            "BCH": bch, 
                            "Product": product if product else "All Products",  # Display "All Products" when aggregated
                            "Month": row["Month"],
                            col_name: row[col_name],
                        })
                
                if combo_export_rows:
                    new_data_df = pd.DataFrame(combo_export_rows)
                    
                    # Read existing data, concatenate, and write back
                    if os.path.exists(temp_parquet_file):
                        existing_df = pq.read_table(temp_parquet_file).to_pandas()
                        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
                    else:
                        combined_df = new_data_df

                    # Write the combined data back to the Parquet file
                    table = pa.Table.from_pandas(combined_df)
                    pq.write_table(table, temp_parquet_file)

                # Mark this combination as completed and save checkpoint
                completed_indices.add(i)
                save_checkpoint(target, completed_indices, len(valid_combinations))
                
                # Update progress bar for the current target
                progress = min((i + 1) / len(valid_combinations), 1.0)
                progress_pct = (len(completed_indices) / len(valid_combinations)) * 100
                target_progress_bar.progress(progress, text=f"Running combinations for {target}... ({len(completed_indices)}/{len(valid_combinations)}, {progress_pct:.1f}%)")

            # After all combinations for the target are done, read from Parquet and display
            target_export_df = pd.DataFrame()
            if os.path.exists(temp_parquet_file):
                target_export_df = pq.read_table(temp_parquet_file).to_pandas()
                
                display_df = target_export_df.copy()
                display_df["Month"] = pd.to_datetime(display_df["Month"]).dt.strftime("%Y-%m-%d")
                forecast_col = f"Forecast_{target}"
                if forecast_col in display_df.columns:
                    display_df[forecast_col] = display_df[forecast_col].round(0).fillna(0).astype(int).apply(format_indian_number)
                
                with target_summary_placeholder.container():
                    st.subheader(f"Summary of filters with forecasts ready for export - {target}")
                    st.dataframe(display_df, width='stretch', hide_index=True)
                
                # Keep temp file for resume status checking - only delete when all targets done
            
            all_targets_results.append(target_export_df)
            target_progress_bar.empty() # Clear progress bar for the completed target
            
            # Clear checkpoint since target is complete
            clear_checkpoint(target)
            completed_targets.add(target)  # Track completion
            st.success(f"✅ All {len(valid_combinations)} combinations completed for '{target}'")
            
            # Check if all targets from the original configuration are complete
            if len(completed_targets) == len(total_targets):
                # All targets done - clear job state and cleanup temp files
                clear_job_state()
                # Clean up all temp parquet files
                for t in total_targets:
                    temp_file = f"temp_results_{t}.parquet"
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                # st.success("🎉 All targets completed! Job state cleared.")

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
            
            # st.markdown(f"**Runtime start:** {runtime_start.strftime('%Y-%m-%d %H:%M:%S')}")
            # st.markdown(f"**Runtime end:** {runtime_end.strftime('%Y-%m-%d %H:%M:%S')}")
            # st.markdown(f"**Total time taken:** {str(total_time).split('.')[0]}")

            # Display final combined table at the bottom
            st.subheader("Final Combined Forecast Summary")
            display_df = final_export_df.copy()
            display_df["Month"] = pd.to_datetime(display_df["Month"]).dt.strftime("%Y-%m-%d")
            forecast_cols = [c for c in display_df.columns if c.startswith("Forecast_")]
            for col in forecast_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(0).fillna(0).astype(int).apply(format_indian_number)
            st.dataframe(display_df, width='stretch', hide_index=True)

            # Clear job state since all targets completed successfully
            clear_job_state()
            # Clean up any remaining temp parquet files
            for t in total_targets:
                temp_file = f"temp_results_{t}.parquet"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            st.session_state.should_auto_resume = False
            st.session_state.pop('auto_resume_state', None)
            
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
                st.dataframe(export_df, width='stretch', hide_index=True)
        else:
            st.warning("No data for the selected combinations.")
elif not run:
    st.info("Select filters and click 'Run models' to start.")

# # -----------------------------
# # File Upload Test Section
# # -----------------------------
# st.markdown("---")
# st.subheader("File Upload Test")
# uploaded_file = st.file_uploader("Upload a CSV file to test file size limit", type=["csv"])
# if uploaded_file is not None:
#     st.success(f"File uploaded successfully! File name: {uploaded_file.name}, Size: {uploaded_file.size / (1024*1024):.2f} MB")
 
# # -----------------------------