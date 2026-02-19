import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Optional

# -----------------------------
# Data loading/cleaning (same as notebook)
# -----------------------------
def load_and_prepare(data_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(data_path)
    df = df_raw.copy()

    # Aliases for convenience (mirrors notebook cell)
    col_map = {
        "Global-CAT": "Global_CAT",
        "Global CAT": "Global_CAT",
        "Global-Sub-Cat": "Global_Sub_Cat",
        "Global Sub Cat": "Global_Sub_Cat",
        "Global-Segment": "Global_Segment",
        "Global Segment": "Global_Segment",
        "LC-RSP-Price": "LC_RSP_Price",
        "Euro-Value": "Euro_Value",
    }
    for k, v in col_map.items():
        if k in df.columns:
            df[v] = df[k]

    # Parse Month
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df = df[(df["Month"] >= "2021-01-01") & (df["Month"] <= "2025-12-31")].copy()

    # Coerce numerics for new schema
    if "Value" in df.columns:
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    if "LC_RSP_Price" in df.columns:
        df["LC_RSP_Price"] = pd.to_numeric(df.get("LC_RSP_Price", np.nan), errors="coerce")

    return df

def filter_df(df: pd.DataFrame,
              countries: Optional[list[str]],
              global_cats: Optional[list[str]],
              bchs: Optional[list[str]],
              global_segments: Optional[list[str]],
              products: Optional[list[str]]) -> pd.DataFrame: 
    d = df.copy()
    if countries and len(countries) > 0:
        d = d[d["Country"].isin(countries)]
    if global_cats and len(global_cats) > 0:
        d = d[d["Global_CAT"].isin(global_cats)]
    if "Global_Segment" in d.columns and global_segments is not None and len(global_segments) > 0:
        d = d[d["Global_Segment"].isin(global_segments)]
    if "BCH" in d.columns and bchs is not None and len(bchs) > 0:
        d = d[d["BCH"].isin(bchs)]
    if "Product" in d.columns and products is not None and len(products) > 0:
        d = d[d["Product"].isin(products)]
    return d


def build_series(df_filt: pd.DataFrame, target_col: str = "Units") -> pd.Series:
    # Support new schema: Measure + Value
    d = df_filt.copy()
    value_col = None

    if "Measure" in d.columns and "Value" in d.columns:
        tgt = target_col.replace("_", " ").strip()  # allow "Euro_Value" -> "Euro Value"
        d["Measure_norm"] = d["Measure"].astype(str).str.strip().str.casefold()
        d = d[d["Measure_norm"] == tgt.casefold()].copy()
        value_col = "Value"
    elif target_col in d.columns:
        value_col = target_col
    else:
        raise KeyError(f"Target '{target_col}' not found (no Measure/Value or target column).")

    base_keys = [c for c in ["Country","Global_CAT","Global_Sub_Cat","Global_Segment","Corp","BCH","Product","Pack","Month"] if c in d.columns]
    df_dedup = (
        d.groupby(base_keys, as_index=False, dropna=False)
         .agg({value_col: "sum"})
    )
    agg = df_dedup.groupby(["Month"], as_index=False)[value_col].sum().sort_values("Month")
    s = pd.Series(agg[value_col].values, index=agg["Month"]).sort_index()
    s.index.freq = "MS"
    return s

# -----------------------------
# Metrics and plotting helpers
# -----------------------------
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)
    return 100 * np.mean(diff)

def metrics_table(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1, y_true))) * 100
    s_mape = smape(y_true, y_pred)
    return {"model": model_name, "RMSE": rmse, "MAE": mae, "MAPE%": mape, "sMAPE%": s_mape}

def split_train_test(s: pd.Series):
    y_train = s[s.index <= pd.Timestamp("2024-12-01")]
    y_test = s[(s.index >= pd.Timestamp("2025-01-01")) & (s.index <= pd.Timestamp("2025-12-01"))]
    return y_train, y_test

# -----------------------------
# Model runners (reuse notebook logic)
# -----------------------------
def run_pmdarima(y_train, y_test):
    import pmdarima as pm
    model = pm.auto_arima(
        y_train, seasonal=True, m=12, stepwise=True, suppress_warnings=True,
        error_action="ignore", maxiter=50
    )
    preds = model.predict(n_periods=len(y_test))
    fcst = model.predict(n_periods=12)
    return preds, fcst, model

def run_prophet(y_train, y_test):
    from prophet import Prophet
    train_df = pd.DataFrame({"ds": y_train.index, "y": y_train.values})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(train_df)

    # Predict test period (12 months)
    future_test = model.make_future_dataframe(periods=len(y_test), freq="MS")
    forecast_test = model.predict(future_test)
    preds = forecast_test.tail(len(y_test))["yhat"].values

    # Predict next 12 months after test (i.e., 2025)
    future_all = model.make_future_dataframe(periods=len(y_test) + 12, freq="MS")
    forecast_all = model.predict(future_all)
    fcst = forecast_all.tail(12)["yhat"].values
    return preds, fcst, model

# ---------- NEW: tuned Prophet ----------
def run_prophet_tuned(y_train, y_test):
    from prophet import Prophet
    train_df = pd.DataFrame({"ds": y_train.index, "y": y_train.values})
    grids = [
        {"seasonality_mode": "additive", "changepoint_prior_scale": 0.1},
        {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.05},
        {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.5},
    ]
    best = None; best_preds = None; best_fcst = None; best_model = None
    for g in grids:
        try:
            model = Prophet(
                yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                seasonality_mode=g["seasonality_mode"], changepoint_prior_scale=g["changepoint_prior_scale"]
            )
            model.fit(train_df)
            future_test = model.make_future_dataframe(periods=len(y_test), freq="MS")
            forecast_test = model.predict(future_test)
            preds = forecast_test.tail(len(y_test))["yhat"].values
            metric = smape(y_test.values, preds)
            if (best is None) or (metric < best):
                best = metric
                best_preds = preds
                future_all = model.make_future_dataframe(periods=len(y_test) + 12, freq="MS")
                forecast_all = model.predict(future_all)
                best_fcst = forecast_all.tail(12)["yhat"].values
                best_model = model
        except Exception:
            continue
    return best_preds, best_fcst, best_model

def run_skforecast_xgb(y_train, y_test):
    try:
        from skforecast.ForecasterAutoreg import ForecasterAutoreg
    except Exception:
        from skforecast.recursive import ForecasterRecursive as ForecasterAutoreg
    from xgboost import XGBRegressor
    forecaster = ForecasterAutoreg(
        regressor=XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, random_state=42
        ),
        lags=12
    )
    forecaster.fit(y=y_train)
    preds = np.array(forecaster.predict(steps=len(y_test)))
    fcst = np.array(forecaster.predict(steps=12))
    return preds, fcst, forecaster

# ---------- NEW: tuned skforecast XGB ----------
def run_skforecast_xgb_tuned(y_train, y_test):
    try:
        from skforecast.ForecasterAutoreg import ForecasterAutoreg
    except Exception:
        from skforecast.recursive import ForecasterRecursive as ForecasterAutoreg
    from xgboost import XGBRegressor

    grids = [
        {"lags": 12, "params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05}},
        {"lags": 12, "params": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03}},
        {"lags": 6,  "params": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1}},
    ]
    best = None
    best_preds, best_fcst, best_model = None, None, None
    for g in grids:
        forecaster = ForecasterAutoreg(
            regressor=XGBRegressor(
                n_estimators=g["params"]["n_estimators"],
                max_depth=g["params"]["max_depth"],
                learning_rate=g["params"]["learning_rate"],
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=0
            ),
            lags=g["lags"]
        )
        forecaster.fit(y=y_train)
        preds = np.array(forecaster.predict(steps=len(y_test)))
        metric = smape(y_test.values, preds)
        if (best is None) or (metric < best):
            best = metric
            best_preds = preds
            best_fcst = np.array(forecaster.predict(steps=12))
            best_model = forecaster
    return best_preds, best_fcst, best_model

def run_sktime_es(y_train, y_test):
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from sktime.forecasting.base import ForecastingHorizon
    y_tr = y_train.copy()
    y_tr.index = pd.PeriodIndex(y_tr.index, freq="M")
    y_ts_idx = pd.PeriodIndex(y_test.index, freq="M")
    fh = ForecastingHorizon(y_ts_idx, is_relative=False)

    model = ExponentialSmoothing(trend="add", seasonal="add", sp=12)
    model.fit(y_tr)
    preds = model.predict(fh).values

    future_fh = ForecastingHorizon(pd.period_range(y_tr.index[-1] + 1, periods=12, freq="M"), is_relative=False)
    fcst = model.predict(future_fh).values
    return preds, fcst, model

# ---------- NEW: tuned sktime ES ----------
def run_sktime_es_tuned(y_train, y_test):
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from sktime.forecasting.base import ForecastingHorizon
    y_tr = y_train.copy(); y_tr.index = pd.PeriodIndex(y_tr.index, freq="M")
    y_ts_idx = pd.PeriodIndex(y_test.index, freq="M")
    fh = ForecastingHorizon(y_ts_idx, is_relative=False)

    grids = [
        {"trend": "add", "seasonal": "add", "sp": 12},
        {"trend": "add", "seasonal": "mul", "sp": 12},
        {"trend": None,  "seasonal": "add", "sp": 12},
    ]
    best = None; best_preds = None; best_fcst = None; best_model = None
    for g in grids:
        try:
            model = ExponentialSmoothing(trend=g["trend"], seasonal=g["seasonal"], sp=g["sp"])
            model.fit(y_tr)
            preds = model.predict(fh).values
            metric = smape(y_test.values, preds)
            if (best is None) or (metric < best):
                best = metric
                best_preds = preds
                future_fh = ForecastingHorizon(pd.period_range(y_tr.index[-1] + 1, periods=12, freq="M"), is_relative=False)
                best_fcst = model.predict(future_fh).values
                best_model = model
        except Exception:
            continue
    return best_preds, best_fcst, best_model

def run_darts_es(y_train, y_test):
    from darts import TimeSeries
    from darts.models import ExponentialSmoothing as DartsES
    series = TimeSeries.from_times_and_values(y_train.index, y_train.values, freq="MS")
    model = DartsES()
    model.fit(series)
    preds_series = model.predict(len(y_test))
    preds = preds_series.values().flatten()
    fcst_series = model.predict(12)
    fcst = fcst_series.values().flatten()
    return preds, fcst, model

# ---------- NEW: tuned Darts ES ----------
def run_darts_es_tuned(y_train, y_test):
    from darts import TimeSeries
    from darts.models import ExponentialSmoothing as DartsES
    series = TimeSeries.from_times_and_values(y_train.index, y_train.values, freq="MS")
    grids = [
        {"trend": None, "seasonal": "add", "seasonal_periods": 12},
        {"trend": "add", "seasonal": "add", "seasonal_periods": 12},
        {"trend": "add", "seasonal": "mul", "seasonal_periods": 12},
    ]
    best = None; best_preds = None; best_fcst = None; best_model = None
    for g in grids:
        try:
            model = DartsES(trend=g["trend"], seasonal=g["seasonal"], seasonal_periods=g["seasonal_periods"])
            model.fit(series)
            preds_series = model.predict(len(y_test))
            preds = preds_series.values().flatten()
            metric = smape(y_test.values, preds)
            if (best is None) or (metric < best):
                best = metric
                best_preds = preds
                best_fcst = model.predict(12).values().flatten()
                best_model = model
        except Exception:
            continue
    return best_preds, best_fcst, best_model


def run_pydlm(y_train, y_test):
    from pydlm import dlm, trend, seasonality

    y = y_train.values.astype(float).tolist()
    model = dlm(y) + trend(1, name="trend", w=1.0) + seasonality(12, name="season", w=1.0)
    model.fit()

    def _predict_means(N: int):
        out = model.predictN(date=model.n - 1, N=N)
        if isinstance(out, tuple):
            means = out[0]
        elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], (list, np.ndarray)):
            means = out[0]
        else:
            means = out
        return np.asarray(means, dtype=float)

    preds = []
    for i in range(len(y_test)):
        pred1 = _predict_means(1)[0]
        preds.append(pred1)
        # Append actual next observation as a single-element list, then refit
        model.append([float(y_test.iloc[i])])
        model.fit()

    fcst = _predict_means(12)
    return np.array(preds, dtype=float), np.array(fcst, dtype=float), model

def run_tsfresh_xgb(
    y_train,
    y_test,
    window=12,
    fc_params: str = "efficient",   # "minimal" | "efficient" | "comprehensive"
    fdr_level: float = 0.2,         # relax selection for small-N
    add_lags: bool = True,
    top_k_if_empty: int = 10,
    xgb_params: Optional[dict] = None
):
    import numpy as np
    import pandas as pd
    from xgboost import XGBRegressor
    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction.settings import (
        MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters
    )

    # ------------- helpers -------------
    def get_fc_params(name: str):
        name = (name or "efficient").lower()
        if name.startswith("min"):
            return MinimalFCParameters()
        if name.startswith("comp"):
            return ComprehensiveFCParameters()
        return EfficientFCParameters()

    def make_windows(series: pd.Series, w: int):
        vals = series.values.astype(float)
        if len(vals) <= w:
            raise ValueError(f"Not enough samples ({len(vals)}) for window={w}")
        ids = np.arange(w, len(vals))  # one id per forecast target
        # Build stacked window values
        value_blocks = [vals[i-w:i] for i in ids]
        df_w = pd.DataFrame({
            "id": np.repeat(ids, w),
            "time": np.tile(np.arange(w), len(ids)),
            "value": np.concatenate(value_blocks, axis=0),
        })
        y = pd.Series(vals[ids], index=ids)
        return df_w, y

    def add_simple_lags(X: pd.DataFrame, y_base: pd.Series, w: int):
        # For id i (predicting y[i]), lags are y[i-1], y[i-2], y[i-3]
        ids = y_base.index
        vals = y_train.values.astype(float)  # lags are from training series
        lag_df = pd.DataFrame(index=ids)
        for k in [1, 2, 3]:
            lag_df[f"lag_{k}"] = [vals[i - k] if (i - k) >= 0 else np.nan for i in ids]
        return X.join(lag_df).fillna(0.0)

    def build_X_last(base_series: pd.Series, fc_conf, cols_needed):
        # Extract TSFresh features for the last window
        df_w, _ = make_windows(base_series, window)
        last_id = df_w["id"].max()
        df_last = df_w[df_w["id"] == last_id]
        X_last = extract_features(
            df_last,
            column_id="id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=fc_conf,
            disable_progressbar=True,
            n_jobs=0,
        )
        impute(X_last)
        if add_lags:
            vals = base_series.values.astype(float)
            lag_enrich = {
                "lag_1": vals[-1] if len(vals) >= 1 else 0.0,
                "lag_2": vals[-2] if len(vals) >= 2 else 0.0,
                "lag_3": vals[-3] if len(vals) >= 3 else 0.0,
            }
            for k, v in lag_enrich.items():
                X_last[k] = v
        # Align to training columns
        X_last = X_last.reindex(columns=cols_needed, fill_value=0.0)
        return X_last

    # ------------- training -------------
    # Auto-shrink window if too few samples
    min_windows = 15
    if len(y_train) - window < min_windows and window > 6:
        window = max(6, len(y_train) - min_windows)
    fc_conf = get_fc_params(fc_params)

    df_train_w, y_train_w = make_windows(y_train, window)
    X_train = extract_features(
        df_train_w,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=fc_conf,
        disable_progressbar=True,
        n_jobs=0,
    )
    impute(X_train)

    # Optionally enrich with simple lags to guarantee informative features
    if add_lags:
        X_train = add_simple_lags(X_train, y_train_w, window)

    # Drop zero-variance features early
    nunique = X_train.nunique()
    zero_var_cols = nunique[nunique <= 1].index.tolist()
    if zero_var_cols:
        X_train = X_train.drop(columns=zero_var_cols)

    # TSFresh selection (lenient FDR for small-N)
    try:
        X_train_sel = select_features(
            X_train, y_train_w, ml_task="regression", fdr_level=fdr_level
        )
    except Exception:
        X_train_sel = X_train

    # Fallback: if empty, pick top-K by absolute Pearson correlation
    if X_train_sel.shape[1] == 0:
        corrs = {}
        yv = y_train_w.values
        for c in X_train.columns:
            x = X_train[c].values
            if np.std(x) == 0:
                corrs[c] = 0.0
            else:
                # safe correlation
                try:
                    corrs[c] = np.corrcoef(x, yv)[0, 1]
                except Exception:
                    corrs[c] = 0.0
        top = (
            pd.Series(corrs)
            .abs()
            .sort_values(ascending=False)
            .head(max(1, min(top_k_if_empty, X_train.shape[1])))
            .index.tolist()
        )
        X_train_sel = X_train[top]

    cols_used = X_train_sel.columns.tolist()

    base_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": 0
    }

    ##############################################
    if xgb_params:
        base_params.update(xgb_params)

    model = XGBRegressor(**base_params)
    ###############################################
    model.fit(X_train_sel, y_train_w)

    # ------------- rolling test preds -------------
    preds = []
    extended = y_train.copy()
    for _ in range(len(y_test)):
        X_last = build_X_last(extended, fc_conf, cols_used)
        p = float(model.predict(X_last)[0])
        preds.append(p)
        # Use actual next for recursive feature construction (teacher forcing)
        next_idx = extended.index[-1] + pd.offsets.MonthBegin(1)
        extended = pd.concat([extended, pd.Series([y_test.iloc[len(preds)-1]], index=[next_idx])])

    # ------------- 12-step forecast -------------
    future_base = pd.concat([y_train, y_test])
    fcst_vals = []
    for _ in range(12):
        X_last = build_X_last(future_base, fc_conf, cols_used)
        p = float(model.predict(X_last)[0])
        fcst_vals.append(p)
        future_base = pd.concat(
            [future_base, pd.Series([p], index=[future_base.index[-1] + pd.offsets.MonthBegin(1)])]
        )

    return np.array(preds), np.array(fcst_vals), model

# ---------- NEW: tuned tsfresh XGB ----------
def run_tsfresh_xgb_tuned(y_train, y_test):
    grids = [
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03},
        {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.1},
    ]
    best = None; best_out = None
    for p in grids:
        try:
            preds, fcst, model = run_tsfresh_xgb(y_train, y_test, window=12, fc_params="efficient", xgb_params=p)
            metric = smape(y_test.values, preds)
            if (best is None) or (metric < best):
                best = metric
                best_out = (preds, fcst, model)
        except Exception:
            continue
    return best_out if best_out else (np.array([]), np.array([]), None)

# -----------------------------
# Evaluate all models and pick best
# -----------------------------

def evaluate_models(series: pd.Series, enable_models: dict, target_name: str = "Units", tune: bool = False):
    y_train, y_test = split_train_test(series)
    results = []
    preds_store = {}
    fcst_store = {}

    if enable_models.get("prophet", True):
        try:
            preds, fcst, _ = (run_prophet_tuned if tune else run_prophet)(y_train, y_test)
            results.append(metrics_table(y_test.values, preds, "prophet"))
            preds_store["prophet"] = preds
            fcst_store["prophet"] = fcst
        except Exception as e:
            print("Prophet failed:", e)

    if enable_models.get("skforecast_xgb", True):
        try:
            preds, fcst, _ = (run_skforecast_xgb_tuned if tune else run_skforecast_xgb)(y_train, y_test)
            results.append(metrics_table(y_test.values, preds, "skforecast_xgb"))
            preds_store["skforecast_xgb"] = preds
            fcst_store["skforecast_xgb"] = fcst
        except Exception as e:
            print("skforecast_xgb failed:", e)

    if enable_models.get("sktime_es", True):
        try:
            preds, fcst, _ = (run_sktime_es_tuned if tune else run_sktime_es)(y_train, y_test)
            results.append(metrics_table(y_test.values, preds, "sktime_es"))
            preds_store["sktime_es"] = preds
            fcst_store["sktime_es"] = fcst
        except Exception as e:
            print("sktime_es failed:", e)

    if enable_models.get("darts_es", True):
        try:
            preds, fcst, _ = (run_darts_es_tuned if tune else run_darts_es)(y_train, y_test)
            results.append(metrics_table(y_test.values, preds, "darts_es"))
            preds_store["darts_es"] = preds
            fcst_store["darts_es"] = fcst
        except Exception as e:
            print("darts_es failed:", e)

    if enable_models.get("pydlm", False):
        try:
            preds, fcst, _ = run_pydlm(y_train, y_test)
            results.append(metrics_table(y_test.values, preds, "pydlm"))
            preds_store["pydlm"] = preds
            fcst_store["pydlm"] = fcst
        except Exception as e:
            print("pydlm failed:", e)

    if enable_models.get("tsfresh_xgb", False):
        try:
            preds, fcst, _ = (run_tsfresh_xgb_tuned if tune else run_tsfresh_xgb)(y_train, y_test)
            results.append(metrics_table(y_test.values, preds, "tsfresh_xgb"))
            preds_store["tsfresh_xgb"] = preds
            fcst_store["tsfresh_xgb"] = fcst
        except Exception as e:
            print("tsfresh_xgb failed:", e)


    results_df = pd.DataFrame(results).sort_values("MAPE%") if results else pd.DataFrame()
    best_model = results_df.iloc[0]["model"] if not results_df.empty else None

    test_compare = pd.DataFrame({"Month": y_test.index, "Actual": y_test.values})
    for m, preds in preds_store.items():
        test_compare[f"Pred_{m}"] = preds

    if best_model:
        fcst = fcst_store[best_model]
        fcst_index = pd.date_range(pd.Timestamp("2026-01-01"), periods=12, freq="MS")
        fcst_df = pd.DataFrame({"Month": fcst_index, f"Forecast_{target_name}": fcst})
    else:
        fcst_df = pd.DataFrame(columns=["Month", f"Forecast_{target_name}"])

    return results_df, best_model, test_compare, fcst_df