from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.optimize import curve_fit

app = FastAPI(title="APIs Hub Analytics Engine", version="1.0.0")

# Setup Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Setup Authentication
API_KEY_NAME = "X-Admin-API-Key"
API_KEY = os.environ.get("ADMIN_API_KEY", "dev_secret_key")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=403, detail="Could not validate API KEY"
    )

class TimeSeriesData(BaseModel):
    dates: List[str]
    values: List[Optional[float]]

class CorrelationRequest(BaseModel):
    series_x: TimeSeriesData
    series_y: TimeSeriesData

class EdgeCaseHandling(BaseModel):
    weighted: bool = True
    grouping: str = "none"  # "none", "histogram", "percentile"

class RegressionRequest(BaseModel):
    independent_vars: Dict[str, TimeSeriesData]
    dependent_var: TimeSeriesData
    edge_case_handling: Optional[EdgeCaseHandling] = None

class TrendRequest(BaseModel):
    series: TimeSeriesData
    metric: Optional[str] = None
    window: Optional[int] = 7
    short_window: Optional[int] = 7
    long_window: Optional[int] = 14
    seasonal_periods: Optional[int] = 7

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "analytics-engine"}

@app.post("/api/v1/stats/correlation", dependencies=[Depends(get_api_key)])
@limiter.limit("5/minute", exempt_when=lambda: True) # Example: Bypass rate limit logic can go here if we dynamically check keys
def calculate_correlation(request: Request, payload: CorrelationRequest):
    """
    Calculates the Pearson correlation coefficient between two time series.
    """
    df_x = pd.DataFrame({"date": payload.series_x.dates, "x": payload.series_x.values})
    df_y = pd.DataFrame({"date": payload.series_y.dates, "y": payload.series_y.values})
    
    df = pd.merge(df_x, df_y, on="date", how="inner").dropna()
    
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Not enough overlapping data points for correlation.")
        
    r_value, p_value = pearsonr(df["x"], df["y"])
    
    return {
        "correlation_coefficient": float(r_value),
        "p_value": float(p_value),
        "data_points": len(df)
    }

def _histogram_elbow_grouping(df: pd.DataFrame, x_col: str, label: str = "others") -> pd.DataFrame:
    if x_col not in df.columns:
        return df

    # Log before filtering zeros
    logger.info(f"histogram_elbow_incoming: x_col={x_col}, n_before_filter={len(df)}, n_zero={(df[x_col] <= 0).sum()}, min_x={float(df[x_col].min()):.4f}, max_x={float(df[x_col].max()):.4f}")

    # Exclude zero/negative x values — they are noise, not a low-volume tail
    df = df[df[x_col] > 0].copy()
    if len(df) < 5:
        logger.info(f"histogram_elbow: {len(df)} rows with positive x, need 5, skipping")
        return df

    values = df[x_col].values
    log_vals = np.log10(values)
    n_bins = max(5, min(50, int(np.sqrt(len(df)))))
    counts, bin_edges = np.histogram(log_vals, bins=n_bins)

    mean_count = np.mean(counts)
    elbow_idx = None
    for i in range(len(counts)):
        if counts[i] < mean_count:
            elbow_idx = i
            break

    if elbow_idx is None or elbow_idx == 0:
        logger.info(f"histogram_elbow: elbow_idx={elbow_idx}, no grouping needed")
        return df

    threshold = 10 ** bin_edges[elbow_idx + 1]

    mask = df[x_col] < threshold

    if mask.sum() < 2:
        logger.info(f"histogram_elbow: threshold={threshold:.4f}, tail_size={mask.sum()}, too small, skipping")
        return df

    tail = df[mask]
    head = df[~mask].copy()

    centroid_y = tail["y"].mean()
    centroid_x = tail[x_col].mean()
    logger.info(f"histogram_elbow: x_col={x_col}, threshold={threshold:.4f}, tail_size={mask.sum()}, head_size={len(head)}, centroid=(x={centroid_x:.4f}, y={centroid_y:.6f}), tail_labels={tail['date'].tolist()}")

    centroid = pd.DataFrame([{
        "date": label,
        "y": centroid_y,
        x_col: centroid_x,
    }])

    result = pd.concat([centroid, head], ignore_index=True)
    result = result.sort_values(x_col).reset_index(drop=True)
    return result


@app.post("/api/v1/stats/regression", dependencies=[Depends(get_api_key)])
@limiter.limit("500/minute") 
def calculate_regression(request: Request, payload: RegressionRequest):
    """
    Performs multiple linear regression with optional WLS and histogram-elbow grouping.
    """
    dfs = []
    df_y = pd.DataFrame({"date": payload.dependent_var.dates, "y": payload.dependent_var.values})
    dfs.append(df_y)
    
    for var_name, ts_data in payload.independent_vars.items():
        df_x = pd.DataFrame({"date": ts_data.dates, var_name: ts_data.values})
        dfs.append(df_x)
        
    # Log raw time series unknown entries before merge
    dep_unknown_count = sum(1 for d in payload.dependent_var.dates if d == "unknown")
    ind_unknown_counts = {}
    for var_name, ts_data in payload.independent_vars.items():
        ind_unknown_counts[var_name] = sum(1 for d in ts_data.dates if d == "unknown")
    total_ind_dates = {k: len(v.dates) for k, v in payload.independent_vars.items()}
    logger.info(f"raw_unknown_series: dep_unknown={dep_unknown_count}, ind_unknown={ind_unknown_counts}, total_dep_dates={len(payload.dependent_var.dates)}, total_ind_dates={total_ind_dates}")
    
    from functools import reduce
    df = reduce(lambda left, right: pd.merge(left, right, on="date", how="inner"), dfs).dropna()
    
    unknown_rows = df[df["date"] == "unknown"]
    if len(unknown_rows) > 0:
        ivar_cols = list(payload.independent_vars.keys())
        icol = ivar_cols[0] if ivar_cols else None
        logger.info(f"merge_unknown_check: n_unknown_rows={len(unknown_rows)}, y_vals={unknown_rows['y'].tolist()}, x_vals={unknown_rows[icol].tolist() if icol else 'no_ivar'}")
    else:
        logger.info("merge_unknown_check: no unknown rows in merged df")
    
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Not enough overlapping data points for regression.")
        
    ind_vars = list(payload.independent_vars.keys())
    ech = payload.edge_case_handling

    # Log incoming data before any processing
    if len(ind_vars) == 1:
        x_col = ind_vars[0]
        x_vals = df[x_col].values
        labels = df["date"].tolist()
        max_idx = int(np.argmax(x_vals))
        logger.info(f"analytics_engine_incoming: n={len(df)}, min_x={float(np.min(x_vals)):.4f}, max_x={float(np.max(x_vals)):.4f}, max_x_label={labels[max_idx]}, grouping={ech.grouping if ech else 'none'}, weighted={ech.weighted if ech else 'none'}")
    else:
        logger.info(f"analytics_engine_incoming: n={len(df)}, vars={ind_vars}")

    # --- Step 1: Apply histogram-elbow grouping (chart readability) ---
    # Groups the low-x tail (noisy, low-volume items) into a single
    # synthetic [[[others]]] point so the scatter plot stays readable.
    if ech and ech.grouping == "histogram" and len(ind_vars) == 1:
        x_col = ind_vars[0]
        df = _histogram_elbow_grouping(df, x_col)
        if len(df) < 2:
            raise HTTPException(status_code=400, detail="Not enough data points after histogram grouping.")

    unknown_rows_after = df[df["date"] == "unknown"]
    if len(unknown_rows_after) > 0:
        ivar_cols = list(payload.independent_vars.keys())
        icol = ivar_cols[0] if ivar_cols else None
        logger.info(f"merge_unknown_after_histogram: n_unknown_rows={len(unknown_rows_after)}, y_vals={unknown_rows_after['y'].tolist()}, x_vals={unknown_rows_after[icol].tolist() if icol else 'no_ivar'}")
    else:
        logger.info("merge_unknown_after_histogram: no unknown rows")

    X = df[ind_vars]
    y = df["y"]

    # --- Step 2: Weighted Least Squares (statistical robustness) ---
    if ech and ech.weighted and len(ind_vars) == 1:
        x_col = ind_vars[0]
        weights = np.clip(df[x_col].values, 1e-10, None)
        X_with_const = sm.add_constant(X)
        model = sm.WLS(y, X_with_const, weights=weights).fit()
        coefficients = {var: float(model.params[var]) for var in ind_vars}
        r_squared = float(model.rsquared)
        baseline_intercept = float(model.params["const"])
    else:
        model = LinearRegression()
        model.fit(X, y)
        coefficients = {var: float(coef) for var, coef in zip(ind_vars, model.coef_)}
        r_squared = float(model.score(X, y))
        baseline_intercept = float(model.intercept_)
    
    scatter = None
    if len(ind_vars) == 1:
        scatter = {
            "x": [float(val) for val in X.iloc[:, 0].tolist()],
            "y": [float(val) for val in y.tolist()],
            "x_label": ind_vars[0],
            "labels": [str(d) for d in df["date"].tolist()]
        }
        if scatter["x"]:
            max_idx = int(np.argmax(scatter["x"]))
            logger.info(f"analytics_engine_outgoing: n={len(scatter['x'])}, min_x={min(scatter['x']):.4f}, max_x={max(scatter['x']):.4f}, max_x_label={scatter['labels'][max_idx]}")

    return {
        "baseline_intercept": baseline_intercept,
        "coefficients": coefficients,
        "r_squared": r_squared,
        "data_points": len(df),
        "scatter_data": scatter,
    }

@app.post("/api/v1/stats/autocorrelation", dependencies=[Depends(get_api_key)])
@limiter.limit("500/minute")
def calculate_autocorrelation(request: Request, payload: RegressionRequest, lags: int = 7):
    """
    Calculates the autocorrelation of the dependent variable.
    Great for proving weekly seasonality (lag=7).
    """
    df = pd.DataFrame({"date": payload.dependent_var.dates, "y": payload.dependent_var.values})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').dropna()
    
    series = df['y']
    if len(series) <= lags:
        raise HTTPException(status_code=400, detail=f"Series length ({len(series)}) must be greater than lags ({lags}).")
        
    acf_value = series.autocorr(lag=lags)
    
    return {
        "success": True,
        "data": {
            "autocorrelation": float(acf_value) if not np.isnan(acf_value) else None,
            "lag_days": lags,
            "data_points": len(series)
        }
    }

@app.post("/api/v1/stats/elasticity", dependencies=[Depends(get_api_key)])
@limiter.limit("500/minute")
def calculate_elasticity(request: Request, payload: RegressionRequest):
    """
    Log-Log Regression for elasticity (diminishing returns).
    """
    dfs = []
    df_y = pd.DataFrame({"date": payload.dependent_var.dates, "y": payload.dependent_var.values})
    dfs.append(df_y)
    
    for var_name, ts_data in payload.independent_vars.items():
        df_x = pd.DataFrame({"date": ts_data.dates, var_name: ts_data.values})
        dfs.append(df_x)
        
    from functools import reduce
    df = reduce(lambda left, right: pd.merge(left, right, on="date", how="inner"), dfs).dropna()
    
    # Filter strictly positive for log transform
    df = df[(df.drop(columns=['date']) > 0).all(axis=1)]
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Not enough overlapping strictly positive data points for Log-Log regression.")
        
    log_y = np.log(df["y"])
    
    ind_vars = list(payload.independent_vars.keys())
    log_X = np.log(df[ind_vars])
    
    X = sm.add_constant(log_X)
    model = sm.OLS(log_y, X).fit()
    x_col = ind_vars[0] if len(ind_vars) > 0 else "x"
    
    return {
        "success": True,
        "data": {
            "elasticity_coefficients": model.params.drop('const').to_dict(),
            "coefficients": model.params.drop('const').to_dict(),
            "baseline_log_intercept": float(model.params.get('const', 0)),
            "baseline_intercept": float(model.params.get('const', 0)),
            "r_squared": float(model.rsquared),
            "data_points": len(df),
            "model_type": "log-log",
            "scatter_data": {
                "x_label": x_col,
                "x": [float(val) for val in df[x_col].values],
                "y": [float(val) for val in df['y'].values],
                "labels": [str(d) for d in df["date"].tolist()]
            }
        }
    }

@app.post("/api/v1/stats/macd", dependencies=[Depends(get_api_key)])
@limiter.limit("500/minute")
def calculate_macd(request: Request, payload: RegressionRequest, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
    """
    Moving Average Convergence Divergence.
    """
    df = pd.DataFrame({"date": payload.dependent_var.dates, "y": payload.dependent_var.values})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').dropna()
    
    series = df['y']
    if len(series) < long_window:
         raise HTTPException(status_code=400, detail=f"Requires at least {long_window} days of data for MACD.")
         
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        "success": True,
        "data": {
            "macd": float(macd_line.iloc[-1]),
            "signal": float(signal_line.iloc[-1]),
            "histogram": float(histogram.iloc[-1]),
            "momentum": "accelerating" if histogram.iloc[-1] > 0 else "decelerating",
            "last_date": str(df.index[-1].date()),
            "series": {
                "dates": df.index.strftime('%Y-%m-%d').tolist(),
                "macd_line": [float(x) if not pd.isna(x) else None for x in macd_line.tolist()],
                "signal_line": [float(x) if not pd.isna(x) else None for x in signal_line.tolist()],
                "histogram": [float(x) if not pd.isna(x) else None for x in histogram.tolist()]
            }
        }
    }

@app.post("/api/v1/stats/anomaly", dependencies=[Depends(get_api_key)])
@limiter.limit("500/minute")
def calculate_anomaly(request: Request, payload: RegressionRequest, window: int = 7, threshold_z: float = 2.0):
    """
    Rolling Z-Score Anomaly Detection.
    """
    df = pd.DataFrame({"date": payload.dependent_var.dates, "y": payload.dependent_var.values})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').dropna()
    
    series = df['y']
    if len(series) < window:
         raise HTTPException(status_code=400, detail=f"Requires at least {window} days of data for anomaly detection.")
    
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    rolling_std = rolling_std.replace(0, np.nan)
    z_scores = (series - rolling_mean) / rolling_std
    
    anomalies = df[z_scores.abs() > threshold_z]
    anomaly_dates = [str(d.date()) for d in anomalies.index]
    
    return {
        "success": True,
        "data": {
            "anomaly_detected": len(anomaly_dates) > 0,
            "anomaly_dates": anomaly_dates,
            "threshold_z": threshold_z,
            "data_points": len(df),
            "series": {
                "dates": [str(d.date()) for d in df.index],
                "values": [float(v) for v in df['y'].values]
            }
        }
    }

@app.post("/api/v1/stats/granger", dependencies=[Depends(get_api_key)])
@limiter.limit("500/minute")
def calculate_granger(request: Request, payload: RegressionRequest, maxlag: int = 3):
    """
    Granger Causality Test.
    """
    if len(payload.independent_vars) != 1:
        raise HTTPException(status_code=400, detail="Granger causality endpoint currently supports exactly one independent variable.")
        
    dfs = []
    df_y = pd.DataFrame({"date": payload.dependent_var.dates, "y": payload.dependent_var.values})
    dfs.append(df_y)
    
    for var_name, ts_data in payload.independent_vars.items():
        df_x = pd.DataFrame({"date": ts_data.dates, var_name: ts_data.values})
        dfs.append(df_x)
        
    from functools import reduce
    df = reduce(lambda left, right: pd.merge(left, right, on="date", how="inner"), dfs).dropna()
    
    if len(df) <= (maxlag * 3) + 1:
        raise HTTPException(status_code=400, detail="Not enough data points for the requested lags.")
        
    x_col = list(payload.independent_vars.keys())[0]
    data_2d = df[['y', x_col]].values
    
    try:
        result = grangercausalitytests(data_2d, maxlag=maxlag, verbose=False)
        p_value = float(result[maxlag][0]['ssr_ftest'][1])
        
        return {
            "success": True,
            "data": {
                "predictive": p_value < 0.05,
                "p_value": p_value,
                "lag_days": maxlag,
                "data_points": len(df)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fill_gaps(df: pd.DataFrame, metric: Optional[str]) -> pd.DataFrame:
    if metric in ['cost_per_result', 'purchase_roas', 'cpm', 'cpc', 'ctr']:
        # Efficiency/ratio metrics: interpolate or forward-fill
        df['y'] = df['y'].interpolate(method='linear', limit_direction='both')
        df['y'] = df['y'].ffill().bfill()
    else:
        # Absolute volume metrics: zero-fill
        df['y'] = df['y'].fillna(0)
    return df

@app.post("/api/v1/stats/trend/linear", dependencies=[Depends(get_api_key)])
@limiter.limit("500/minute")
def calculate_trend_linear(request: Request, payload: TrendRequest):
    df = pd.DataFrame({"date": payload.series.dates, "y": payload.series.values})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df = fill_gaps(df, payload.metric)
    df = df.reset_index()
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Not enough data points")
    
    x = np.arange(len(df)).reshape(-1, 1)
    y = df['y'].values
    model = LinearRegression().fit(x, y)
    trend_values = model.predict(x)
    
    return {
        "success": True,
        "trend": [{"date": str(d.date()), "value": float(v)} for d, v in zip(df['date'], trend_values)],
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_)
    }

@app.post("/api/v1/stats/trend/sma", dependencies=[Depends(get_api_key)])
@limiter.limit("500/minute")
def calculate_trend_sma(request: Request, payload: TrendRequest):
    df = pd.DataFrame({"date": payload.series.dates, "y": payload.series.values})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df = fill_gaps(df, payload.metric)
    
    if len(df) < payload.window:
        raise HTTPException(status_code=400, detail="Not enough data points for window")
    
    sma = df['y'].rolling(window=payload.window, min_periods=1).mean()
    
    return {
        "success": True,
        "trend": [{"date": str(d.date()), "value": float(v)} for d, v in zip(sma.index, sma.values)]
    }

@app.post("/api/v1/stats/trend/ema", dependencies=[Depends(get_api_key)])
@limiter.limit("500/minute")
def calculate_trend_ema(request: Request, payload: TrendRequest):
    df = pd.DataFrame({"date": payload.series.dates, "y": payload.series.values})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df = fill_gaps(df, payload.metric)
    
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Not enough data points")
    
    ema_short = df['y'].ewm(span=payload.short_window, adjust=False).mean()
    ema_long = df['y'].ewm(span=payload.long_window, adjust=False).mean()
    
    return {
        "success": True,
        "trend_short": [{"date": str(d.date()), "value": float(v)} for d, v in zip(ema_short.index, ema_short.values)],
        "trend_long": [{"date": str(d.date()), "value": float(v)} for d, v in zip(ema_long.index, ema_long.values)]
    }

@app.post("/api/v1/stats/trend/holt-winters", dependencies=[Depends(get_api_key)])
@limiter.limit("500/minute")
def calculate_trend_holt_winters(request: Request, payload: TrendRequest):
    df = pd.DataFrame({"date": payload.series.dates, "y": payload.series.values})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df = fill_gaps(df, payload.metric)
    
    if len(df) < payload.seasonal_periods * 2:
        # Fallback to simple EMA if not enough data
        trend = df['y'].ewm(span=payload.seasonal_periods, adjust=False).mean()
        return {
            "success": True,
            "trend": [{"date": str(d.date()), "value": float(v)} for d, v in zip(trend.index, trend.values)],
            "note": "Fell back to EMA due to insufficient data for seasonality"
        }
    
    # We want the trend component (level + trend), removing the seasonal component
    model = ExponentialSmoothing(df['y'], trend='add', seasonal='add', seasonal_periods=payload.seasonal_periods, initialization_method="estimated")
    fit_model = model.fit()
    
    # Reconstruct the signal without the seasonal component (just level + trend)
    level = fit_model.level
    trend_component = fit_model.trend if fit_model.trend is not None else 0
    smooth_signal = level + trend_component
    
    return {
        "success": True,
        "trend": [{"date": str(d.date()), "value": float(v)} for d, v in zip(smooth_signal.index, smooth_signal.values)]
    }

@app.post("/api/v1/stats/trend/logarithmic", dependencies=[Depends(get_api_key)])
@limiter.limit("500/minute")
def calculate_trend_logarithmic(request: Request, payload: TrendRequest):
    df = pd.DataFrame({"date": payload.series.dates, "y": payload.series.values})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df = fill_gaps(df, payload.metric)
    df = df.reset_index()
    if len(df) < 3:
        raise HTTPException(status_code=400, detail="Not enough data points")
    
    x = np.arange(1, len(df) + 1)
    y = df['y'].values
    
    def log_func(x, a, b):
        return a * np.log(x) + b
        
    try:
        popt, _ = curve_fit(log_func, x, y)
        trend_values = log_func(x, *popt)
    except:
        # Fallback to linear if optimization fails
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        trend_values = model.predict(x.reshape(-1, 1))
        
    return {
        "success": True,
        "trend": [{"date": str(d.date()), "value": float(v)} for d, v in zip(df['date'], trend_values)]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
