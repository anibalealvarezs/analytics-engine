from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

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
    values: List[float]

class CorrelationRequest(BaseModel):
    series_x: TimeSeriesData
    series_y: TimeSeriesData

class RegressionRequest(BaseModel):
    independent_vars: Dict[str, TimeSeriesData]
    dependent_var: TimeSeriesData

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

@app.post("/api/v1/stats/regression", dependencies=[Depends(get_api_key)])
@limiter.limit("5/minute") 
def calculate_regression(request: Request, payload: RegressionRequest):
    """
    Performs multiple linear regression.
    """
    dfs = []
    df_y = pd.DataFrame({"date": payload.dependent_var.dates, "y": payload.dependent_var.values})
    dfs.append(df_y)
    
    for var_name, ts_data in payload.independent_vars.items():
        df_x = pd.DataFrame({"date": ts_data.dates, var_name: ts_data.values})
        dfs.append(df_x)
        
    from functools import reduce
    df = reduce(lambda left, right: pd.merge(left, right, on="date", how="inner"), dfs).dropna()
    
    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Not enough overlapping data points for regression.")
        
    ind_vars = list(payload.independent_vars.keys())
    X = df[ind_vars]
    y = df["y"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    coefficients = {var: float(coef) for var, coef in zip(ind_vars, model.coef_)}
    
    return {
        "baseline_intercept": float(model.intercept_),
        "coefficients": coefficients,
        "r_squared": float(model.score(X, y)),
        "data_points": len(df)
    }

@app.post("/api/v1/stats/autocorrelation", dependencies=[Depends(get_api_key)])
@limiter.limit("5/minute")
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
@limiter.limit("5/minute")
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
    
    return {
        "success": True,
        "data": {
            "elasticity_coefficients": model.params.drop('const').to_dict(),
            "baseline_log_intercept": float(model.params.get('const', 0)),
            "r_squared": float(model.rsquared),
            "data_points": len(df)
        }
    }

@app.post("/api/v1/stats/macd", dependencies=[Depends(get_api_key)])
@limiter.limit("5/minute")
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
            "last_date": str(df.index[-1].date())
        }
    }

@app.post("/api/v1/stats/anomaly", dependencies=[Depends(get_api_key)])
@limiter.limit("5/minute")
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
            "data_points": len(df)
        }
    }

@app.post("/api/v1/stats/granger", dependencies=[Depends(get_api_key)])
@limiter.limit("5/minute")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
