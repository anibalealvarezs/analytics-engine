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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
