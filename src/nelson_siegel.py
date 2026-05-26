import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import warnings

# Maturities in years corresponding to our column labels
MATURITY_MAP = {
    "3m": 0.25, "6m": 0.5, "1y": 1.0, "2y": 2.0, "3y": 3.0,
    "5y": 5.0, "7y": 7.0, "10y": 10.0, "20y": 20.0, "30y": 30.0
}


def nelson_siegel(tau: np.ndarray, beta0: float, beta1: float, beta2: float, lam: float) -> np.ndarray:
    """
    Nelson-Siegel yield curve model.
    beta0 = level, beta1 = slope, beta2 = curvature, lam = decay factor
    """
    factor1 = (1 - np.exp(-tau / lam)) / (tau / lam)
    factor2 = factor1 - np.exp(-tau / lam)
    return beta0 + beta1 * factor1 + beta2 * factor2


def fit_ns_single(yields: pd.Series, maturities: np.ndarray) -> dict:
    """Fit Nelson-Siegel to a single row of yields. Returns parameters or None."""
    mask = ~np.isnan(yields.values)
    y = yields.values[mask]
    t = maturities[mask]

    if len(y) < 4:
        return None
    
    try: 
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(
                nelson_siegel, t, y, 
                p0=[4.0, -2.0, 2.0, 1.5],       # initial guesses
                bounds=([0, -15, -15, 0.1], [15, 15, 15, 10]),
                maxfev=5000
            )
        beta0, beta1, beta2, lam = popt
        fitted = nelson_siegel(t, *popt)
        rmse = np.sqrt(np.mean((y - fitted) ** 2))
        return {"beta0": beta0, "beta1": beta1, "beta2": beta2, "lam": lam, "rmse": rmse}
    except Exception:
        return None
    

def fit_ns_panel(yields_df: pd.DataFrame) -> pd.DataFrame:
    """Fit Nelson-Siegel to every date in the yields DataFrame."""
    maturities = np.array([MATURITY_MAP[col] for col in yields_df.columns])
    results = []

    for date, row in yields_df.iterrows():
        params = fit_ns_single(row, maturities)
        if params:
            params["date"] = date
            results.append(params)

    df = pd.DataFrame(results).set_index("date")
    print(f"  Fitted NS model to {len(df)} / {len(yields_df)} dates")
    print(f"  Median RMSE: {df['rmse'].median():.4f} bps-equivalent")
    return df


def reconstruct_curve(params: pd.Series, maturities: np.ndarray = None) -> np.ndarray:
    """Reconstruct yield curve from NS parameters at given maturities."""
    if maturities is None:
        maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    return nelson_siegel(maturities, params["beta0"], params["beta1"], params["beta2"], params["lam"])


if __name__ == "__main__":
    yields = pd.read_csv("data/treasury_yields.csv", index_col="date", parse_dates=True)

    print("Fitting Nelson-Siegel model...")
    ns_params = fit_ns_panel(yields)
    ns_params.to_csv("data/ns_parameters.csv")
    print(f"  Saved to data/ns_parameters.csv")

    # Sanity check — print latest fitted parameters
    latest = ns_params.iloc[-1]
    print(f"\nLatest NS parameters ({ns_params.index[-1].date()}):")
    print(f"  Level  (beta0): {latest['beta0']:.4f}")
    print(f"  Slope  (beta1): {latest['beta1']:.4f}")
    print(f"  Curve  (beta2): {latest['beta2']:.4f}")
    print(f"  Lambda (lam):   {latest['lam']:.4f}")
    print(f"  RMSE:           {latest['rmse']:.4f}")
