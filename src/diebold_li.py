import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nelson_siegel import reconstruct_curve, MATURITY_MAP

WINDOW = 1260          # 5-year rolling window (~252 trading days/year)
HORIZONS = {
    "1m":  21,         # trading days
    "3m":  63,
    "6m":  126
}


def select_lag(df: pd.DataFrame, maxlags: int = 10) -> None:
    """
    Use statsmodels VAR to run lag selection on df.
    Print the AIC and BIC optimal lag.
    """
    model = VAR(df)
    results = model.select_order(maxlags)
    print(results.summary())
    print(f"  AIC optimal lag: {results.aic}")
    print(f"  BIC optimal lag: {results.bic}")


def fit_var1(df: pd.DataFrame):
    """Fit VAR(1) on the given window. Returns fitted model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VAR(df)
        result = model.fit(1)
    return result


def forecast_betas(result, steps: int) -> np.ndarray:
    """
    Produce a multi-step forecast from a fitted VAR(1) result.
    Returns array of shape (steps, 3) — one row per step, one col per beta.
    """
    last_obs = result.endog[-result.k_ar:]
    forecast = result.forecast(last_obs, steps=steps)
    return forecast


def reconstruct_yields(beta_forecasts: np.ndarray) -> np.ndarray:
    """Convert beta forecasts to yield curves using reconstruct_curve()."""
    yields = []
    for row in beta_forecasts:
        params = pd.Series({
            "beta0":  row[0], 
            "beta1":  row[1], 
            "beta2":  row[2], 
            "lam":  1.5
        })
        yields.append(reconstruct_curve(params))
    return np.array(yields)


def evaluate_model(result) -> None:
    """
    Run and print diagnostics on the fitted VAR(1):
    1. Granger causality — does each beta Granger-cause the others?
    2. Durbin-Watson statistic on residuals to check autocorrelation
    """
    betas = ["beta0", "beta1", "beta2"]
    print("Granger causality test: ")
    for caused in betas:
        causing = [b for b in betas if b != caused]
        test = result.test_causality(caused, causing, kind="f")
        print(f"  {causing} -> {caused}: p-value = {test.pvalue:.4f}")

    print("\nDurbin-Watson Statistics (near 2.0 = no autocorrelation):")
    dw = durbin_watson(result.resid)
    for i, beta in enumerate(betas):
        print(f"  {beta}: {dw[i]:.4f}")


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Roll through data, fit VAR(1) at each step, collect forecasts."""
    betas = ["beta0", "beta1", "beta2"]
    records = []
    dates = df.index

    print(f"  Running walk-forward validation ({len(dates) - WINDOW} steps)...")
    for i in range(WINDOW, len(dates)):
        train = df.iloc[i - WINDOW:i]
        date = dates[i]

        result = fit_var1(train)

        for horizon_label, horizon_days in HORIZONS.items():
            if i + horizon_days >= len(dates):
                continue

            forecast = forecast_betas(result, steps=horizon_days)
            forecast_vals = forecast[-1]                      # take last step
            actual_vals = df.iloc[i + horizon_days][betas].values

            record = {"date": date, "horizon": horizon_label}
            for j, beta in enumerate(betas):
                record[f"{beta}_forecast"] = forecast_vals[j]
                record[f"{beta}_actual"]   = actual_vals[j]
                record[f"{beta}_error"]    = forecast_vals[j] - actual_vals[j]
            records.append(record)

        if (i - WINDOW) % 252 == 0:
            year = date.year
            print(f"    {year} done...")

    return pd.DataFrame(records).set_index("date")


if __name__ == "__main__":
    ns_params = pd.read_csv("data/ns_parameters.csv", index_col="date", parse_dates=True)
    betas = ns_params[["beta0", "beta1", "beta2"]].dropna()

    # Lag selection diagnostic
    print("Running lag selection...")
    select_lag(betas)

    # Evaluate model on full sample for diagnostics
    print("\nFitting VAR(1) on full sample for diagnostics...")
    full_result = fit_var1(betas)
    evaluate_model(full_result)

    # Walk-forward forecasting
    print("\nRunning walk-forward forecasting...")
    forecasts = walk_forward(betas)
    forecasts.to_csv("data/dl_forecasts.csv")
    print(f"\n  Saved {len(forecasts)} forecast records to data/dl_forecasts.csv")

    # Summary statistics
    print("\nForecast RMSE by horizon:")
    for horizon in ["1m", "3m", "6m"]:
        subset = forecasts[forecasts["horizon"] == horizon]
        for beta in ["beta0", "beta1", "beta2"]:
            rmse = np.sqrt((subset[f"{beta}_error"] ** 2).mean())
            print(f"  {horizon} | {beta}: {rmse:.4f}")