import os
import numpy as np
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import warnings

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))


def fetch_recession_indicator(start: str = "1970-01-01") -> pd.DataFrame:
    """Fetch USREC and T10Y3M monthly series from FRED."""
    print("  Fetching USREC (recession indicator)...")
    usrec = fred.get_series("USREC", observation_start=start)
    usrec.index = usrec.index.to_period("M").to_timestamp()

    print("  Fetching T10Y3M (3m10y spread)...")
    spread = fred.get_series("T10Y3M", observation_start=start)

    # Resample spread to monthly (take end-of-month value)
    spread_monthly = spread.resample("MS").last()

    df = pd.DataFrame({
        "spread_3m10y": spread_monthly,
        "USREC": usrec
    }).dropna()

    df.index.name = "date"
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply 12-month forward lag to USREC to create recession_actual."""
    df['recession_actual'] = df['USREC'].shift(-12)  
    df = df.drop(columns=['USREC'])  
    df = df.dropna() 
    return df


def fit_probit(df: pd.DataFrame):
    """Fit NY Fed probit model: P(recession in 12m) = Φ(β0 + β1 * spread)."""
    X = sm.add_constant(df["spread_3m10y"])
    y = df["recession_actual"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.Probit(y, X)
        result = model.fit(disp=False)

    print(result.summary())
    return result


def evaluate_model(result, df: pd.DataFrame) -> pd.DataFrame:
    """ 
    1. Generate recession_prob using result.predict()
    2. Generate recession_pred by thresholding at 0.5
    3. Print AUC, confusion matrix, classification report
    """
    X = sm.add_constant(df["spread_3m10y"])
    recession_prob = result.predict(X)
    recession_pred = (recession_prob > 0.2).astype(int)

    print(f"AUC: {roc_auc_score(df['recession_actual'], recession_prob):.4f}")
    print(confusion_matrix(df['recession_actual'], recession_pred))
    print(classification_report(df['recession_actual'], recession_pred))

    return pd.DataFrame({
        "spread_3m10y":    df["spread_3m10y"],
        "recession_actual": df["recession_actual"],
        "recession_prob":   recession_prob,
        "recession_pred":   recession_pred
    }, index=df.index)
    


if __name__ == "__main__":
    print("Fetching data...")
    df = fetch_recession_indicator()

    print("\nPreparing data...")
    df = prepare_data(df)
    print(f"  {len(df)} monthly observations ready")

    print("\nFitting probit model...")
    result = fit_probit(df)

    print("\nEvaluating model...")
    predictions = evaluate_model(result, df)
    predictions.to_csv("data/recession_predictions.csv")
    print(f"\n  Saved to data/recession_predictions.csv")

    # Preview
    print("\nLatest 5 predictions:")
    print(predictions.tail())