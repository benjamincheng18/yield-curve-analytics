import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

# US Treasury yield maturities and their FRED series IDs
TREASURY_SERIES = {
    "3m": "DGS3MO",
    "6m": "DGS6MO",
    "1y": "DGS1",
    "2y": "DGS2",
    "3y": "DGS3",
    "5y": "DGS5",
    "7y": "DGS7",
    "10y": "DGS10",
    "20y": "DGS20",
    "30y": "DGS30",
}

MACRO_SERIES = {
    "cpi_yoy":        "CPIAUCSL",      # CPI (we'll compute YoY manually)
    "unemployment":   "UNRATE",
    "vix":            "VIXCLS",
    "spread_3m10y":   "T10Y3M",        # NY Fed recession model input
    "spread_2y10y":   "T10Y2Y",
}


def fetch_treasury_yields(start: str = "1990-01-01", end: str = None) -> pd.DataFrame:
    """Fetch daily US Treasury yields across all maturities from FRED."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    frames = {}
    for label, series_id in TREASURY_SERIES.items():
        print(f"  Fetching {label} ({series_id})...")
        frames[label] = fred.get_series(series_id, observation_start=start, observation_end=end)
    df = pd.DataFrame(frames)
    df.index.name = "date"
    df.dropna(how="all", inplace=True)
    return df


def fetch_macro_indicators(start: str = "1990-01-01", end: str = None) -> pd.DataFrame:
    """Fetch macro indicators from FRED."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    frames = {}
    for label, series_id in MACRO_SERIES.items():
        print(f"  Fetching {label} ({series_id})...")
        frames[label] = fred.get_series(series_id, observation_start=start, observation_end=end)
    df = pd.DataFrame(frames)
    df.index.name = "date"
    # Compute CPI YoY % change
    df["cpi_yoy"] = df["cpi_yoy"].pct_change(12) * 100
    df.dropna(how="all", inplace=True)
    return df

if __name__ == "__main__":
    print("Fetching Treasury yields...")
    yields = fetch_treasury_yields()
    yields.to_csv("data/treasury_yields.csv")
    print(f"  Saved {yields.shape[0]} rows to data/treasury_yields.csv")

    print("Fetching macro indicators...")
    macro = fetch_macro_indicators()
    macro.to_csv("data/macro_indicators.csv")
    print(f"  Saved {macro.shape[0]} rows to data/macro_indicators.csv")