# Yield Curve Analytics & Forecasting

A quantitative fixed income research project replicating core tools used on 
sell-side rates desks and systematic macro funds — Nelson-Siegel curve fitting, 
Diebold-Li factor forecasting, and the NY Fed recession probability model.

---

## Overview

This project builds a full yield curve analytics and forecasting system using 
daily US Treasury data from 1990 to present. The Nelson-Siegel model is fitted 
to 9,103 daily yield curves, decomposing each into three economically 
interpretable factors: level, slope, and curvature. The Diebold-Li state-space 
framework then treats these factors as a VAR(1) system, producing walk-forward 
out-of-sample forecasts at 1, 3, and 6-month horizons. Finally, a probit 
regression replicates the NY Fed recession probability model using the 
3-month/10-year spread, achieving an AUC of 0.80 on held-out data.

---

## Methodology

### 1. Data Collection (`src/data_loader.py`)
Daily US Treasury yields across 10 maturities (3m to 30y) and macro indicators 
(CPI, unemployment, VIX, yield spreads) are pulled from FRED using the `fredapi` 
library. Data spans 1990–present (~9,100 daily observations).

### 2. Nelson-Siegel Fitting (`src/nelson_siegel.py`)
The three-factor Nelson-Siegel model is fitted to each daily yield curve via 
nonlinear least squares (`scipy.optimize.curve_fit`). The resulting parameters — 
beta0 (level), beta1 (slope), beta2 (curvature) — provide a compact, 
interpretable description of the curve shape on each date. Median in-sample 
RMSE across all dates is 0.05%, confirming excellent fit quality.

### 3. Diebold-Li Forecasting (`src/diebold_li.py`)
Following Diebold and Li (2006), the NS factors are modelled as a VAR(1) system 
and forecast using walk-forward validation (5-year rolling window, ~7,800 
out-of-sample steps). Lag selection via AIC/BIC suggested higher lags, but 
VAR(1) is retained for three reasons: the dominant information gain occurs at 
lag 1; VAR(1) is more parsimonious and stable on rolling windows; and BIC, 
which penalises complexity more heavily, converges toward lower lags.

Granger causality tests reveal that slope (beta1) is not significantly predicted 
by level or curvature (p=0.59), suggesting the short end of the curve follows 
independent monetary policy dynamics. Durbin-Watson statistics are near 2.0 for 
all factors, confirming adequate residual whiteness.

### 4. Recession Probability Model (`src/recession_model.py`)
The NY Fed probit model estimates recession probability by passing a linear 
function of the 3m10y spread through the standard normal CDF (Φ). The dependent 
variable is a 12-month forward-shifted NBER recession indicator (USREC), meaning 
the model predicts whether a recession will occur within the next 12 months rather 
than contemporaneously. The dataset contains only 36 recession months out of 520 
observations, creating a significant class imbalance that inflates raw accuracy. 
The decision threshold is lowered to 0.2 to improve recall on the minority class. 
An AUC of 0.80 confirms the model has strong discriminative power consistent with 
the academic literature.

---

## Key Results

| Model | Metric | Value |
|---|---|---|
| Nelson-Siegel | Median RMSE | 0.0513% |
| Nelson-Siegel | Dates fitted | 9,103 / 9,103 |
| Diebold-Li | 1m beta0 RMSE | 0.3363 |
| Diebold-Li | 1m beta1 RMSE | 0.6874 |
| Diebold-Li | 1m beta2 RMSE | 1.9483 |
| Recession Model | AUC-ROC | 0.8010 |
| Recession Model | Probit coef (spread) | -0.4297 (p<0.001) |

---

## Project Structure

yield-curve-analytics/
├── data/                   # Raw and processed data from FRED
├── src/
│   ├── data_loader.py      # FRED data ingestion
│   ├── nelson_siegel.py    # NS model fitting
│   ├── diebold_li.py       # VAR(1) walk-forward forecasting
│   ├── recession_model.py  # NY Fed probit model
│   └── visualise.py        # Interactive dashboard
├── outputs/
│   └── dashboard.html      # Interactive Plotly dashboard
├── requirements.txt
└── README.md

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/benjamincheng18/yield-curve-analytics.git
cd yield-curve-analytics

# 2. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Add your FRED API key
echo "FRED_API_KEY=your_key_here" > .env

# 4. Run the full pipeline
python src/data_loader.py
python src/nelson_siegel.py
python src/diebold_li.py
python src/recession_model.py
python src/visualise.py

# 5. Open the dashboard
open outputs/dashboard.html
```

---

## Dependencies

See `requirements.txt`. Key libraries:
- `fredapi` — pulls macroeconomic and Treasury yield data from FRED
- `statsmodels` — VAR(1) time series modelling and probit regression
- `scipy` — nonlinear least squares fitting for the Nelson-Siegel model
- `sklearn` — model evaluation metrics (AUC, confusion matrix)
- `plotly` — interactive yield curve and recession probability dashboard
- `python-dotenv` — secure loading of FRED API key from .env

---

## References

- Nelson, C.R. and Siegel, A.F. (1987). Parsimonious modeling of yield curves. 
  *Journal of Business*, 60(4), 473–489.
- Diebold, F.X. and Li, C. (2006). Forecasting the term structure of government 
  bond yields. *Journal of Econometrics*, 130(2), 337–364.
- Estrella, A. and Mishkin, F.S. (1996). The yield curve as a predictor of 
  US recessions. *Federal Reserve Bank of New York Current Issues*, 2(7).