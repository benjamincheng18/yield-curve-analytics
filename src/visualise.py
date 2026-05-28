import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nelson_siegel import nelson_siegel, MATURITY_MAP

MATURITIES = np.array(list(MATURITY_MAP.values()))
MATURITY_LABELS = list(MATURITY_MAP.keys())


def load_data() -> dict:
    """Load all project CSVs into dataframes."""
    return {
        "yields":     pd.read_csv("data/treasury_yields.csv",        index_col="date", parse_dates=True),
        "ns":         pd.read_csv("data/ns_parameters.csv",          index_col="date", parse_dates=True),
        "forecasts":  pd.read_csv("data/dl_forecasts.csv",           index_col="date", parse_dates=True),
        "recession":  pd.read_csv("data/recession_predictions.csv",  index_col="date", parse_dates=True),
    }


def add_recession_bands(fig: go.Figure, recession_df: pd.DataFrame, row: int, col: int) -> None:
    """Add NBER recession shading to a subplot."""
    changes = recession_df["recession_actual"].diff()
    starts  = recession_df[changes == 1].index
    ends    = recession_df[changes == -1].index

    # Handle edge case where data starts mid-recession
    if recession_df["recession_actual"].iloc[0] == 1:
        starts = [recession_df.index[0]] + list(starts)

    for start, end in zip(starts, ends):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="grey", opacity=0.25,
            layer="below", line_width=0,
            row=row, col=col
        )


def plot_historical_curves(fig: go.Figure, yields_df: pd.DataFrame, row: int, col: int) -> None:
    """Plot yield curves for selected years across all maturities."""
    years = [1995, 2000, 2005, 2008, 2010, 2015, 2020, 2023]
    for y in years:
        yc = yields_df.asof(pd.Timestamp(f"{y}-01-01"))
        fig.add_trace(go.Scatter(
            x=MATURITY_LABELS,
            y=yc.values,
            name=str(y),
            mode="lines+markers"
        ), row=row, col=col)



def plot_ns_factors(fig: go.Figure, ns_df: pd.DataFrame, recession_df: pd.DataFrame, row: int, col: int) -> None:
    """Plot beta0, beta1, beta2 as three separate lines over time."""
    for beta, color in zip(["beta0", "beta1", "beta2"], ["blue", "red", "green"]):
        fig.add_trace(go.Scatter(
            x=ns_df.index, y=ns_df[beta],
            name=beta, line=dict(color=color, width=1),
            mode="lines"
        ), row=row, col=col)

    fig.add_hline(y=0, line_dash="dash", line_color="black", row=row, col=col)

    add_recession_bands(fig, data["recession"], row=row, col=col)  


def plot_beta_forecasts(fig: go.Figure, forecasts_df: pd.DataFrame, row: int, col: int) -> None:
    """Plot walk-forward forecast vs actual for beta0 at 1m horizon."""
    df_1m = forecasts_df[forecasts_df["horizon"] == "1m"].copy()

    for beta, color in zip(["beta0", "beta1", "beta2"], ["blue", "red", "green"]):
        fig.add_trace(go.Scatter(
            x=df_1m.index, y=df_1m[f"{beta}_actual"],
            name=f"{beta} actual", line=dict(color=color, width=1),
            legendgroup=beta
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=df_1m.index, y=df_1m[f"{beta}_forecast"],
            name=f"{beta} forecast", line=dict(color=color, width=1, dash="dash"),
            legendgroup=beta
        ), row=row, col=col)


def plot_recession_prob(fig: go.Figure, recession_df: pd.DataFrame, row: int, col: int) -> None:
    """Plot recession_prob as a filled area chart over time."""
    fig.add_trace(go.Scatter(
        x=recession_df.index, y=recession_df['recession_prob'],
        name="Recession Probability",
        fill='tozeroy',             
        line=dict(color='red')
    ), row=row, col=col)

    fig.add_hline(y=0.2, line_dash="dash", line_color="red", row=row, col=col)

    add_recession_bands(fig, recession_df, row=row, col=col)



def build_dashboard(data: dict) -> go.Figure:
    """Combine all plots into a single dashboard figure."""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            "Historical US Treasury Yield Curves",
            "Nelson-Siegel Factors Over Time (Level / Slope / Curvature)",
            "Diebold-Li Walk-Forward Forecast vs Actual (1m Horizon)",
            "NY Fed Recession Probability Model (3m10y Spread)"
        ],
        vertical_spacing=0.08,
        shared_xaxes=False
    )

    plot_historical_curves(fig, data["yields"], row=1, col=1)
    plot_ns_factors(fig, data["ns"], data["recession"], row=2, col=1)
    plot_beta_forecasts(fig, data["forecasts"], row=3, col=1)
    plot_recession_prob(fig, data["recession"], row=4, col=1)

    fig.update_layout(
        height=1800,
        title_text="US Yield Curve Analytics Dashboard",
        title_font_size=20,
        showlegend=True,
        template="plotly_white"
    )
    return fig


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()

    print("Building dashboard...")
    fig = build_dashboard(data)

    os.makedirs("outputs", exist_ok=True)
    fig.write_html("outputs/dashboard.html")
    print("Saved to outputs/dashboard.html")
    print("Open the file in your browser to view.")