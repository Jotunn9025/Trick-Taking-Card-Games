"""Live training metrics dashboard (Dash + Plotly).

Port of the original ad-hoc ``1.py`` monitor: plots raw + smoothed trends for
every ``player_{pid}_<metric>`` column in ``training_metrics.csv`` with a
5-second auto-refresh.

Usage:
    uv sync --extra dashboards
    uv run ttcg-dashboard --csv checkpoints/training_metrics.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

BASE_METRICS = ["avg_payoff", "rl_loss", "sl_loss", "won_pct", "under_pct", "over_pct"]
PLAYERS = [0, 1, 2, 3]


def load_data(csv_file: str) -> pd.DataFrame:
    """Load a training_metrics.csv (with fallback to the default location)."""
    if not os.path.exists(csv_file) and csv_file == "checkpoints/training_metrics.csv":
        for candidate in ["checkpoints17/training_metrics.csv", "checkpoints/training_metrics.csv"]:
            if os.path.exists(candidate):
                csv_file = candidate
                break
    if not os.path.exists(csv_file):
        return pd.DataFrame()
    return pd.read_csv(csv_file).sort_values("episode")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Judgement AI training metrics dashboard")
    parser.add_argument("--csv", type=str, default="checkpoints/training_metrics.csv",
                        help="Path to training_metrics.csv")
    args = parser.parse_args(argv)

    try:
        from dash import Dash, dcc, html, Input, Output
    except ImportError:
        print("Dash dashboard requires the optional dependencies:")
        print("  uv sync --extra dashboards")
        sys.exit(1)

    import plotly.graph_objects as go

    app = Dash(__name__)
    app.title = "Judgement AI Training Dashboard"

    app.layout = html.Div(
        [
            html.H2("Judgement AI Training Metrics",
                    style={"color": "#ddd", "fontFamily": "sans-serif", "textAlign": "center"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Metric:", style={"color": "#ddd", "fontFamily": "sans-serif"}),
                            dcc.Dropdown(
                                id="metric-dropdown",
                                options=[{"label": m.replace("_", " ").title(), "value": m}
                                         for m in BASE_METRICS],
                                value=BASE_METRICS[0],
                                clearable=False,
                                style={"backgroundColor": "#222", "color": "black", "marginTop": "5px"},
                            ),
                        ],
                        style={"width": "30%", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            html.Label("Player:", style={"color": "#ddd", "fontFamily": "sans-serif"}),
                            dcc.Dropdown(
                                id="player-dropdown",
                                options=[{"label": "All Players", "value": "all"}]
                                        + [{"label": f"Player {p}", "value": p} for p in PLAYERS],
                                value="all",
                                clearable=False,
                                style={"backgroundColor": "#222", "color": "black", "marginTop": "5px"},
                            ),
                        ],
                        style={"width": "30%", "display": "inline-block", "marginLeft": "2%"},
                    ),
                    html.Div(
                        [
                            html.Label("Trend Smoothing (Rolling Average):",
                                       style={"color": "#ddd", "fontFamily": "sans-serif"}),
                            html.Div(
                                dcc.Slider(id="smoothing-slider", min=1, max=1000, step=1,
                                           value=5, marks={1: "none", 500: "500", 1000: "1000"}),
                                style={"paddingTop": "10px"},
                            ),
                        ],
                        style={"width": "35%", "display": "inline-block",
                               "marginLeft": "2%", "verticalAlign": "top"},
                    ),
                ],
                style={"padding": "20px", "backgroundColor": "#1a1a1a",
                       "borderRadius": "10px", "marginBottom": "20px", "border": "1px solid #333"},
            ),
            dcc.Interval(id="interval-update", interval=5000, n_intervals=0),  # 5s auto-refresh
            html.Div(dcc.Graph(id="metric-graph"), style={"height": "65vh"}),
        ],
        style={"backgroundColor": "#0a0a0a", "padding": "20px",
               "minHeight": "100vh", "fontFamily": "sans-serif"},
    )

    @app.callback(
        Output("metric-graph", "figure"),
        [Input("metric-dropdown", "value"),
         Input("player-dropdown", "value"),
         Input("smoothing-slider", "value"),
         Input("interval-update", "n_intervals")],
    )
    def update_graph(selected_metric, selected_player, smoothing, n_intervals):
        current_df = load_data(args.csv)
        fig = go.Figure()

        if current_df.empty:
            fig.update_layout(title="No Data Found", template="plotly_dark")
            return fig

        plot_players = PLAYERS if selected_player == "all" else [int(selected_player)]

        for p in plot_players:
            col = f"player_{p}_{selected_metric}"
            if col not in current_df.columns:
                continue
            # Raw (noisy) data as subtle dots; smoothed trend as a bold line.
            fig.add_trace(go.Scatter(
                x=current_df["episode"], y=current_df[col], mode="markers",
                name=f"Player {p} Raw Data", marker=dict(size=4),
                opacity=0.3, showlegend=False,
            ))
            trend = current_df[col].rolling(window=smoothing, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=current_df["episode"], y=trend, mode="lines",
                name=f"Player {p} Trend", line=dict(width=3),
            ))

        fig.update_layout(
            title=f"Tracking <b>{selected_metric.replace('_', ' ').title()}</b> Progression",
            template="plotly_dark",
            xaxis_title="Training Episode",
            yaxis_title="Value",
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(bgcolor="black", font_size=13,
                            font_family="Rockwell", font_color="white"),
        )
        return fig

    print("\n" + "=" * 50)
    print("  DASHBOARD LIVE!")
    print("  Open http://127.0.0.1:8050 in your web browser.")
    print("=" * 50 + "\n")
    app.run(debug=True, use_reloader=False)


if __name__ == "__main__":
    main()