"""
IPL Visualization Module
All Plotly chart builders used by the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── THEME ──────────────────────────────────────────────────────────────────────
GOLD = "#FFD700"
TEAL = "#00D4AA"
RED  = "#FF4B4B"
NAVY = "#0A0E2A"
CARD = "#0F1535"
GRID = "rgba(255,255,255,0.07)"

TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=NAVY,
        plot_bgcolor=CARD,
        font=dict(family="DM Sans, sans-serif", color="#E8EAF6"),
        colorway=[GOLD, TEAL, RED, "#A78BFA", "#38BDF8", "#FB923C", "#34D399"],
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        margin=dict(l=50, r=30, t=50, b=50),
    )
)


def _apply_theme(fig, title: str = None, height: int = 400):
    """Apply standard dark theme to a figure."""
    fig.update_layout(
        **TEMPLATE["layout"],
        height=height,
        title=dict(text=title or "", font=dict(size=16, color=GOLD), x=0.02),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
    )
    return fig


# ── PLAYER CHARTS ──────────────────────────────────────────────────────────────

def plot_top_batsmen(df: pd.DataFrame, n: int = 15) -> go.Figure:
    top = df.head(n).sort_values("total_runs")
    fig = go.Figure(go.Bar(
        x=top["total_runs"], y=top["batter"], orientation="h",
        marker=dict(
            color=top["total_runs"],
            colorscale=[[0, "#1a3a5c"], [0.5, TEAL], [1, GOLD]],
            showscale=False,
        ),
        text=top["total_runs"], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Runs: %{x}<br>SR: %{customdata:.1f}<extra></extra>",
        customdata=top["strike_rate"],
    ))
    return _apply_theme(fig, f"Top {n} Run-Scorers", height=500)


def plot_top_bowlers(df: pd.DataFrame, n: int = 15) -> go.Figure:
    top = df.head(n).sort_values("wickets")
    fig = go.Figure(go.Bar(
        x=top["wickets"], y=top["bowler"], orientation="h",
        marker=dict(
            color=top["wickets"],
            colorscale=[[0, "#1a1a3e"], [0.5, "#A78BFA"], [1, RED]],
            showscale=False,
        ),
        text=top["wickets"], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Wickets: %{x}<br>Economy: %{customdata:.2f}<extra></extra>",
        customdata=top["economy"],
    ))
    return _apply_theme(fig, f"Top {n} Wicket-Takers", height=500)


def plot_player_season_runs(df: pd.DataFrame, player: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df["season"], y=df["runs"], mode="lines+markers",
        line=dict(color=GOLD, width=2.5),
        marker=dict(size=8, color=GOLD),
        fill="tozeroy", fillcolor="rgba(255,215,0,0.08)",
        name=player,
    ))
    return _apply_theme(fig, f"{player} — Season-wise Runs")


# ── TEAM CHARTS ────────────────────────────────────────────────────────────────

def plot_win_percentage(df: pd.DataFrame) -> go.Figure:
    top = df.head(15).sort_values("win_pct")
    fig = go.Figure(go.Bar(
        x=top["win_pct"], y=top["team"], orientation="h",
        marker=dict(
            color=top["win_pct"],
            colorscale=[[0, "#1a3a5c"], [0.5, TEAL], [1, GOLD]],
        ),
        text=[f"{v}%" for v in top["win_pct"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Win %: %{x}%<br>Played: %{customdata}<extra></extra>",
        customdata=top["played"],
    ))
    return _apply_theme(fig, "Team Win Percentages (%)", height=480)


def plot_toss_impact(df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        df, names="toss_decision", values="toss_win_pct",
        color_discrete_sequence=[GOLD, TEAL],
        hole=0.45,
    )
    fig.update_traces(textinfo="label+percent", pull=[0.05, 0])
    return _apply_theme(fig, "Toss Win → Match Win % by Decision")


def plot_season_scoring(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure([
        go.Scatter(
            x=df["season"], y=df["avg_first_innings_score"],
            mode="lines+markers",
            line=dict(color=TEAL, width=2.5),
            marker=dict(size=8, color=TEAL),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.07)",
            name="Avg 1st Innings",
        )
    ])
    return _apply_theme(fig, "Season-wise Average First Innings Score")


# ── VENUE CHARTS ───────────────────────────────────────────────────────────────

def plot_venue_stats(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = df.head(top_n).sort_values("avg_first_innings")
    fig = go.Figure(go.Bar(
        x=top["avg_first_innings"], y=top["venue"], orientation="h",
        marker_color=TEAL,
        text=top["avg_first_innings"], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg Score: %{x}<br>Matches: %{customdata}<extra></extra>",
        customdata=top["matches_played"],
    ))
    return _apply_theme(fig, f"Top {top_n} Venues by Avg 1st Innings Score", height=480)


def plot_venue_heatmap(pivot: pd.DataFrame, top_teams: int = 10,
                       top_venues: int = 12) -> go.Figure:
    teams = pivot.sum(axis=1).nlargest(top_teams).index
    venues = pivot.sum(axis=0).nlargest(top_venues).index
    sub = pivot.loc[teams, venues]

    fig = go.Figure(go.Heatmap(
        z=sub.values, x=sub.columns, y=sub.index,
        colorscale=[[0, NAVY], [0.4, "#1a3a5c"], [1, GOLD]],
        text=sub.values.astype(int),
        texttemplate="%{text}",
        hovertemplate="Team: %{y}<br>Venue: %{x}<br>Wins: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **TEMPLATE["layout"],
        height=500,
        title=dict(text="Team Wins per Venue (Heatmap)", font=dict(size=16, color=GOLD), x=0.02),
        xaxis=dict(tickangle=-35),
    )
    return fig


# ── WIN PROBABILITY CHART ──────────────────────────────────────────────────────

def plot_win_probability(balls: list, probs: list, events: list = None,
                         team: str = "Batting Team") -> go.Figure:
    """
    Plot ball-by-ball win probability with momentum overlay and event markers.

    Parameters
    ----------
    balls  : list of ball numbers (0-indexed)
    probs  : list of win probabilities (0–1)
    events : list of dicts with keys 'ball', 'type' ('wicket'|'boundary'), 'desc'
    team   : name of the batting team
    """
    probs_arr = np.array(probs)

    # Smoothed curve (rolling average)
    window = 6
    smooth = pd.Series(probs_arr).rolling(window, min_periods=1, center=True).mean().values

    # Momentum = difference between consecutive smoothed probs
    momentum = np.gradient(smooth) * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
    )

    # ── Win probability area ──────────────────────────────────────────────────
    overs = [b / 6 for b in balls]
    fig.add_trace(
        go.Scatter(
            x=overs, y=smooth * 100,
            mode="lines", name=f"{team} Win %",
            line=dict(color=TEAL, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.12)",
        ), row=1, col=1
    )

    # Reference line at 50 %
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  row=1, col=1)

    # ── Event markers ─────────────────────────────────────────────────────────
    if events:
        for ev in events:
            idx = min(ev["ball"], len(smooth) - 1)
            color = RED if ev["type"] == "wicket" else GOLD
            symbol = "x" if ev["type"] == "wicket" else "star"
            fig.add_trace(
                go.Scatter(
                    x=[idx / 6], y=[smooth[idx] * 100],
                    mode="markers",
                    marker=dict(size=12, color=color, symbol=symbol,
                                line=dict(color="white", width=1)),
                    name=ev.get("desc", ev["type"]),
                    showlegend=False,
                    hovertemplate=f"<b>{ev.get('desc','')}</b><br>Over: %{{x:.1f}}<extra></extra>",
                ), row=1, col=1
            )

    # ── Momentum bar chart ────────────────────────────────────────────────────
    colors = [TEAL if m >= 0 else RED for m in momentum]
    fig.add_trace(
        go.Bar(
            x=overs, y=momentum,
            marker_color=colors, name="Momentum",
            hovertemplate="Over: %{x:.1f}<br>Swing: %{y:.2f}%<extra></extra>",
        ), row=2, col=1
    )

    fig.update_layout(
        **TEMPLATE["layout"],
        height=560,
        title=dict(text=f"Ball-by-Ball Win Probability — {team}", 
                   font=dict(size=16, color=GOLD), x=0.02),
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Win Prob (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Momentum", row=2, col=1)
    fig.update_xaxes(title_text="Over", row=2, col=1)

    return fig


# ── COMPARISON CHART ───────────────────────────────────────────────────────────

def plot_player_comparison(df: pd.DataFrame) -> go.Figure:
    """Radar / bar comparison for two players."""
    metrics = ["Runs", "Strike Rate", "Fours", "Sixes", "Innings"]
    players = df.index.tolist()
    colors = [GOLD, TEAL]

    fig = go.Figure()
    for i, player in enumerate(players):
        vals = [float(df.loc[player, m]) for m in metrics]
        fig.add_trace(go.Bar(
            name=player, x=metrics, y=vals,
            marker_color=colors[i % 2],
        ))
    fig.update_layout(
        **TEMPLATE["layout"],
        barmode="group", height=380,
        title=dict(text="Player Comparison", font=dict(size=16, color=GOLD), x=0.02),
    )
    return fig
