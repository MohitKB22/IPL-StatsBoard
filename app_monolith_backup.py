"""
IPL Advanced Analytics Dashboard (2008–2025)
============================================
Professional-grade Streamlit app with:
  • Player & Team Analytics
  • Venue Insights
  • ML Match Prediction
  • Ball-by-Ball Win Probability
  • Best XI Generator
  • Player Comparison
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Imports ────────────────────────────────────────────────────────────────────
from src.data_loader import load_all_data
from src.preprocessing import get_clean_data
from src.analysis import (
    top_batsmen, top_bowlers, team_win_percentage, toss_impact,
    season_scoring_trends, venue_stats, team_venue_wins,
    compare_batsmen, generate_best_xi, batsman_season_runs,
)
from src.visualization import (
    plot_top_batsmen, plot_top_bowlers, plot_win_percentage,
    plot_toss_impact, plot_season_scoring, plot_venue_stats,
    plot_venue_heatmap, plot_win_probability, plot_player_comparison,
    plot_player_season_runs, NAVY, GOLD, TEAL, RED, CARD,
)
from src.model import get_trained_match_model
from src.win_probability import (
    get_trained_wp_model, create_match_state, extract_key_events,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="IPL Analytics · 2008–2025",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080C20;
    color: #E8EAF6;
}

.stApp { background-color: #080C20; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0E2A 0%, #060918 100%);
    border-right: 1px solid rgba(255,215,0,0.15);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label {
    color: rgba(232,234,246,0.7) !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0F1535 0%, #141840 100%);
    border: 1px solid rgba(255,215,0,0.12);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label { color: rgba(232,234,246,0.55) !important; font-size: 0.75rem; letter-spacing: 0.06em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #FFD700 !important; font-weight: 700; font-size: 1.6rem; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #00D4AA !important; }

/* Headers */
h1 { font-family: 'Space Grotesk', sans-serif; color: #FFD700 !important; letter-spacing: -0.02em; }
h2 { font-family: 'Space Grotesk', sans-serif; color: #E8EAF6 !important; font-weight: 500; }
h3 { color: rgba(232,234,246,0.8) !important; font-weight: 500; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border-bottom: none;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    color: rgba(232,234,246,0.5);
    border-radius: 7px;
    padding: 8px 18px;
    font-size: 0.85rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(255,215,0,0.12) !important;
    color: #FFD700 !important;
    border: 1px solid rgba(255,215,0,0.25) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #FFD700, #FFA500);
    color: #080C20;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(255,215,0,0.35); }

/* Divider */
hr { border-color: rgba(255,215,0,0.12) !important; }

/* Data tables */
[data-testid="stDataFrame"] { border: 1px solid rgba(255,215,0,0.1); border-radius: 8px; }

/* Selectboxes */
.stSelectbox [data-baseweb="select"] > div { background-color: #0F1535 !important; border-color: rgba(255,215,0,0.2) !important; }

/* Info/Success boxes */
.stAlert { border-radius: 8px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080C20; }
::-webkit-scrollbar-thumb { background: rgba(255,215,0,0.3); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_and_process():
    try:
        matches_raw, deliveries_raw = load_all_data()
        matches, deliveries, merged = get_clean_data(matches_raw, deliveries_raw)
        return matches, deliveries, merged
    except FileNotFoundError as e:
        return None, None, None


matches, deliveries, merged = load_and_process()

if matches is None:
    st.error("⚠️  Data files not found. Please place `matches.csv` and `deliveries.csv` in the `data/` folder.")
    st.markdown("""
    **Quick Start:**
    1. Download IPL datasets from [Kaggle – IPL Complete Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
    2. Place `matches.csv` and `deliveries.csv` in the `data/` folder
    3. Reload this page
    """)
    st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-size:2.2rem;'>🏏</div>
        <div style='font-family: Space Grotesk; font-size:1.1rem; font-weight:700; color:#FFD700;'>IPL Analytics</div>
        <div style='font-size:0.7rem; color:rgba(232,234,246,0.4); letter-spacing:0.1em;'>2008 – 2025</div>
    </div>
    <hr style='margin:0.75rem 0;'/>
    """, unsafe_allow_html=True)

    seasons = sorted(matches["season"].unique().tolist())
    all_teams = sorted(list(set(matches["team1"].tolist() + matches["team2"].tolist())))

    selected_season = st.selectbox("Season", ["All Seasons"] + seasons)
    selected_team = st.selectbox("Team", ["All Teams"] + all_teams)

    batters = sorted(deliveries["batter"].dropna().unique().tolist())
    bowlers_list = sorted(deliveries["bowler"].dropna().unique().tolist())
    venues_list = sorted(matches["venue"].dropna().unique().tolist())
    match_ids = sorted(merged["match_id"].unique().tolist())

    selected_player = st.selectbox("Player (Batter)", batters[:50])
    selected_venue = st.selectbox("Venue", ["All Venues"] + venues_list)

    st.markdown("<hr style='margin:0.5rem 0;'/>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem; color:rgba(232,234,246,0.3); text-align:center;'>Data · Scikit-learn · Plotly</div>", unsafe_allow_html=True)


# ── Filter helpers ─────────────────────────────────────────────────────────────
def filter_deliveries():
    df = deliveries.copy()
    if selected_season != "All Seasons":
        df = df[df["season"] == selected_season]
    if selected_team != "All Teams":
        df = df[(df["batting_team"] == selected_team) | (df["bowling_team"] == selected_team)]
    return df


def filter_matches():
    df = matches.copy()
    if selected_season != "All Seasons":
        df = df[df["season"] == selected_season]
    if selected_team != "All Teams":
        df = df[(df["team1"] == selected_team) | (df["team2"] == selected_team)]
    if selected_venue != "All Venues":
        df = df[df["venue"] == selected_venue]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding: 1.5rem 0 0.5rem 0;'>
    <h1 style='margin:0; font-size:2rem;'>🏏 IPL Advanced Analytics Dashboard</h1>
    <p style='color:rgba(232,234,246,0.45); margin:0.3rem 0 0 0; font-size:0.9rem; letter-spacing:0.03em;'>
        18 seasons · Ball-by-ball intelligence · ML-powered predictions
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ─────────────────────────────────────────────────────────────────────
filt_m = filter_matches()
filt_d = filter_deliveries()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Seasons", matches["season"].nunique())
c2.metric("Matches", f"{len(filt_m):,}")
c3.metric("Deliveries", f"{len(filt_d):,}")
c4.metric("Teams", len(all_teams))
c5.metric("Venues", filt_m["venue"].nunique())

st.markdown("<hr style='margin:1rem 0;'/>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "🏏 Players",
    "🏆 Teams",
    "📈 Trends",
    "🏟️ Venues",
    "🤖 Prediction",
    "📊 Win Probability",
    "⚔️ Compare",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · PLAYERS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Batting Leaderboard")
    bat_df = top_batsmen(filt_d, n=20,
                         season=None if selected_season == "All Seasons" else selected_season,
                         team=None if selected_team == "All Teams" else selected_team)
    st.plotly_chart(plot_top_batsmen(bat_df, n=15), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Batsmen — Stats**")
        st.dataframe(
            bat_df[["batter", "total_runs", "strike_rate", "fours", "sixes", "innings"]]
            .head(10)
            .rename(columns={
                "batter": "Player", "total_runs": "Runs",
                "strike_rate": "SR", "fours": "4s", "sixes": "6s", "innings": "Inn"
            }),
            use_container_width=True, hide_index=True,
        )

    with col2:
        st.markdown("**Top Bowlers — Stats**")
        bowl_df = top_bowlers(filt_d, n=10,
                              season=None if selected_season == "All Seasons" else selected_season,
                              team=None if selected_team == "All Teams" else selected_team)
        st.dataframe(
            bowl_df[["bowler", "wickets", "economy", "overs"]].head(10)
            .rename(columns={"bowler": "Bowler", "wickets": "Wkts", "economy": "Econ", "overs": "Ovs"}),
            use_container_width=True, hide_index=True,
        )

    st.markdown("<hr style='margin:1rem 0;'/>", unsafe_allow_html=True)

    st.subheader("Bowling Leaderboard")
    st.plotly_chart(plot_top_bowlers(bowl_df, n=10), use_container_width=True)

    st.markdown("<hr style='margin:1rem 0;'/>", unsafe_allow_html=True)

    st.subheader(f"Season Journey — {selected_player}")
    season_runs = batsman_season_runs(deliveries, selected_player)
    if not season_runs.empty:
        st.plotly_chart(plot_player_season_runs(season_runs, selected_player), use_container_width=True)
    else:
        st.info("No data found for this player.")

    # Best XI
    st.markdown("<hr style='margin:1rem 0;'/>", unsafe_allow_html=True)
    st.subheader("🌟 Best XI Generator")
    gen_season = None if selected_season == "All Seasons" else selected_season
    xi = generate_best_xi(filt_d, season=gen_season)
    cols = st.columns(11)
    icons = ["🏏", "🏏", "🏏", "🏏", "🏏", "🏏", "🏏", "⚾", "⚾", "⚾", "⚾"]
    for i, (col, player) in enumerate(zip(cols, xi["best_xi"])):
        with col:
            st.markdown(f"""
            <div style='background:{CARD}; border:1px solid rgba(255,215,0,0.15); border-radius:8px;
                        padding:8px 4px; text-align:center; font-size:0.7rem; color:#E8EAF6; min-height:60px;'>
                <div style='font-size:1.2rem;'>{icons[i]}</div>
                <div style='font-weight:600; color:#FFD700; font-size:0.65rem; margin-top:4px;'>
                    {player.split()[-1] if ' ' in player else player}
                </div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · TEAMS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Team Performance Overview")

    win_df = team_win_percentage(filt_m)
    st.plotly_chart(plot_win_percentage(win_df), use_container_width=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        toss_df = toss_impact(filt_m)
        st.subheader("Toss Impact")
        st.plotly_chart(plot_toss_impact(toss_df), use_container_width=True)
    with col2:
        st.subheader("Full Team Table")
        st.dataframe(
            win_df[["team", "played", "won", "win_pct"]]
            .rename(columns={"team": "Team", "played": "Played", "won": "Won", "win_pct": "Win %"}),
            use_container_width=True, hide_index=True, height=360,
        )

    # Toss stats
    st.markdown("<hr style='margin:1rem 0;'/>", unsafe_allow_html=True)
    st.subheader("Toss Decision Analysis")
    t_cols = st.columns(len(toss_df))
    for i, row in toss_df.iterrows():
        t_cols[i % len(t_cols)].metric(
            f"Chose to {row['toss_decision'].title()}",
            f"{row['toss_win_pct']}%",
            f"{int(row['matches'])} matches",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · TRENDS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Season Scoring Trends")
    filt_mrg = merged.copy()
    if selected_season != "All Seasons":
        filt_mrg = filt_mrg[filt_mrg["season"] == selected_season]
    if selected_team != "All Teams":
        filt_mrg = filt_mrg[
            (filt_mrg["batting_team"] == selected_team) |
            (filt_mrg["bowling_team"] == selected_team)
        ]

    trend_df = season_scoring_trends(filt_mrg)
    st.plotly_chart(plot_season_scoring(trend_df), use_container_width=True)

    # Over-by-over run rate
    st.subheader("Over-by-Over Average Scoring")
    over_df = (
        filt_d.groupby("over")["total_runs"]
        .mean()
        .reset_index()
        .rename(columns={"total_runs": "avg_runs"})
    )
    fig_over = go.Figure(go.Bar(
        x=over_df["over"] + 1, y=over_df["avg_runs"].round(2),
        marker=dict(
            color=over_df["avg_runs"],
            colorscale=[[0, "#1a3a5c"], [0.5, TEAL], [1, GOLD]],
        ),
        text=over_df["avg_runs"].round(2), textposition="outside",
    ))
    fig_over.update_layout(
        paper_bgcolor="#0A0E2A", plot_bgcolor="#0F1535",
        font=dict(color="#E8EAF6"),
        height=400, margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Over", yaxis_title="Avg Runs / Over",
        title=dict(text="Average Runs per Over", font=dict(color=GOLD, size=15)),
    )
    st.plotly_chart(fig_over, use_container_width=True)

    # Boundary frequency by over
    st.subheader("Boundary Frequency by Over")
    filt_d2 = filt_d.copy()
    filt_d2["is_boundary"] = filt_d2["runs_off_bat"].isin([4, 6]).astype(int)
    boundary_df = filt_d2.groupby("over")["is_boundary"].mean().reset_index()
    boundary_df["boundary_pct"] = (boundary_df["is_boundary"] * 100).round(2)
    fig_b = go.Figure(go.Scatter(
        x=boundary_df["over"] + 1, y=boundary_df["boundary_pct"],
        mode="lines+markers", line=dict(color=GOLD, width=2),
        fill="tozeroy", fillcolor="rgba(255,215,0,0.07)",
    ))
    fig_b.update_layout(
        paper_bgcolor="#0A0E2A", plot_bgcolor="#0F1535",
        font=dict(color="#E8EAF6"), height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Over", yaxis_title="Boundary %",
        title=dict(text="Boundary % per Over", font=dict(color=GOLD, size=15)),
    )
    st.plotly_chart(fig_b, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · VENUES
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Venue Scoring Analysis")
    v_stats = venue_stats(filt_m, merged)
    if not v_stats.empty:
        st.plotly_chart(plot_venue_stats(v_stats), use_container_width=True)
    else:
        st.info("No venue data for current filters.")

    st.subheader("Team Wins per Venue (Heatmap)")
    hm_pivot = team_venue_wins(filt_m)
    if not hm_pivot.empty and hm_pivot.shape[1] >= 2:
        st.plotly_chart(plot_venue_heatmap(hm_pivot), use_container_width=True)
    else:
        st.info("Not enough venue-match combinations for heatmap.")

    # Venue advantage score
    st.markdown("<hr style='margin:1rem 0;'/>", unsafe_allow_html=True)
    st.subheader("Venue Advantage Score")
    st.caption("Venues where home-ish teams consistently outperform — derived from win-rate deviation.")
    adv = v_stats.copy()
    if "avg_first_innings" in adv.columns and "matches_played" in adv.columns:
        mean_score = adv["avg_first_innings"].mean()
        adv["advantage_score"] = (
            ((adv["avg_first_innings"] - mean_score) / mean_score * 100) * 
            np.log1p(adv["matches_played"])
        ).round(1)
        st.dataframe(
            adv[["venue", "avg_first_innings", "matches_played", "advantage_score"]]
            .sort_values("advantage_score", ascending=False)
            .head(15)
            .rename(columns={
                "venue": "Venue", "avg_first_innings": "Avg Score",
                "matches_played": "Matches", "advantage_score": "Advantage Score",
            }),
            use_container_width=True, hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("🤖 Match Winner Prediction")
    st.caption("Pre-match ML prediction using team, toss, venue — powered by Random Forest.")

    with st.spinner("Loading prediction model…"):
        match_model = get_trained_match_model(matches)

    acc = match_model.accuracy
    if acc:
        st.success(f"Model accuracy on holdout set: **{acc}%**")

    col1, col2, col3 = st.columns(3)
    with col1:
        pred_team1 = st.selectbox("Team 1", all_teams, key="pt1")
    with col2:
        remaining = [t for t in all_teams if t != pred_team1]
        pred_team2 = st.selectbox("Team 2", remaining, key="pt2")
    with col3:
        pred_toss_winner = st.selectbox("Toss Winner", [pred_team1, pred_team2], key="ptw")

    col4, col5 = st.columns(2)
    with col4:
        pred_toss_decision = st.selectbox("Toss Decision", ["bat", "field"], key="ptd")
    with col5:
        pred_venue = st.selectbox("Venue", venues_list, key="pv")

    if st.button("🎯 Predict Winner", key="pred_btn"):
        try:
            result = match_model.predict(
                pred_team1, pred_team2, pred_toss_winner,
                pred_toss_decision, pred_venue,
            )
            winner = max(result, key=result.get)
            loser = min(result, key=result.get)
            win_prob = result[winner]
            lose_prob = result[loser]

            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #0F1535, #141840);
                        border: 1px solid rgba(255,215,0,0.3); border-radius: 16px;
                        padding: 2rem; text-align: center; margin-top: 1rem;'>
                <div style='font-size: 0.75rem; color: rgba(232,234,246,0.4);
                             letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.5rem;'>
                    PREDICTED WINNER
                </div>
                <div style='font-family: Space Grotesk; font-size: 2rem;
                             font-weight: 700; color: #FFD700;'>
                    🏆 {winner}
                </div>
                <div style='font-size: 0.95rem; color: rgba(232,234,246,0.6); margin-top: 0.5rem;'>
                    Win probability: <strong style='color:#00D4AA;'>{win_prob}%</strong>
                    &nbsp;vs&nbsp;
                    {loser}: <strong style='color:#FF4B4B;'>{lose_prob}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bar
            fig_bar = go.Figure(go.Bar(
                x=[result[pred_team1], result[pred_team2]],
                y=[pred_team1, pred_team2],
                orientation="h",
                marker=dict(
                    color=[GOLD if pred_team1 == winner else RED,
                           GOLD if pred_team2 == winner else RED],
                ),
                text=[f"{result[pred_team1]}%", f"{result[pred_team2]}%"],
                textposition="inside",
            ))
            fig_bar.update_layout(
                paper_bgcolor="#0A0E2A", plot_bgcolor="#0F1535",
                font=dict(color="#E8EAF6"), height=180,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.0)"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 · WIN PROBABILITY
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("📊 Ball-by-Ball Win Probability")
    st.caption("Select a match to see how win probability evolved delivery by delivery.")

    with st.spinner("Training win probability model…"):
        wp_model = get_trained_wp_model(merged)

    if wp_model.auc:
        st.success(f"Win Probability Model AUC: **{wp_model.auc}**")

    # Match filter
    wp_season = st.selectbox("Season", seasons, index=len(seasons) - 1, key="wp_season")
    season_matches = matches[matches["season"] == wp_season].sort_values("date", ascending=False)
    season_matches["label"] = (
        season_matches["team1"] + " vs " + season_matches["team2"] +
        " — " + season_matches["date"].dt.strftime("%d %b %Y").fillna("")
    )

    if season_matches.empty:
        st.info("No matches for this season.")
    else:
        selected_label = st.selectbox("Select Match", season_matches["label"].tolist(), key="wp_match")
        chosen_row = season_matches[season_matches["label"] == selected_label].iloc[0]
        chosen_match_id = chosen_row["id"]
        batting_team = chosen_row["team2"]  # team that bats 2nd

        if st.button("🔍 Analyse Match", key="wp_btn"):
            with st.spinner("Computing ball-by-ball probabilities…"):
                match_state = create_match_state(merged, chosen_match_id)

            if match_state.empty:
                st.warning("Insufficient data for this match (possibly incomplete deliveries).")
            else:
                wp_result = wp_model.predict_match(match_state)

                if wp_result.empty:
                    st.warning("Could not compute probabilities for this match.")
                else:
                    events = extract_key_events(match_state)

                    balls = wp_result["ball_number"].tolist()
                    probs = wp_result["win_probability"].tolist()

                    fig_wp = plot_win_probability(balls, probs, events=events, team=batting_team)
                    st.plotly_chart(fig_wp, use_container_width=True)

                    # Momentum swing annotation
                    smooth = pd.Series(probs).rolling(6, min_periods=1, center=True).mean()
                    momentum = smooth.diff().abs()
                    top_swings = momentum.nlargest(3).index.tolist()

                    st.markdown("**⚡ Top Momentum Shifts**")
                    sw_cols = st.columns(3)
                    for i, idx in enumerate(top_swings):
                        ball = int(balls[idx]) if idx < len(balls) else 0
                        prob = round(float(probs[idx]) * 100, 1) if idx < len(probs) else 0
                        over = ball // 6
                        ball_in_over = ball % 6
                        sw_cols[i].metric(
                            f"Over {over}.{ball_in_over}",
                            f"{prob}%",
                            f"Δ {round(float(momentum.iloc[idx]) * 100, 1)}%",
                        )

                    # Simulation slider (BONUS)
                    st.markdown("<hr style='margin:1rem 0;'/>", unsafe_allow_html=True)
                    st.subheader("🎮 Ball-by-Ball Simulator")
                    max_ball = len(balls) - 1
                    sim_ball = st.slider(
                        "Scrub through the innings →", 0, max_ball, max_ball // 2,
                        format="Ball %d"
                    )
                    sim_prob = round(float(probs[sim_ball]) * 100, 1)
                    sim_over = balls[sim_ball] // 6
                    sim_b = balls[sim_ball] % 6

                    st.markdown(f"""
                    <div style='background:{CARD}; border:1px solid rgba(0,212,170,0.25);
                                border-radius:12px; padding:1.2rem 1.5rem; display:flex;
                                align-items:center; gap:2rem;'>
                        <div>
                            <div style='font-size:0.7rem; color:rgba(232,234,246,0.4); text-transform:uppercase;'>Over</div>
                            <div style='font-size:1.6rem; font-weight:700; color:{GOLD};'>{sim_over}.{sim_b}</div>
                        </div>
                        <div>
                            <div style='font-size:0.7rem; color:rgba(232,234,246,0.4); text-transform:uppercase;'>Win Probability</div>
                            <div style='font-size:1.6rem; font-weight:700; color:{TEAL};'>{sim_prob}%</div>
                        </div>
                        <div style='flex:1; background:rgba(255,255,255,0.04); border-radius:8px; height:16px; position:relative; overflow:hidden;'>
                            <div style='background:linear-gradient(90deg,{TEAL},{GOLD}); height:100%; width:{sim_prob}%; transition:width 0.3s ease;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Commentary-style insight
                    if sim_prob > 75:
                        commentary = f"🟢 {batting_team} are cruising — overwhelming favourites at this stage."
                    elif sim_prob > 55:
                        commentary = f"🟡 {batting_team} slightly ahead, but the match is still very much alive."
                    elif sim_prob > 45:
                        commentary = "⚖️ It's anyone's game — a perfectly balanced contest."
                    elif sim_prob > 25:
                        commentary = f"🟠 Defending team on top — {batting_team} need a miracle partnership."
                    else:
                        commentary = f"🔴 {batting_team} on the ropes — mountain to climb from here."
                    st.info(f"**Commentary:** {commentary}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 · COMPARE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("⚔️ Player vs Player Comparison")

    c1, c2 = st.columns(2)
    with c1:
        player_a = st.selectbox("Player A", batters, index=0, key="pa")
    with c2:
        remaining_b = [b for b in batters if b != player_a]
        player_b = st.selectbox("Player B", remaining_b, index=0, key="pb")

    if st.button("Compare →", key="cmp_btn"):
        cmp_df = compare_batsmen(deliveries, player_a, player_b)
        st.plotly_chart(plot_player_comparison(cmp_df), use_container_width=True)

        col1, col2 = st.columns(2)
        for i, player in enumerate([player_a, player_b]):
            col = col1 if i == 0 else col2
            with col:
                row = cmp_df.loc[player]
                col.markdown(f"""
                <div style='background:{CARD}; border:1px solid rgba(255,215,0,0.12);
                             border-radius:12px; padding:1.2rem; text-align:center;'>
                    <div style='font-family:Space Grotesk; font-size:1.1rem; font-weight:700;
                                 color:{"#FFD700" if i==0 else "#00D4AA"}'>{player}</div>
                    <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-top:1rem;'>
                        <div style='background:rgba(255,255,255,0.04); border-radius:8px; padding:0.5rem;'>
                            <div style='font-size:0.65rem; color:rgba(232,234,246,0.4); text-transform:uppercase;'>Runs</div>
                            <div style='font-size:1.2rem; font-weight:700;'>{int(row['Runs']):,}</div>
                        </div>
                        <div style='background:rgba(255,255,255,0.04); border-radius:8px; padding:0.5rem;'>
                            <div style='font-size:0.65rem; color:rgba(232,234,246,0.4); text-transform:uppercase;'>SR</div>
                            <div style='font-size:1.2rem; font-weight:700;'>{row['Strike Rate']}</div>
                        </div>
                        <div style='background:rgba(255,255,255,0.04); border-radius:8px; padding:0.5rem;'>
                            <div style='font-size:0.65rem; color:rgba(232,234,246,0.4); text-transform:uppercase;'>4s</div>
                            <div style='font-size:1.2rem; font-weight:700;'>{int(row['Fours'])}</div>
                        </div>
                        <div style='background:rgba(255,255,255,0.04); border-radius:8px; padding:0.5rem;'>
                            <div style='font-size:0.65rem; color:rgba(232,234,246,0.4); text-transform:uppercase;'>6s</div>
                            <div style='font-size:1.2rem; font-weight:700;'>{int(row['Sixes'])}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Match side-by-side (BONUS)
    st.markdown("<hr style='margin:1.5rem 0;'/>", unsafe_allow_html=True)
    st.subheader("📋 Match Side-by-Side Comparison")
    st.caption("Compare two matches' innings progression.")

    all_match_ids = sorted(merged["match_id"].unique().tolist())
    m1, m2 = st.columns(2)
    with m1:
        mid1 = st.selectbox("Match A ID", all_match_ids, index=0, key="mid1")
    with m2:
        mid2 = st.selectbox("Match B ID", all_match_ids, index=min(1, len(all_match_ids)-1), key="mid2")

    if st.button("Compare Matches →", key="mc_btn"):
        def match_inn1_runs(mid):
            inn = merged[(merged["match_id"] == mid) & (merged["innings"] == 1)]
            return inn.groupby("over")["total_runs"].sum().reset_index()

        mr1 = match_inn1_runs(mid1)
        mr2 = match_inn1_runs(mid2)

        fig_cmp = go.Figure([
            go.Scatter(x=mr1["over"]+1, y=mr1["total_runs"].cumsum(),
                       name=f"Match {mid1}", mode="lines+markers",
                       line=dict(color=GOLD, width=2)),
            go.Scatter(x=mr2["over"]+1, y=mr2["total_runs"].cumsum(),
                       name=f"Match {mid2}", mode="lines+markers",
                       line=dict(color=TEAL, width=2)),
        ])
        fig_cmp.update_layout(
            paper_bgcolor="#0A0E2A", plot_bgcolor="#0F1535",
            font=dict(color="#E8EAF6"), height=380,
            xaxis_title="Over", yaxis_title="Cumulative Runs",
            title=dict(text="1st Innings Progression Comparison",
                       font=dict(color=GOLD, size=15)),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem 0;
             color:rgba(232,234,246,0.2); font-size:0.75rem; letter-spacing:0.05em;'>
    IPL Analytics Dashboard · Built with Streamlit, Plotly & Scikit-learn<br>
    Data: 2008–2025 · ML: RandomForest + GradientBoosting
</div>
""", unsafe_allow_html=True)
