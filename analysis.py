"""
IPL Analysis Module
Provides analytics functions for players, teams, venues, and seasons.
"""

import pandas as pd
import numpy as np
import streamlit as st


# ── BATSMAN ANALYTICS ──────────────────────────────────────────────────────────

@st.cache_data
def top_batsmen(deliveries: pd.DataFrame, n: int = 20, season: str = None,
                team: str = None) -> pd.DataFrame:
    """Return top n batsmen by total runs."""
    df = deliveries.copy()
    if season:
        df = df[df["season"] == season]
    if team:
        df = df[df["batting_team"] == team]

    grp = df.groupby("batter").agg(
        total_runs=("runs_off_bat", "sum"),
        balls_faced=("runs_off_bat", "count"),
        fours=("runs_off_bat", lambda x: (x == 4).sum()),
        sixes=("runs_off_bat", lambda x: (x == 6).sum()),
        innings=("match_id", "nunique"),
    ).reset_index()

    grp["strike_rate"] = (grp["total_runs"] / grp["balls_faced"] * 100).round(2)
    grp["avg_per_innings"] = (grp["total_runs"] / grp["innings"]).round(2)

    return grp.sort_values("total_runs", ascending=False).head(n)


@st.cache_data
def batsman_season_runs(deliveries: pd.DataFrame, batter: str) -> pd.DataFrame:
    """Season-wise runs for a specific batsman."""
    df = deliveries[deliveries["batter"] == batter].copy()
    return (
        df.groupby("season")["runs_off_bat"]
        .sum()
        .reset_index()
        .rename(columns={"runs_off_bat": "runs"})
        .sort_values("season")
    )


# ── BOWLER ANALYTICS ───────────────────────────────────────────────────────────

@st.cache_data
def top_bowlers(deliveries: pd.DataFrame, n: int = 20, season: str = None,
                team: str = None) -> pd.DataFrame:
    """Return top n bowlers by wickets."""
    df = deliveries.copy()
    if season:
        df = df[df["season"] == season]
    if team:
        df = df[df["bowling_team"] == team]

    # Count only dismissals that credit the bowler
    credited = ["caught", "bowled", "lbw", "stumped",
                "caught and bowled", "hit wicket"]
    wicket_df = df[df["dismissal_kind"].isin(credited)]

    wickets = wicket_df.groupby("bowler").size().reset_index(name="wickets")
    balls = df.groupby("bowler").size().reset_index(name="balls")
    runs_given = df.groupby("bowler")["runs_off_bat"].sum().reset_index()

    grp = wickets.merge(balls, on="bowler").merge(runs_given, on="bowler")
    grp["overs"] = (grp["balls"] / 6).round(2)
    grp["economy"] = (grp["runs_off_bat"] / grp["overs"]).round(2)
    grp["bowling_avg"] = np.where(
        grp["wickets"] > 0, grp["runs_off_bat"] / grp["wickets"], np.inf
    ).round(2)

    return grp.sort_values("wickets", ascending=False).head(n)


# ── TEAM ANALYTICS ─────────────────────────────────────────────────────────────

@st.cache_data
def team_win_percentage(matches: pd.DataFrame) -> pd.DataFrame:
    """Win percentage per team across all seasons."""
    teams = list(set(matches["team1"].tolist() + matches["team2"].tolist()))
    rows = []
    for team in teams:
        played = matches[(matches["team1"] == team) | (matches["team2"] == team)]
        won = matches[matches["winner"] == team]
        rows.append({
            "team": team,
            "played": len(played),
            "won": len(won),
            "win_pct": round(len(won) / len(played) * 100, 1) if len(played) else 0,
        })
    return pd.DataFrame(rows).sort_values("win_pct", ascending=False)


@st.cache_data
def toss_impact(matches: pd.DataFrame) -> pd.DataFrame:
    """Did toss winner also win the match? By decision."""
    df = matches[matches["winner"] != "No Result"].copy()
    grp = (
        df.groupby("toss_decision")
        .agg(
            matches=("id", "count"),
            toss_and_match_win=("toss_match_win", "sum"),
        )
        .reset_index()
    )
    grp["toss_win_pct"] = (grp["toss_and_match_win"] / grp["matches"] * 100).round(1)
    return grp


@st.cache_data
def season_scoring_trends(merged: pd.DataFrame) -> pd.DataFrame:
    """Average first-innings total per season."""
    inn1 = merged[(merged["innings"] == 1)].copy()
    per_match = inn1.groupby(["season", "match_id"])["total_runs"].sum().reset_index()
    return (
        per_match.groupby("season")["total_runs"]
        .mean()
        .reset_index()
        .rename(columns={"total_runs": "avg_first_innings_score"})
        .sort_values("season")
    )


# ── VENUE ANALYTICS ────────────────────────────────────────────────────────────

@st.cache_data
def venue_stats(matches: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Average scores and win rates by venue."""
    inn1 = merged[merged["innings"] == 1]
    avg_score = (
        inn1.groupby(["match_id", "venue"])["total_runs"]
        .sum()
        .reset_index()
        .groupby("venue")["total_runs"]
        .mean()
        .round(1)
        .reset_index()
        .rename(columns={"total_runs": "avg_first_innings"})
    )

    match_count = matches.groupby("venue").size().reset_index(name="matches_played")
    return avg_score.merge(match_count, on="venue").sort_values(
        "avg_first_innings", ascending=False
    )


@st.cache_data
def team_venue_wins(matches: pd.DataFrame) -> pd.DataFrame:
    """Wins matrix: team vs venue (for heatmap)."""
    df = matches[matches["winner"] != "No Result"].copy()
    pivot = (
        df.groupby(["winner", "venue"])
        .size()
        .reset_index(name="wins")
        .pivot(index="winner", columns="venue", values="wins")
        .fillna(0)
    )
    return pivot


# ── PLAYER COMPARISON ──────────────────────────────────────────────────────────

def compare_batsmen(deliveries: pd.DataFrame, p1: str, p2: str) -> pd.DataFrame:
    """Side-by-side batting stats for two players."""
    rows = []
    for player in [p1, p2]:
        df = deliveries[deliveries["batter"] == player]
        balls = len(df)
        runs = df["runs_off_bat"].sum()
        rows.append({
            "Player": player,
            "Runs": int(runs),
            "Balls": balls,
            "Strike Rate": round(runs / balls * 100, 2) if balls else 0,
            "Fours": int((df["runs_off_bat"] == 4).sum()),
            "Sixes": int((df["runs_off_bat"] == 6).sum()),
            "Innings": df["match_id"].nunique(),
        })
    return pd.DataFrame(rows).set_index("Player")


# ── BEST XI ────────────────────────────────────────────────────────────────────

def generate_best_xi(deliveries: pd.DataFrame, season: str = None) -> dict:
    """Generate a 'best XI' based on batting + bowling performance."""
    batsmen_df = top_batsmen(deliveries, n=50, season=season)
    bowlers_df = top_bowlers(deliveries, n=50, season=season)

    top_bat = batsmen_df.head(7)["batter"].tolist()
    top_bowl = bowlers_df.head(4)["bowler"].tolist()

    # Remove overlaps — all-rounders count for both
    xi = list(dict.fromkeys(top_bat + top_bowl))[:11]
    return {"best_xi": xi}
