"""
IPL Preprocessing Module
Merges, cleans and standardizes IPL data.
"""

import pandas as pd
import numpy as np
import streamlit as st

# Team name normalization map — covers all franchise renames across IPL history
TEAM_NAME_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Pune Warriors": "Rising Pune Supergiant",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}


def normalize_team_names(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Replace old franchise names with current ones."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAP)
    return df


@st.cache_data
def preprocess_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the matches dataframe."""
    df = matches.copy()

    team_cols = ["team1", "team2", "toss_winner", "winner"]
    df = normalize_team_names(df, team_cols)

    # Extract year from season (handles both "2023" and "2023/24")
    df["year"] = df["season"].str[:4].astype(int)

    # Binary toss-winner flag: did the toss winner also win the match?
    df["toss_match_win"] = (df["toss_winner"] == df["winner"]).astype(int)

    return df


@st.cache_data
def preprocess_deliveries(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Clean the deliveries dataframe and derive basic columns."""
    df = deliveries.copy()

    team_cols = ["batting_team", "bowling_team"]
    df = normalize_team_names(df, team_cols)

    # Ensure total_runs
    if "total_runs" not in df.columns:
        df["total_runs"] = df["runs_off_bat"] + df["extras"]

    # Ball number within innings (0-indexed, 0–119 for 20 overs)
    df["ball_number"] = (df["over"] * 6 + df["ball"]).astype(int)

    # Is wicket?
    df["is_wicket"] = (df["player_dismissed"].notna() & (df["player_dismissed"] != "")).astype(int)

    return df


@st.cache_data
def merge_data(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Merge deliveries with match metadata.
    Returns the enriched deliveries frame.
    """
    match_meta = matches[[
        "id", "season", "year", "date", "venue", "city",
        "team1", "team2", "toss_winner", "toss_decision", "winner",
        "result", "result_margin", "player_of_match"
    ]].copy()

    merged = deliveries.merge(match_meta, left_on="match_id", right_on="id", how="left")
    return merged


@st.cache_data
def compute_innings_cumulative(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Add cumulative run and wicket columns within each innings.
    Groups by match_id + innings.
    """
    df = deliveries.copy()
    df = df.sort_values(["match_id", "innings", "ball_number"])

    grp = df.groupby(["match_id", "innings"])
    df["cumulative_runs"] = grp["total_runs"].cumsum()
    df["cumulative_wickets"] = grp["is_wicket"].cumsum()

    return df


def get_clean_data(matches_raw: pd.DataFrame, deliveries_raw: pd.DataFrame):
    """Full preprocessing pipeline. Returns (matches, deliveries, merged)."""
    matches = preprocess_matches(matches_raw)
    deliveries = preprocess_deliveries(deliveries_raw)
    merged = merge_data(matches, deliveries)
    merged = compute_innings_cumulative(merged)
    return matches, deliveries, merged
