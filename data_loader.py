"""
IPL Data Loader Module
Handles loading and initial validation of IPL datasets.
"""

import pandas as pd
import numpy as np
import os
import streamlit as st


@st.cache_data
def load_matches(filepath: str = "data/matches.csv") -> pd.DataFrame:
    """Load and validate matches dataset."""
    df = pd.read_csv(filepath)

    # Ensure correct dtypes
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "season" in df.columns:
        df["season"] = df["season"].astype(str)

    # Handle missing values (pandas 2.x CoW-safe)
    fills = {
        "result": "No Result", "result_margin": 0,
        "player_of_match": "Unknown", "venue": "Unknown Venue",
        "city": "Unknown", "winner": "No Result", "toss_decision": "Unknown",
    }
    for col, val in fills.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    return df


@st.cache_data
def load_deliveries(filepath: str = "data/deliveries.csv") -> pd.DataFrame:
    """Load and validate deliveries dataset."""
    df = pd.read_csv(filepath)

    # Ensure correct dtypes
    numeric_cols = [
        "wides", "noballs", "byes", "legbyes", "penalty",
        "runs_off_bat", "extras", "total_runs"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Handle missing values (CoW-safe)
    df["player_dismissed"] = df["player_dismissed"].fillna("")
    df["dismissal_kind"] = df["dismissal_kind"].fillna("")
    df["fielder"] = df["fielder"].fillna("")

    # Ensure total_runs column exists
    if "total_runs" not in df.columns:
        df["total_runs"] = df.get("runs_off_bat", 0) + df.get("extras", 0)

    return df


@st.cache_data
def load_all_data(matches_path: str = "data/matches.csv",
                  deliveries_path: str = "data/deliveries.csv"):
    """Load both datasets and return as tuple."""
    matches = load_matches(matches_path)
    deliveries = load_deliveries(deliveries_path)
    return matches, deliveries


def validate_data(matches: pd.DataFrame, deliveries: pd.DataFrame) -> dict:
    """Return a dict with basic validation stats."""
    return {
        "matches_shape": matches.shape,
        "deliveries_shape": deliveries.shape,
        "seasons": sorted(matches["season"].unique().tolist()),
        "teams": sorted(
            list(set(matches["team1"].unique()) | set(matches["team2"].unique()))
        ),
        "venues": sorted(matches["venue"].unique().tolist()),
        "match_ids_in_deliveries": deliveries["match_id"].nunique(),
    }
