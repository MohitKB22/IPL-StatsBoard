"""
IPL Feature Engineering Module
Creates ML-ready features for win-probability and match-prediction models.
"""

import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def engineer_win_probability_features(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Build 2nd-innings ball-by-ball features for the win-probability model.

    Returns a dataframe with one row per delivery in 2nd innings,
    enriched with target, runs_remaining, RRR, CRR, etc.
    """
    df = merged[merged["innings"] == 2].copy()

    # --- Compute 1st-innings totals per match ---
    inn1 = (
        merged[merged["innings"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "target_runs"})
    )
    inn1["target"] = inn1["target_runs"] + 1  # need one more run to win

    df = df.merge(inn1[["match_id", "target"]], on="match_id", how="left")

    # Runs remaining after this delivery
    df["runs_remaining"] = (df["target"] - df["cumulative_runs"]).clip(lower=0)

    # Balls remaining after this delivery  (max 120 balls per innings)
    df["balls_remaining"] = (120 - df["ball_number"]).clip(lower=0)

    # Current Run Rate (per over, after this ball)
    balls_bowled = df["ball_number"] + 1
    df["current_run_rate"] = (df["cumulative_runs"] / (balls_bowled / 6)).replace(
        [np.inf, -np.inf], 0
    )

    # Required Run Rate (per over, from this ball onward)
    df["required_run_rate"] = np.where(
        df["balls_remaining"] > 0,
        (df["runs_remaining"] / (df["balls_remaining"] / 6)),
        np.inf,
    )
    df["required_run_rate"] = df["required_run_rate"].replace([np.inf, -np.inf], 36)

    # Win label: batting team == winner
    df["win_label"] = (df["batting_team"] == df["winner"]).astype(int)

    # Drop rows where target is missing
    df = df.dropna(subset=["target"])

    return df


@st.cache_data
def engineer_match_prediction_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Features for pre-match winner prediction.
    Returns one row per match with encoded categorical features.
    """
    df = matches.copy()
    df = df[df["winner"].notna() & (df["winner"] != "No Result")].copy()

    # Label: did team1 win?
    df["team1_win"] = (df["winner"] == df["team1"]).astype(int)

    cat_cols = ["team1", "team2", "toss_winner", "venue", "toss_decision"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df


def get_wp_feature_columns():
    """Return the feature column names used by the win-probability model."""
    return [
        "cumulative_runs",
        "cumulative_wickets",
        "balls_remaining",
        "runs_remaining",
        "current_run_rate",
        "required_run_rate",
    ]


def get_match_feature_columns():
    """Return feature column names for the match-winner model."""
    return ["team1_enc", "team2_enc", "toss_winner_enc", "venue_enc", "toss_decision_enc"]
