"""
IPL Win Probability Module
Train and serve a ball-by-ball win probability model (2nd innings).
"""

import pandas as pd
import numpy as np
import pickle
import os
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.feature_engineering import engineer_win_probability_features, get_wp_feature_columns

WP_MODEL_PATH = "data/win_prob_model.pkl"

FEATURE_COLS = get_wp_feature_columns()


class WinProbabilityModel:
    """Ball-by-ball win probability for the 2nd innings."""

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42
        )
        self.trained = False
        self.auc = None

    def train(self, merged: pd.DataFrame) -> float:
        """Train on all 2nd-innings balls. Returns AUC score."""
        df = engineer_win_probability_features(merged)
        df = df.dropna(subset=FEATURE_COLS + ["win_label"])

        X = df[FEATURE_COLS]
        y = df["win_label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        self.model.fit(X_train, y_train)
        proba = self.model.predict_proba(X_test)[:, 1]
        self.auc = round(roc_auc_score(y_test, proba), 4)
        self.trained = True
        return self.auc

    def predict_match(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a single match's 2nd-innings delivery data (already feature-engineered),
        return a dataframe with columns: ball_number, win_probability.
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")

        df = match_df.dropna(subset=FEATURE_COLS).copy()
        if df.empty:
            return pd.DataFrame(columns=["ball_number", "win_probability"])

        df["win_probability"] = self.model.predict_proba(df[FEATURE_COLS])[:, 1]
        return df[["ball_number", "win_probability"]].reset_index(drop=True)

    def save(self, path: str = WP_MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str = WP_MODEL_PATH):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.trained = True


@st.cache_resource
def get_trained_wp_model(merged: pd.DataFrame) -> WinProbabilityModel:
    """Load from disk if available, else train."""
    m = WinProbabilityModel()
    if os.path.exists(WP_MODEL_PATH):
        try:
            m.load()
            return m
        except Exception:
            pass
    m.train(merged)
    try:
        m.save()
    except Exception:
        pass
    return m


# ── MATCH STATE BUILDER ────────────────────────────────────────────────────────

def create_match_state(merged: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """
    Build feature-engineered 2nd-innings frame for a specific match.
    Mirrors engineer_win_probability_features but for a single match.
    """
    match = merged[merged["match_id"] == match_id].copy()

    # First innings total
    inn1_total = match[match["innings"] == 1]["total_runs"].sum()
    target = inn1_total + 1

    inn2 = match[match["innings"] == 2].copy()
    if inn2.empty:
        return pd.DataFrame()

    inn2 = inn2.sort_values("ball_number")
    inn2["target"] = target
    inn2["runs_remaining"] = (target - inn2["cumulative_runs"]).clip(lower=0)
    inn2["balls_remaining"] = (120 - inn2["ball_number"]).clip(lower=0)

    balls_bowled = inn2["ball_number"] + 1
    inn2["current_run_rate"] = (inn2["cumulative_runs"] / (balls_bowled / 6)).replace(
        [np.inf, -np.inf], 0
    )
    inn2["required_run_rate"] = np.where(
        inn2["balls_remaining"] > 0,
        inn2["runs_remaining"] / (inn2["balls_remaining"] / 6),
        36,
    )
    inn2["required_run_rate"] = inn2["required_run_rate"].clip(upper=36)

    return inn2


# ── EVENT EXTRACTION ──────────────────────────────────────────────────────────

def extract_key_events(match_df: pd.DataFrame) -> list:
    """Return a list of dicts for wickets and boundaries in the 2nd innings."""
    events = []
    for _, row in match_df.iterrows():
        b = int(row["ball_number"])
        if row.get("is_wicket", 0) == 1:
            events.append({
                "ball": b,
                "type": "wicket",
                "desc": f"Wicket! {row.get('player_dismissed', '')} ({row.get('dismissal_kind', '')})",
            })
        elif row.get("runs_off_bat", 0) in [4, 6]:
            kind = "SIX" if row["runs_off_bat"] == 6 else "FOUR"
            events.append({
                "ball": b,
                "type": "boundary",
                "desc": f"{kind}! {row.get('batter', '')}",
            })
    return events
