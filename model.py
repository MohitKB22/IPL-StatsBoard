"""
IPL ML Models Module
Match-winner prediction (pre-match RandomForest).
"""

import pandas as pd
import numpy as np
import pickle
import os
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "data/match_winner_model.pkl"
ENCODER_PATH = "data/match_winner_encoders.pkl"


class MatchWinnerModel:
    """Pre-match winner prediction using RandomForest."""

    CAT_COLS = ["team1", "team2", "toss_winner", "venue", "toss_decision"]

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        self.encoders: dict[str, LabelEncoder] = {}
        self.trained = False
        self.accuracy = None

    def _encode(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        out = df.copy()
        for col in self.CAT_COLS:
            if col not in out.columns:
                continue
            enc_col = f"{col}_enc"
            if fit:
                le = LabelEncoder()
                out[enc_col] = le.fit_transform(out[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders[col]
                # Handle unseen labels gracefully
                known = set(le.classes_)
                out[col] = out[col].apply(lambda x: x if x in known else le.classes_[0])
                out[enc_col] = le.transform(out[col].astype(str))
        return out

    def train(self, matches: pd.DataFrame) -> float:
        df = matches[matches["winner"] != "No Result"].copy()
        df["win_label"] = (df["winner"] == df["team1"]).astype(int)

        df = self._encode(df, fit=True)
        feature_cols = [f"{c}_enc" for c in self.CAT_COLS if c in matches.columns]
        X = df[feature_cols]
        y = df["win_label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        self.accuracy = round(accuracy_score(y_test, preds) * 100, 2)
        self.trained = True
        return self.accuracy

    def predict(self, team1: str, team2: str, toss_winner: str,
                toss_decision: str, venue: str) -> dict:
        """Return win probabilities for both teams."""
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        row = pd.DataFrame([{
            "team1": team1, "team2": team2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "venue": venue,
        }])
        row = self._encode(row, fit=False)
        feature_cols = [f"{c}_enc" for c in self.CAT_COLS if c in row.columns]
        proba = self.model.predict_proba(row[feature_cols])[0]
        classes = self.model.classes_
        team1_prob = proba[list(classes).index(1)] if 1 in classes else 0.5
        return {
            team1: round(team1_prob * 100, 1),
            team2: round((1 - team1_prob) * 100, 1),
        }

    def save(self, model_path: str = MODEL_PATH, enc_path: str = ENCODER_PATH):
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(enc_path, "wb") as f:
            pickle.dump(self.encoders, f)

    def load(self, model_path: str = MODEL_PATH, enc_path: str = ENCODER_PATH):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(enc_path, "rb") as f:
            self.encoders = pickle.load(f)
        self.trained = True


@st.cache_resource
def get_trained_match_model(matches: pd.DataFrame) -> MatchWinnerModel:
    """Load from disk if cached, otherwise train fresh."""
    m = MatchWinnerModel()
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        try:
            m.load()
            return m
        except Exception:
            pass
    m.train(matches)
    try:
        m.save()
    except Exception:
        pass
    return m
