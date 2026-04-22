"""
tests/test_pipeline.py
======================
Unit + integration tests for the IPL Analytics pipeline.

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_pipeline.py
"""

import sys
import os
import warnings
import unittest
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_all_data
from src.preprocessing import (
    normalize_team_names, preprocess_matches, preprocess_deliveries,
    merge_data, compute_innings_cumulative, get_clean_data,
)
from src.feature_engineering import (
    engineer_win_probability_features, get_wp_feature_columns,
)
from src.analysis import (
    top_batsmen, top_bowlers, team_win_percentage, toss_impact,
    season_scoring_trends, venue_stats, team_venue_wins,
    compare_batsmen, generate_best_xi, batsman_season_runs,
)
from src.model import MatchWinnerModel
from src.win_probability import WinProbabilityModel, create_match_state, extract_key_events


# ── Shared fixtures ────────────────────────────────────────────────────────────

_matches_raw = None
_deliveries_raw = None
_matches = None
_deliveries = None
_merged = None


def load_fixtures():
    global _matches_raw, _deliveries_raw, _matches, _deliveries, _merged
    if _matches is None:
        _matches_raw, _deliveries_raw = load_all_data()
        _matches, _deliveries, _merged = get_clean_data(_matches_raw, _deliveries_raw)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        load_fixtures()

    def test_matches_loaded(self):
        self.assertGreater(len(_matches_raw), 0)
        self.assertIn("id", _matches_raw.columns)
        self.assertIn("team1", _matches_raw.columns)

    def test_deliveries_loaded(self):
        self.assertGreater(len(_deliveries_raw), 0)
        self.assertIn("match_id", _deliveries_raw.columns)
        self.assertIn("batter", _deliveries_raw.columns)

    def test_no_critical_nulls(self):
        for col in ["team1", "team2", "season"]:
            self.assertEqual(_matches_raw[col].isna().sum(), 0,
                             f"Unexpected nulls in matches.{col}")

    def test_total_runs_nonneg(self):
        self.assertTrue((_deliveries_raw["total_runs"] >= 0).all())


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        load_fixtures()

    def test_team_name_normalisation(self):
        df = pd.DataFrame({"team": ["Delhi Daredevils", "Mumbai Indians",
                                     "Kings XI Punjab"]})
        out = normalize_team_names(df, ["team"])
        self.assertNotIn("Delhi Daredevils", out["team"].values)
        self.assertNotIn("Kings XI Punjab", out["team"].values)
        self.assertIn("Delhi Capitals", out["team"].values)
        self.assertIn("Punjab Kings", out["team"].values)

    def test_year_column_created(self):
        self.assertIn("year", _matches.columns)
        self.assertTrue((_matches["year"] >= 2008).all())

    def test_ball_number_column(self):
        self.assertIn("ball_number", _deliveries.columns)
        self.assertTrue((_deliveries["ball_number"] >= 0).all())

    def test_is_wicket_binary(self):
        self.assertIn("is_wicket", _deliveries.columns)
        self.assertTrue(_deliveries["is_wicket"].isin([0, 1]).all())

    def test_merge_shape(self):
        self.assertEqual(len(_merged), len(_deliveries))

    def test_cumulative_runs_nonneg(self):
        self.assertTrue((_merged["cumulative_runs"] >= 0).all())

    def test_cumulative_wickets_max_10(self):
        self.assertTrue((_merged["cumulative_wickets"] <= 10).all())


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        load_fixtures()
        self.wp_df = engineer_win_probability_features(_merged)

    def test_feature_columns_present(self):
        for col in get_wp_feature_columns():
            self.assertIn(col, self.wp_df.columns, f"Missing feature: {col}")

    def test_win_label_binary(self):
        self.assertTrue(self.wp_df["win_label"].isin([0, 1]).all())

    def test_balls_remaining_range(self):
        self.assertTrue((self.wp_df["balls_remaining"] >= 0).all())
        self.assertTrue((self.wp_df["balls_remaining"] <= 120).all())

    def test_runs_remaining_nonneg(self):
        self.assertTrue((self.wp_df["runs_remaining"] >= 0).all())

    def test_no_nans_in_features(self):
        for col in get_wp_feature_columns():
            n = self.wp_df[col].isna().sum()
            self.assertEqual(n, 0, f"{col} has {n} NaN values")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalysis(unittest.TestCase):

    def setUp(self):
        load_fixtures()

    def test_top_batsmen_returns_df(self):
        df = top_batsmen(_deliveries, n=10)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)
        self.assertIn("total_runs", df.columns)
        self.assertIn("strike_rate", df.columns)

    def test_top_batsmen_descending(self):
        df = top_batsmen(_deliveries, n=20)
        runs = df["total_runs"].tolist()
        self.assertEqual(runs, sorted(runs, reverse=True))

    def test_top_bowlers_returns_df(self):
        df = top_bowlers(_deliveries, n=10)
        self.assertEqual(len(df), 10)
        self.assertIn("wickets", df.columns)
        self.assertIn("economy", df.columns)

    def test_team_win_percentage(self):
        df = team_win_percentage(_matches)
        self.assertIn("win_pct", df.columns)
        self.assertTrue((df["win_pct"] >= 0).all())
        self.assertTrue((df["win_pct"] <= 100).all())

    def test_toss_impact_decisions(self):
        df = toss_impact(_matches)
        self.assertIn("bat", df["toss_decision"].values)
        self.assertIn("field", df["toss_decision"].values)

    def test_season_scoring_trend_sorted(self):
        df = season_scoring_trends(_merged)
        self.assertEqual(df["season"].tolist(), sorted(df["season"].tolist()))

    def test_venue_stats_avg_positive(self):
        df = venue_stats(_matches, _merged)
        self.assertTrue((df["avg_first_innings"] > 0).all())

    def test_team_venue_wins_pivot(self):
        pivot = team_venue_wins(_matches)
        self.assertIsInstance(pivot, pd.DataFrame)
        self.assertGreater(pivot.shape[0], 0)

    def test_compare_batsmen(self):
        top2 = top_batsmen(_deliveries, n=2)["batter"].tolist()
        df = compare_batsmen(_deliveries, top2[0], top2[1])
        self.assertEqual(len(df), 2)
        self.assertIn("Runs", df.columns)
        self.assertIn("Strike Rate", df.columns)

    def test_best_xi_count(self):
        xi = generate_best_xi(_deliveries)
        self.assertEqual(len(xi["best_xi"]), 11)

    def test_batsman_season_runs(self):
        # batsman_season_runs needs a frame with a 'season' column → use merged
        player = top_batsmen(_deliveries, n=1)["batter"].iloc[0]
        df = batsman_season_runs(_merged, player)
        self.assertIn("season", df.columns)
        self.assertIn("runs", df.columns)
        self.assertGreater(len(df), 0)

    def test_season_filter(self):
        # top_batsmen with season filter needs 'season' column → use merged
        season = _matches["season"].iloc[0]
        df = top_batsmen(_merged, n=5, season=season)
        self.assertLessEqual(len(df), 5)


# ══════════════════════════════════════════════════════════════════════════════
# ML MODELS
# ══════════════════════════════════════════════════════════════════════════════

class TestMatchWinnerModel(unittest.TestCase):

    def setUp(self):
        load_fixtures()
        self.model = MatchWinnerModel()
        self.model.train(_matches)

    def test_trained_flag(self):
        self.assertTrue(self.model.trained)

    def test_accuracy_in_range(self):
        self.assertGreater(self.model.accuracy, 30)
        self.assertLess(self.model.accuracy, 100)

    def test_prediction_sums_to_100(self):
        teams = _matches["team1"].unique()[:2].tolist()
        venues = _matches["venue"].unique()[:1].tolist()
        pred = self.model.predict(teams[0], teams[1], teams[0], "bat", venues[0])
        total = sum(pred.values())
        self.assertAlmostEqual(total, 100.0, places=0)

    def test_prediction_both_teams_in_result(self):
        teams = _matches["team1"].unique()[:2].tolist()
        venues = _matches["venue"].unique()[:1].tolist()
        pred = self.model.predict(teams[0], teams[1], teams[0], "field", venues[0])
        self.assertIn(teams[0], pred)
        self.assertIn(teams[1], pred)

    def test_save_and_reload(self, tmp_path="/tmp/test_match_model.pkl",
                             tmp_enc="/tmp/test_match_enc.pkl"):
        self.model.save(tmp_path, tmp_enc)
        m2 = MatchWinnerModel()
        m2.load(tmp_path, tmp_enc)
        self.assertTrue(m2.trained)
        teams = _matches["team1"].unique()[:2].tolist()
        venues = _matches["venue"].unique()[:1].tolist()
        pred = m2.predict(teams[0], teams[1], teams[0], "bat", venues[0])
        self.assertEqual(len(pred), 2)


class TestWinProbabilityModel(unittest.TestCase):

    def setUp(self):
        load_fixtures()
        self.model = WinProbabilityModel()
        self.model.train(_merged)

    def test_auc_above_threshold(self):
        self.assertGreater(self.model.auc, 0.65,
                           f"AUC too low: {self.model.auc}")

    def test_predict_match_range(self):
        mid = _merged["match_id"].unique()[0]
        state = create_match_state(_merged, mid)
        result = self.model.predict_match(state)
        if not result.empty:
            self.assertTrue((result["win_probability"] >= 0).all())
            self.assertTrue((result["win_probability"] <= 1).all())

    def test_create_match_state_columns(self):
        mid = _merged["match_id"].unique()[2]
        state = create_match_state(_merged, mid)
        for col in get_wp_feature_columns():
            self.assertIn(col, state.columns, f"Missing column: {col}")

    def test_extract_key_events(self):
        mid = _merged["match_id"].unique()[3]
        state = create_match_state(_merged, mid)
        events = extract_key_events(state)
        self.assertIsInstance(events, list)
        for ev in events:
            self.assertIn("ball", ev)
            self.assertIn("type", ev)
            self.assertIn(ev["type"], ["wicket", "boundary"])

    def test_save_and_reload(self, tmp_path="/tmp/test_wp_model.pkl"):
        self.model.save(tmp_path)
        m2 = WinProbabilityModel()
        m2.load(tmp_path)
        self.assertTrue(m2.trained)
        mid = _merged["match_id"].unique()[0]
        state = create_match_state(_merged, mid)
        result = m2.predict_match(state)
        self.assertFalse(result.empty)


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION — full pipeline smoke test
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):

    def test_full_pipeline_runs(self):
        """Smoke test: run the complete pipeline without errors."""
        load_fixtures()
        self.assertGreater(len(_matches), 0)
        self.assertGreater(len(_deliveries), 0)
        self.assertGreater(len(_merged), 0)

        bat = top_batsmen(_deliveries, n=5)
        self.assertEqual(len(bat), 5)

        bowl = top_bowlers(_deliveries, n=5)
        self.assertEqual(len(bowl), 5)

        m = MatchWinnerModel()
        acc = m.train(_matches)
        self.assertIsNotNone(acc)

        wp = WinProbabilityModel()
        auc = wp.train(_merged)
        self.assertGreater(auc, 0.5)

        mid = _merged["match_id"].unique()[10]
        state = create_match_state(_merged, mid)
        result = wp.predict_match(state)
        self.assertIn("win_probability", result.columns)


# ── Runner ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  IPL Dashboard — Test Suite")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.dirname(__file__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
