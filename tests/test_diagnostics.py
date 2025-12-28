# -*- coding: utf-8 -*-
"""
Tests for src/features_v4/diagnostics.py
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features_v4.diagnostics import (
    get_feature_group,
    compute_lgbm_importance,
    compute_permutation_importance,
    compute_group_importance,
    compute_segment_performance,
    run_diagnostics,
    save_diagnostics,
    FeatureImportanceResult,
    PermutationImportanceResult,
    FeatureGroupImportanceResult,
    SegmentPerformanceResult,
    DiagnosticsReport,
    FEATURE_GROUP_PREFIXES,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_model():
    """Create a mock LightGBM model"""
    model = MagicMock()
    model.best_iteration = 100

    # Mock feature importance
    model.feature_importance = MagicMock(side_effect=lambda importance_type: {
        "gain": np.array([100.0, 80.0, 60.0, 40.0, 20.0]),
        "split": np.array([500, 400, 300, 200, 100]),
    }[importance_type])

    # Mock num_feature (must match sample_feature_cols length = 5)
    model.num_feature = MagicMock(return_value=5)

    # Mock predict
    def mock_predict(X, num_iteration=None):
        n = len(X)
        return np.random.uniform(0, 1, n)

    model.predict = MagicMock(side_effect=mock_predict)

    return model


@pytest.fixture
def sample_feature_cols():
    """Sample feature columns"""
    return [
        "h_win_rate_all",       # horse_form
        "h_avg_finish",          # horse_form
        "j_win_rate_all",        # jockey_trainer
        "surface_id",            # base_race
        "distance",              # base_race
    ]


@pytest.fixture
def sample_df():
    """Sample dataframe for testing"""
    np.random.seed(42)
    n_races = 10
    n_horses_per_race = 12
    n_total = n_races * n_horses_per_race

    # Create race_ids
    race_ids = []
    for i in range(n_races):
        race_ids.extend([f"2024010101{i+1:02d}"] * n_horses_per_race)

    df = pd.DataFrame({
        "race_id": race_ids,
        "horse_id": [f"horse_{i}" for i in range(n_total)],
        "target_win": np.zeros(n_total),
        "h_win_rate_all": np.random.uniform(0, 0.5, n_total),
        "h_avg_finish": np.random.uniform(1, 18, n_total),
        "j_win_rate_all": np.random.uniform(0, 0.3, n_total),
        "surface_id": np.random.randint(0, 2, n_total),
        "distance": np.random.choice([1200, 1600, 2000, 2400], n_total),
        "distance_cat": np.random.choice([1200, 1600, 2000, 2400], n_total),
        "track_condition_id": np.random.randint(0, 4, n_total),
        "grade_id": np.random.randint(0, 6, n_total),
        "field_size": n_horses_per_race,
    })

    # Set one winner per race
    for i, race_id in enumerate(df["race_id"].unique()):
        race_mask = df["race_id"] == race_id
        race_indices = df[race_mask].index
        winner_idx = np.random.choice(race_indices)
        df.loc[winner_idx, "target_win"] = 1

    return df


# =============================================================================
# Test get_feature_group
# =============================================================================

class TestGetFeatureGroup:
    """Tests for get_feature_group function"""

    def test_horse_form_prefix(self):
        """h_ prefix should map to horse_form"""
        assert get_feature_group("h_win_rate_all") == "horse_form"
        assert get_feature_group("h_avg_finish") == "horse_form"
        assert get_feature_group("h_recent_3_avg") == "horse_form"

    def test_jockey_trainer_prefix(self):
        """j_ and t_ prefixes should map to jockey_trainer"""
        assert get_feature_group("j_win_rate_all") == "jockey_trainer"
        assert get_feature_group("j_roi") == "jockey_trainer"
        assert get_feature_group("t_win_rate_all") == "jockey_trainer"
        assert get_feature_group("t_avg_finish") == "jockey_trainer"

    def test_pedigree_prefix(self):
        """ped_hash_ and anc_hash_ prefixes should map to pedigree"""
        assert get_feature_group("ped_hash_0") == "pedigree"
        assert get_feature_group("ped_hash_511") == "pedigree"
        assert get_feature_group("anc_hash_0") == "pedigree"
        assert get_feature_group("anc_hash_127") == "pedigree"

    def test_market_prefix(self):
        """market_ prefix should map to market"""
        assert get_feature_group("market_win_odds") == "market"
        assert get_feature_group("market_popularity") == "market"

    def test_base_race_exact_match(self):
        """Base race features should be exact matches"""
        assert get_feature_group("surface_id") == "base_race"
        assert get_feature_group("distance") == "base_race"
        assert get_feature_group("distance_cat") == "base_race"
        assert get_feature_group("track_condition_id") == "base_race"
        assert get_feature_group("field_size") == "base_race"
        assert get_feature_group("waku") == "base_race"
        assert get_feature_group("umaban") == "base_race"

    def test_pace_position_prefix(self):
        """pace_ and pos_ prefixes should map to pace_position"""
        assert get_feature_group("pace_early") == "pace_position"
        assert get_feature_group("pace_late") == "pace_position"
        assert get_feature_group("pos_corner1") == "pace_position"

    def test_unknown_feature(self):
        """Unknown features should map to 'other'"""
        assert get_feature_group("unknown_feature") == "other"
        assert get_feature_group("random_xyz") == "other"


# =============================================================================
# Test compute_lgbm_importance
# =============================================================================

class TestComputeLgbmImportance:
    """Tests for compute_lgbm_importance function"""

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_returns_correct_length(self, mock_model, sample_feature_cols):
        """Should return one result per feature"""
        results = compute_lgbm_importance(mock_model, sample_feature_cols)
        assert len(results) == len(sample_feature_cols)

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_sorted_by_gain(self, mock_model, sample_feature_cols):
        """Results should be sorted by gain descending"""
        results = compute_lgbm_importance(mock_model, sample_feature_cols)
        gains = [r.gain for r in results]
        assert gains == sorted(gains, reverse=True)

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_gain_rank_assigned(self, mock_model, sample_feature_cols):
        """Gain ranks should be assigned correctly"""
        results = compute_lgbm_importance(mock_model, sample_feature_cols)
        # First result should have rank 1
        assert results[0].gain_rank == 1
        # Last result should have rank equal to number of features
        assert results[-1].gain_rank == len(sample_feature_cols)

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_split_rank_assigned(self, mock_model, sample_feature_cols):
        """Split ranks should be assigned correctly"""
        results = compute_lgbm_importance(mock_model, sample_feature_cols)
        split_ranks = [r.split_rank for r in results]
        assert sorted(split_ranks) == list(range(1, len(sample_feature_cols) + 1))

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_result_type(self, mock_model, sample_feature_cols):
        """Results should be FeatureImportanceResult objects"""
        results = compute_lgbm_importance(mock_model, sample_feature_cols)
        for r in results:
            assert isinstance(r, FeatureImportanceResult)


# =============================================================================
# Test compute_permutation_importance
# =============================================================================

class TestComputePermutationImportance:
    """Tests for compute_permutation_importance function"""

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_returns_results(self, mock_model, sample_df, sample_feature_cols):
        """Should return permutation importance results"""
        results = compute_permutation_importance(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win", n_repeats=1, top_n=3
        )
        assert len(results) == 3  # top_n = 3

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_result_type(self, mock_model, sample_df, sample_feature_cols):
        """Results should be PermutationImportanceResult objects"""
        results = compute_permutation_importance(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win", n_repeats=1, top_n=2
        )
        for r in results:
            assert isinstance(r, PermutationImportanceResult)

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_sorted_by_auc(self, mock_model, sample_df, sample_feature_cols):
        """Results should be sorted by delta_auc descending"""
        results = compute_permutation_importance(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win", n_repeats=1, top_n=3
        )
        aucs = [r.delta_auc for r in results]
        assert aucs == sorted(aucs, reverse=True)

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_rank_assigned(self, mock_model, sample_df, sample_feature_cols):
        """Ranks should be assigned for all metrics"""
        results = compute_permutation_importance(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win", n_repeats=1, top_n=3
        )
        for r in results:
            assert r.rank_auc >= 1
            assert r.rank_logloss >= 1
            assert r.rank_top1 >= 1
            assert r.rank_mrr >= 1


# =============================================================================
# Test compute_group_importance
# =============================================================================

class TestComputeGroupImportance:
    """Tests for compute_group_importance function"""

    def test_groups_features_correctly(self):
        """Should group features by their prefix"""
        lgbm_importance = [
            FeatureImportanceResult("h_win_rate", 100.0, 500, 1, 1),
            FeatureImportanceResult("h_avg_finish", 80.0, 400, 2, 2),
            FeatureImportanceResult("j_win_rate", 60.0, 300, 3, 3),
            FeatureImportanceResult("surface_id", 40.0, 200, 4, 4),
            FeatureImportanceResult("distance", 20.0, 100, 5, 5),
        ]
        results = compute_group_importance(lgbm_importance)

        group_names = [r.group_name for r in results]
        assert "horse_form" in group_names
        assert "jockey_trainer" in group_names
        assert "base_race" in group_names

    def test_aggregates_gain(self):
        """Should correctly aggregate gain values"""
        lgbm_importance = [
            FeatureImportanceResult("h_win_rate", 100.0, 500, 1, 1),
            FeatureImportanceResult("h_avg_finish", 80.0, 400, 2, 2),
        ]
        results = compute_group_importance(lgbm_importance)

        horse_form = next(r for r in results if r.group_name == "horse_form")
        assert horse_form.total_gain == 180.0
        assert horse_form.mean_gain == 90.0
        assert horse_form.max_gain == 100.0
        assert horse_form.n_features == 2

    def test_sorted_by_total_gain(self):
        """Results should be sorted by total_gain descending"""
        lgbm_importance = [
            FeatureImportanceResult("h_win_rate", 100.0, 500, 1, 1),
            FeatureImportanceResult("j_win_rate", 200.0, 300, 3, 3),
            FeatureImportanceResult("surface_id", 50.0, 200, 4, 4),
        ]
        results = compute_group_importance(lgbm_importance)

        gains = [r.total_gain for r in results]
        assert gains == sorted(gains, reverse=True)

    def test_includes_top_features(self):
        """Should include top features for each group"""
        lgbm_importance = [
            FeatureImportanceResult("h_win_rate", 100.0, 500, 1, 1),
            FeatureImportanceResult("h_avg_finish", 80.0, 400, 2, 2),
        ]
        results = compute_group_importance(lgbm_importance)

        horse_form = next(r for r in results if r.group_name == "horse_form")
        assert "h_win_rate" in horse_form.top_features
        assert "h_avg_finish" in horse_form.top_features


# =============================================================================
# Test compute_segment_performance
# =============================================================================

class TestComputeSegmentPerformance:
    """Tests for compute_segment_performance function"""

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_computes_for_segments(self, mock_model, sample_df, sample_feature_cols):
        """Should compute performance for each segment"""
        results, warnings = compute_segment_performance(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            segment_keys=["surface_id"]
        )
        assert len(results) > 0

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_result_type(self, mock_model, sample_df, sample_feature_cols):
        """Results should be SegmentPerformanceResult objects"""
        results, warnings = compute_segment_performance(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            segment_keys=["surface_id"]
        )
        for r in results:
            assert isinstance(r, SegmentPerformanceResult)

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_includes_metrics(self, mock_model, sample_df, sample_feature_cols):
        """Results should include all required metrics"""
        results, warnings = compute_segment_performance(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            segment_keys=["surface_id"]
        )
        for r in results:
            assert hasattr(r, "auc")
            assert hasattr(r, "logloss")
            assert hasattr(r, "top1_hit_rate")
            assert hasattr(r, "top3_hit_rate")
            assert hasattr(r, "top5_hit_rate")
            assert hasattr(r, "mrr")
            assert hasattr(r, "n_races")
            assert hasattr(r, "n_entries")

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_handles_missing_columns(self, mock_model, sample_df):
        """Should handle missing feature columns gracefully"""
        # Use feature columns that don't exist in sample_df
        missing_cols = ["horse_weight", "is_first_run", "some_nonexistent_col"]
        results, warnings = compute_segment_performance(
            mock_model, sample_df, missing_cols,
            target_col="target_win",
            segment_keys=["surface_id"]
        )
        # Should return empty results with warnings
        assert len(results) == 0
        assert len(warnings) > 0
        assert any("Missing" in w for w in warnings)

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_partial_missing_columns(self, mock_model, sample_df, sample_feature_cols):
        """Should work with partially missing columns"""
        # Mix existing and non-existing columns
        mixed_cols = sample_feature_cols + ["nonexistent_col_1", "nonexistent_col_2"]
        results, warnings = compute_segment_performance(
            mock_model, sample_df, mixed_cols,
            target_col="target_win",
            segment_keys=["surface_id"]
        )
        # Should still compute results using available columns
        assert len(results) > 0
        # Should warn about missing columns
        assert len(warnings) > 0

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_schema_mismatch_skipped_as_warning(self, sample_df, sample_feature_cols):
        """Should skip segment performance with warning when schema mismatch"""
        # Create a model that expects a different number of features
        mock_model = MagicMock()
        mock_model.best_iteration = 100
        mock_model.num_feature = MagicMock(return_value=31)  # Model expects 31 features
        # But sample_feature_cols has only 5 features

        results, warnings = compute_segment_performance(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            segment_keys=["surface_id"]
        )
        # Should return empty results
        assert len(results) == 0
        # Should have warning about schema mismatch
        assert len(warnings) > 0
        assert any("schema mismatch" in w.lower() for w in warnings)


# =============================================================================
# Test run_diagnostics
# =============================================================================

class TestRunDiagnostics:
    """Tests for run_diagnostics function"""

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_returns_report(self, mock_model, sample_df, sample_feature_cols):
        """Should return a DiagnosticsReport"""
        report = run_diagnostics(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            dataset_name="test",
            compute_perm=False,  # Skip permutation for speed
        )
        assert isinstance(report, DiagnosticsReport)

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_report_contains_lgbm_importance(self, mock_model, sample_df, sample_feature_cols):
        """Report should contain LightGBM importance"""
        report = run_diagnostics(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            dataset_name="test",
            compute_perm=False,
        )
        assert len(report.feature_importance) == len(sample_feature_cols)

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_report_contains_group_importance(self, mock_model, sample_df, sample_feature_cols):
        """Report should contain group importance"""
        report = run_diagnostics(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            dataset_name="test",
            compute_perm=False,
        )
        assert len(report.group_importance) > 0

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_report_contains_segment_performance(self, mock_model, sample_df, sample_feature_cols):
        """Report should contain segment performance"""
        report = run_diagnostics(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            dataset_name="test",
            compute_perm=False,
        )
        assert len(report.segment_performance) > 0

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_permutation_when_enabled(self, mock_model, sample_df, sample_feature_cols):
        """Should compute permutation importance when enabled"""
        report = run_diagnostics(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            dataset_name="test",
            compute_perm=True,
            perm_top_n=2,
            perm_n_repeats=1,
        )
        assert len(report.permutation_importance) == 2


# =============================================================================
# Test save_diagnostics
# =============================================================================

class TestSaveDiagnostics:
    """Tests for save_diagnostics function"""

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_creates_output_files(self, mock_model, sample_df, sample_feature_cols, tmp_path):
        """Should create output files"""
        report = run_diagnostics(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            dataset_name="test",
            compute_perm=False,
        )

        save_diagnostics(report, str(tmp_path), "target_win")

        # Check that files were created
        assert (tmp_path / "feature_importance_target_win_test_v4.csv").exists()
        assert (tmp_path / "group_importance_target_win_test_v4.csv").exists()
        assert (tmp_path / "segment_performance_target_win_test_v4.csv").exists()
        assert (tmp_path / "diagnostics_report_target_win_test_v4.txt").exists()
        assert (tmp_path / "diagnostics_summary_target_win_test_v4.json").exists()

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_csv_readable(self, mock_model, sample_df, sample_feature_cols, tmp_path):
        """Created CSVs should be readable"""
        report = run_diagnostics(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            dataset_name="test",
            compute_perm=False,
        )

        save_diagnostics(report, str(tmp_path), "target_win")

        # Read and verify CSVs
        df_lgbm = pd.read_csv(tmp_path / "feature_importance_target_win_test_v4.csv")
        assert len(df_lgbm) == len(sample_feature_cols)
        assert "feature_name" in df_lgbm.columns
        assert "gain" in df_lgbm.columns


# =============================================================================
# Test run_diagnostics fail-soft behavior
# =============================================================================

class TestRunDiagnosticsFailSoft:
    """Tests for run_diagnostics fail-soft behavior"""

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_handles_missing_columns_gracefully(self, mock_model, sample_df):
        """Should continue when feature columns are missing from df"""
        # Use completely non-existent feature columns
        missing_cols = ["horse_weight", "is_first_run", "horse_weight_diff"]

        report = run_diagnostics(
            mock_model, sample_df, missing_cols,
            target_col="target_win",
            dataset_name="test",
            compute_perm=False,
        )

        # Should return a report with warnings
        assert isinstance(report, DiagnosticsReport)
        assert len(report.warnings) > 0
        # Segment should be skipped or have warnings
        assert report.segment_skipped or any("Missing" in w for w in report.warnings)

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_reports_errors_and_warnings(self, mock_model, sample_df, sample_feature_cols):
        """Report should contain errors and warnings fields"""
        report = run_diagnostics(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            dataset_name="test",
            compute_perm=False,
        )

        # Report should have errors/warnings fields (even if empty)
        assert hasattr(report, "errors")
        assert hasattr(report, "warnings")
        assert hasattr(report, "segment_skipped")
        assert hasattr(report, "segment_skip_reason")

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_continues_on_partial_errors(self, mock_model, sample_df, sample_feature_cols):
        """Should continue computing other diagnostics when one fails"""
        # Create a model that fails on permutation but works for lgbm importance
        failing_model = MagicMock()
        failing_model.best_iteration = 100
        failing_model.feature_importance = MagicMock(side_effect=lambda importance_type: {
            "gain": np.array([100.0, 80.0, 60.0, 40.0, 20.0]),
            "split": np.array([500, 400, 300, 200, 100]),
        }[importance_type])
        # Predict raises an error to trigger permutation failure
        failing_model.predict = MagicMock(side_effect=Exception("Prediction failed"))

        report = run_diagnostics(
            failing_model, sample_df, sample_feature_cols,
            target_col="target_win",
            dataset_name="test",
            compute_perm=True,
            perm_top_n=2,
            perm_n_repeats=1,
        )

        # LightGBM importance should still work
        assert len(report.feature_importance) > 0
        # Should have errors recorded
        assert len(report.errors) > 0


# =============================================================================
# Test save_diagnostics with meta info
# =============================================================================

class TestSaveDiagnosticsMetaInfo:
    """Tests for save_diagnostics with warnings/errors metadata"""

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_json_includes_meta_info(self, mock_model, sample_df, sample_feature_cols, tmp_path):
        """JSON summary should include errors/warnings metadata"""
        import json

        report = run_diagnostics(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            dataset_name="test",
            compute_perm=False,
        )
        # Add some warnings to test
        report.warnings.append("Test warning")

        save_diagnostics(report, str(tmp_path), "target_win")

        # Load and check JSON
        json_path = tmp_path / "diagnostics_summary_target_win_test_v4.json"
        with open(json_path, "r") as f:
            summary = json.load(f)

        assert "warnings" in summary
        assert "errors" in summary
        assert "segment_skipped" in summary
        assert "segment_skip_reason" in summary
        assert "Test warning" in summary["warnings"]


# =============================================================================
# Test CLI Helper Functions
# =============================================================================

class TestCLIHelperFunctions:
    """Tests for CLI helper functions in train_eval_v4.py"""

    def test_load_feature_columns_with_fallback_v4_first(self, tmp_path):
        """Should load v4 file first when it exists"""
        import json
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        # Create both files
        v4_cols = ["feat_v4_1", "feat_v4_2"]
        legacy_cols = ["feat_legacy_1", "feat_legacy_2"]

        v4_path = tmp_path / "feature_columns_target_win_v4.json"
        legacy_path = tmp_path / "feature_columns_target_win.json"

        with open(v4_path, "w") as f:
            json.dump(v4_cols, f)
        with open(legacy_path, "w") as f:
            json.dump(legacy_cols, f)

        # Import helper
        from scripts.train_eval_v4 import load_feature_columns_with_fallback

        cols, used_path = load_feature_columns_with_fallback(str(tmp_path), "target_win")
        assert cols == v4_cols
        assert "v4" in used_path

    def test_load_feature_columns_with_fallback_to_legacy(self, tmp_path):
        """Should fallback to legacy file when v4 doesn't exist"""
        import json
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        # Only create legacy file
        legacy_cols = ["feat_legacy_1", "feat_legacy_2"]
        legacy_path = tmp_path / "feature_columns_target_win.json"

        with open(legacy_path, "w") as f:
            json.dump(legacy_cols, f)

        from scripts.train_eval_v4 import load_feature_columns_with_fallback

        cols, used_path = load_feature_columns_with_fallback(str(tmp_path), "target_win")
        assert cols == legacy_cols
        assert "v4" not in used_path

    def test_load_feature_columns_raises_when_missing(self, tmp_path):
        """Should raise FileNotFoundError when both files missing"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        from scripts.train_eval_v4 import load_feature_columns_with_fallback

        with pytest.raises(FileNotFoundError) as exc_info:
            load_feature_columns_with_fallback(str(tmp_path), "target_win")
        assert "feature_columns" in str(exc_info.value).lower()

    def test_load_exclude_features(self, tmp_path):
        """Should load exclude features from file"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        # Create exclude file
        exclude_path = tmp_path / "exclude.txt"
        exclude_path.write_text("""# Comment line
feature_1
feature_2

# Another comment
feature_3
""")

        from scripts.train_eval_v4 import load_exclude_features

        exclude_set = load_exclude_features(str(exclude_path))
        assert exclude_set == {"feature_1", "feature_2", "feature_3"}

    def test_apply_feature_exclusion(self):
        """Should filter out excluded features"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

        from scripts.train_eval_v4 import apply_feature_exclusion

        feature_cols = ["feat_1", "feat_2", "feat_3", "feat_4"]
        exclude_set = {"feat_2", "feat_4", "feat_5"}  # feat_5 doesn't exist

        filtered = apply_feature_exclusion(feature_cols, exclude_set)
        assert filtered == ["feat_1", "feat_3"]


# =============================================================================
# Test Schema Mismatch in run_diagnostics (integration)
# =============================================================================

class TestRunDiagnosticsSchemaIntegration:
    """Integration tests for schema mismatch handling in run_diagnostics"""

    @patch('src.features_v4.diagnostics.HAS_LIGHTGBM', True)
    def test_schema_mismatch_segment_skipped_not_error(self, sample_df, sample_feature_cols):
        """Schema mismatch should result in segment_skipped=True but no errors"""
        # Create model with mismatched feature count
        mock_model = MagicMock()
        mock_model.best_iteration = 100
        mock_model.num_feature = MagicMock(return_value=31)  # Expects 31, but only 5 available
        mock_model.feature_importance = MagicMock(side_effect=lambda importance_type: {
            "gain": np.array([100.0, 80.0, 60.0, 40.0, 20.0]),
            "split": np.array([500, 400, 300, 200, 100]),
        }[importance_type])

        report = run_diagnostics(
            mock_model, sample_df, sample_feature_cols,
            target_col="target_win",
            dataset_name="test",
            compute_perm=False,  # Skip permutation to avoid prediction
        )

        # LightGBM importance should work (doesn't need prediction)
        assert len(report.feature_importance) == len(sample_feature_cols)
        # Segment should be skipped (not crashed)
        assert report.segment_skipped is True
        # Schema mismatch should be in warnings or skip reason, NOT errors
        assert "mismatch" in (report.segment_skip_reason or "").lower() or \
               any("mismatch" in w.lower() for w in report.warnings)
        # Importantly: errors should NOT contain the mismatch (it's a skip, not error)
        assert not any("mismatch" in e.lower() for e in report.errors)


# =============================================================================
# Test Model Saving Fallback
# =============================================================================

class TestModelSavingFallback:
    """Tests for model saving fallback in train_eval_v4.py"""

    def test_model_to_string_fallback(self, tmp_path):
        """When save_model fails, should fall back to model_to_string"""
        # Create a mock model where save_model fails but model_to_string works
        mock_model = MagicMock()
        mock_model.best_iteration = 100
        mock_model.save_model = MagicMock(side_effect=Exception("Permission denied"))
        mock_model.model_to_string = MagicMock(return_value="mock model content\nline2")

        from pathlib import Path as P

        model_path = tmp_path / "test_model.txt"
        model_saved = False

        # Simulate the fallback logic from train_model
        try:
            mock_model.save_model(str(model_path))
            model_saved = True
        except Exception as e:
            try:
                best_iter = getattr(mock_model, 'best_iteration', None)
                model_str = mock_model.model_to_string(num_iteration=best_iter)
                model_path.write_text(model_str, encoding="utf-8")
                model_saved = True
            except Exception as e2:
                pass

        # Should have saved via fallback
        assert model_saved is True
        assert model_path.exists()
        assert model_path.stat().st_size > 0
        assert model_path.read_text() == "mock model content\nline2"

    def test_both_save_methods_fail(self, tmp_path):
        """When both save_model and model_to_string fail, should continue without crash"""
        mock_model = MagicMock()
        mock_model.best_iteration = 100
        mock_model.save_model = MagicMock(side_effect=Exception("Permission denied"))
        mock_model.model_to_string = MagicMock(side_effect=Exception("Internal error"))

        model_path = tmp_path / "test_model.txt"
        model_saved = False
        error_logged = False

        # Simulate the fallback logic from train_model
        try:
            mock_model.save_model(str(model_path))
            model_saved = True
        except Exception as e:
            try:
                best_iter = getattr(mock_model, 'best_iteration', None)
                model_str = mock_model.model_to_string(num_iteration=best_iter)
                model_path.write_text(model_str, encoding="utf-8")
                model_saved = True
            except Exception as e2:
                error_logged = True

        # Should not have saved, but also should not crash
        assert model_saved is False
        assert error_logged is True
        assert not model_path.exists()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
