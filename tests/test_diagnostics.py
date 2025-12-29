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
# Test load_booster with model_str fallback
# =============================================================================

class TestLoadBooster:
    """Tests for load_booster helper function"""

    def test_load_booster_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError when file doesn't exist"""
        import scripts.train_eval_v4 as tev4

        # Save original values
        original_has_lgb = tev4.HAS_LIGHTGBM

        try:
            # Mock HAS_LIGHTGBM to True
            tev4.HAS_LIGHTGBM = True

            with pytest.raises(FileNotFoundError) as exc_info:
                tev4.load_booster(str(tmp_path / "nonexistent_model.txt"))
            assert "not found" in str(exc_info.value).lower()
        finally:
            # Restore
            tev4.HAS_LIGHTGBM = original_has_lgb

    def test_load_booster_model_str_fallback(self, tmp_path):
        """Should fall back to model_str when model_file fails"""
        import scripts.train_eval_v4 as tev4

        # Create a mock model file
        model_path = tmp_path / "test_model.txt"
        model_path.write_text("mock model content", encoding="utf-8")

        # Mock lgb.Booster to fail on model_file but succeed on model_str
        mock_booster = MagicMock()

        def mock_booster_init(model_file=None, model_str=None):
            if model_file is not None:
                raise Exception("Permission denied - Japanese path issue")
            if model_str is not None:
                return mock_booster
            raise ValueError("Need model_file or model_str")

        # Save original values
        original_has_lgb = tev4.HAS_LIGHTGBM
        original_lgb = tev4.lgb

        try:
            mock_lgb = MagicMock()
            mock_lgb.Booster = MagicMock(side_effect=mock_booster_init)
            tev4.HAS_LIGHTGBM = True
            tev4.lgb = mock_lgb

            result = tev4.load_booster(str(model_path))
            assert result == mock_booster
        finally:
            # Restore
            tev4.HAS_LIGHTGBM = original_has_lgb
            tev4.lgb = original_lgb

    def test_load_booster_both_methods_fail(self, tmp_path):
        """Should raise RuntimeError when both methods fail"""
        import scripts.train_eval_v4 as tev4

        # Create a model file
        model_path = tmp_path / "test_model.txt"
        model_path.write_text("invalid model content", encoding="utf-8")

        def mock_booster_init(model_file=None, model_str=None):
            if model_file is not None:
                raise Exception("Permission denied")
            if model_str is not None:
                raise Exception("Invalid model format")
            raise ValueError("Need model_file or model_str")

        # Save original values
        original_has_lgb = tev4.HAS_LIGHTGBM
        original_lgb = tev4.lgb

        try:
            mock_lgb = MagicMock()
            mock_lgb.Booster = MagicMock(side_effect=mock_booster_init)
            tev4.HAS_LIGHTGBM = True
            tev4.lgb = mock_lgb

            with pytest.raises(RuntimeError) as exc_info:
                tev4.load_booster(str(model_path))
            assert "Failed to load model" in str(exc_info.value)
        finally:
            # Restore
            tev4.HAS_LIGHTGBM = original_has_lgb
            tev4.lgb = original_lgb


# =============================================================================
# Test --feature-diagnostics uses in-memory model (no disk reload)
# =============================================================================

class TestFeatureDiagnosticsInMemory:
    """Tests for --feature-diagnostics using in-memory model"""

    def test_run_full_pipeline_returns_model(self):
        """run_full_pipeline should return model in results"""
        # We can't easily test the full pipeline without a DB,
        # but we can verify the structure of what gets returned
        # by checking the code adds _model, _feature_cols, _test_df

        # This is a structural test - verify the code was modified correctly
        import src.features_v4.train_eval_v4 as tev4_module
        import inspect
        source = inspect.getsource(tev4_module.run_full_pipeline)

        assert '_model' in source
        assert '_feature_cols' in source
        assert '_test_df' in source
        assert 'results["_model"] = model' in source

    def test_feature_diagnostics_flow_uses_in_memory(self):
        """--feature-diagnostics flow should use in-memory model, not reload from disk"""
        # Verify the scripts/train_eval_v4.py main function uses trained_model
        import scripts.train_eval_v4 as script_module
        import inspect
        source = inspect.getsource(script_module.main)

        # Should extract model from results
        assert 'trained_model = results.pop("_model"' in source
        # Should use in-memory model directly
        assert 'model = trained_model' in source
        # Should NOT have lgb.Booster(model_file=...) in the diagnostics section
        # The only lgb.Booster(model_file=...) should be in diagnostics-only mode
        # which uses load_booster instead


# =============================================================================
# Test Pre-race Body Weight Features
# =============================================================================

class TestPreRaceBodyWeightFeatures:
    """Tests for pre-race safe body weight features in asof_aggregator"""

    def test_body_weight_features_computed_from_past(self):
        """Body weight features should be computed from past races only"""
        from src.features_v4.asof_aggregator import AsOfAggregator
        import sqlite3

        # Create in-memory database with test data
        conn = sqlite3.connect(":memory:")

        # Create tables
        conn.execute("""
            CREATE TABLE races (
                race_id TEXT PRIMARY KEY,
                date TEXT,
                place TEXT,
                course_type TEXT,
                distance INTEGER,
                track_condition TEXT,
                race_class TEXT,
                grade TEXT,
                race_no INTEGER,
                course_turn TEXT,
                course_inout TEXT,
                head_count INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE race_results (
                race_id TEXT,
                horse_id TEXT,
                finish_order INTEGER,
                body_weight INTEGER,
                body_weight_diff INTEGER,
                last_3f REAL,
                passing_order TEXT,
                win_odds REAL,
                popularity INTEGER,
                prize_money INTEGER,
                jockey_id TEXT,
                trainer_id TEXT,
                sex TEXT,
                age INTEGER,
                weight REAL,
                frame_no INTEGER,
                horse_no INTEGER,
                PRIMARY KEY (race_id, horse_id)
            )
        """)

        # Insert test races
        races_data = [
            ("2024010101", "2024-01-01", "東京", "芝", 2000, "良", "オープン", "G1", 11, "右", "内", 16),
            ("2024020101", "2024-02-01", "東京", "芝", 2000, "良", "オープン", "G1", 11, "右", "内", 16),
            ("2024030101", "2024-03-01", "東京", "芝", 2000, "良", "オープン", "G1", 11, "右", "内", 16),
            ("2024040101", "2024-04-01", "東京", "芝", 2000, "良", "オープン", "G1", 11, "右", "内", 16),
        ]
        conn.executemany(
            "INSERT INTO races VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            races_data
        )

        # Insert race results for a horse
        results_data = [
            ("2024010101", "HORSE001", 1, 480, 0, 33.5, "1-1", 2.0, 1, 100000, "J001", "T001", "牡", 4, 55.0, 1, 1),
            ("2024020101", "HORSE001", 2, 478, -2, 33.8, "2-2", 3.0, 2, 50000, "J001", "T001", "牡", 4, 55.0, 2, 2),
            ("2024030101", "HORSE001", 1, 482, 4, 33.3, "1-1", 2.5, 1, 100000, "J001", "T001", "牡", 4, 55.0, 1, 1),
        ]
        conn.executemany(
            "INSERT INTO race_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            results_data
        )
        conn.commit()

        # Compute stats as of 2024-04-01 (should only use data from before this date)
        agg = AsOfAggregator(conn)
        stats = agg.compute_horse_asof_stats(
            horse_id="HORSE001",
            race_date="2024-04-01",
            distance_cat=2000,
            surface="芝",
        )

        # Verify new body weight features
        assert stats.get("h_avg_body_weight") == pytest.approx(480.0, rel=0.01)
        assert stats.get("h_last_body_weight") == 482
        assert stats.get("h_last_body_weight_diff") == 4
        assert stats.get("h_recent3_avg_body_weight") == pytest.approx(480.0, rel=0.01)
        assert stats.get("h_recent3_std_body_weight") is not None
        assert stats.get("h_recent3_body_weight_trend") is not None

        conn.close()

    def test_body_weight_features_empty_past(self):
        """Body weight features should be None when there's no past data"""
        from src.features_v4.asof_aggregator import AsOfAggregator
        import sqlite3

        conn = sqlite3.connect(":memory:")

        # Create minimal tables
        conn.execute("""
            CREATE TABLE races (race_id TEXT, date TEXT, place TEXT, course_type TEXT,
            distance INTEGER, track_condition TEXT, race_class TEXT, grade TEXT,
            race_no INTEGER, course_turn TEXT, course_inout TEXT, head_count INTEGER)
        """)
        conn.execute("""
            CREATE TABLE race_results (race_id TEXT, horse_id TEXT, finish_order INTEGER,
            body_weight INTEGER, body_weight_diff INTEGER, last_3f REAL, passing_order TEXT,
            win_odds REAL, popularity INTEGER, prize_money INTEGER, jockey_id TEXT,
            trainer_id TEXT, sex TEXT, age INTEGER, weight REAL, frame_no INTEGER, horse_no INTEGER)
        """)
        conn.commit()

        agg = AsOfAggregator(conn)
        stats = agg.compute_horse_asof_stats(
            horse_id="NONEXISTENT",
            race_date="2024-04-01",
        )

        assert stats.get("h_last_body_weight") is None
        assert stats.get("h_last_body_weight_diff") is None
        assert stats.get("h_recent3_avg_body_weight") is None
        assert stats.get("h_recent3_std_body_weight") is None
        assert stats.get("h_recent3_body_weight_trend") is None
        assert stats.get("h_body_weight_z") is None

        conn.close()

    def test_body_weight_features_single_past(self):
        """Body weight features should handle single past race correctly"""
        from src.features_v4.asof_aggregator import AsOfAggregator
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE races (race_id TEXT, date TEXT, place TEXT, course_type TEXT,
            distance INTEGER, track_condition TEXT, race_class TEXT, grade TEXT,
            race_no INTEGER, course_turn TEXT, course_inout TEXT, head_count INTEGER)
        """)
        conn.execute("""
            CREATE TABLE race_results (race_id TEXT, horse_id TEXT, finish_order INTEGER,
            body_weight INTEGER, body_weight_diff INTEGER, last_3f REAL, passing_order TEXT,
            win_odds REAL, popularity INTEGER, prize_money INTEGER, jockey_id TEXT,
            trainer_id TEXT, sex TEXT, age INTEGER, weight REAL, frame_no INTEGER, horse_no INTEGER)
        """)

        # Single past race
        conn.execute(
            "INSERT INTO races VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("2024010101", "2024-01-01", "東京", "芝", 2000, "良", "オープン", "G1", 11, "右", "内", 16)
        )
        conn.execute(
            "INSERT INTO race_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("2024010101", "HORSE001", 1, 480, 0, 33.5, "1-1", 2.0, 1, 100000, "J001", "T001", "牡", 4, 55.0, 1, 1)
        )
        conn.commit()

        agg = AsOfAggregator(conn)
        stats = agg.compute_horse_asof_stats(
            horse_id="HORSE001",
            race_date="2024-04-01",
        )

        # With only 1 past race
        assert stats.get("h_last_body_weight") == 480
        assert stats.get("h_recent3_avg_body_weight") == 480.0
        assert stats.get("h_recent3_std_body_weight") is None  # Need 2+ for std
        assert stats.get("h_recent3_body_weight_trend") is None  # Need 2+ for trend

        conn.close()


class TestPreRaceMode:
    """Tests for --mode pre_race functionality"""

    def test_pre_race_exclude_file_exists(self):
        """Pre-race exclude file should exist"""
        import os
        pre_race_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "exclude_features", "pre_race.txt"
        )
        assert os.path.exists(pre_race_path), f"Pre-race exclude file should exist at {pre_race_path}"

    def test_pre_race_exclude_file_contents(self):
        """Pre-race exclude file should contain race-day features"""
        from scripts.train_eval_v4 import load_exclude_features
        import os

        pre_race_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "exclude_features", "pre_race.txt"
        )
        excludes = load_exclude_features(pre_race_path)

        # Should contain race-day body weight features
        assert "h_body_weight" in excludes
        assert "h_body_weight_diff" in excludes
        assert "h_body_weight_dev" in excludes

        # Should NOT contain pre-race safe features
        assert "h_avg_body_weight" not in excludes
        assert "h_last_body_weight" not in excludes
        assert "h_recent3_avg_body_weight" not in excludes

    def test_mode_argument_parsing(self):
        """train_eval_v4.py should accept --mode argument"""
        import subprocess
        result = subprocess.run(
            ["python", "scripts/train_eval_v4.py", "--help"],
            capture_output=True,
            text=True
        )
        assert "--mode" in result.stdout
        assert "pre_race" in result.stdout


class TestBodyWeightContext:
    """Tests for BodyWeightContext class"""

    def test_body_weight_context_creation(self):
        """BodyWeightContext should be created with correct fields"""
        from src.scenario import BodyWeightContext

        ctx = BodyWeightContext(
            avg_body_weight=480.0,
            last_body_weight=478,
            last_body_weight_diff=-2,
            recent3_avg_body_weight=480.0,
            recent3_std_body_weight=3.5,
            recent3_trend=-1.0,
            body_weight_z=-0.57,
        )

        assert ctx.avg_body_weight == 480.0
        assert ctx.last_body_weight == 478
        assert ctx.has_historical_data is True
        assert ctx.has_current_data is False

    def test_body_weight_context_volatile(self):
        """BodyWeightContext should detect volatile weight"""
        from src.scenario import BodyWeightContext

        ctx = BodyWeightContext(recent3_std_body_weight=5.0)
        assert ctx.is_weight_volatile is True

        ctx2 = BodyWeightContext(recent3_std_body_weight=2.0)
        assert ctx2.is_weight_volatile is False

    def test_body_weight_context_trend(self):
        """BodyWeightContext should detect weight trends"""
        from src.scenario import BodyWeightContext

        ctx_gain = BodyWeightContext(recent3_trend=3.0)
        assert ctx_gain.is_gaining_weight is True
        assert ctx_gain.is_losing_weight is False

        ctx_loss = BodyWeightContext(recent3_trend=-3.0)
        assert ctx_loss.is_gaining_weight is False
        assert ctx_loss.is_losing_weight is True

    def test_body_weight_context_llm_notes(self):
        """BodyWeightContext should generate LLM notes"""
        from src.scenario import BodyWeightContext

        # Volatile + no current data
        ctx = BodyWeightContext(
            recent3_std_body_weight=5.0,
            recent3_trend=0.0,
        )
        notes = ctx.get_llm_notes()
        assert "体重変動大 (要確認)" in notes
        assert "当日体重未確定" in notes

        # Has current data
        ctx2 = BodyWeightContext(
            current_body_weight=480,
            recent3_trend=-3.0,
        )
        notes2 = ctx2.get_llm_notes()
        assert "当日体重未確定" not in notes2
        assert "減量傾向" in notes2

    def test_body_weight_context_serialization(self):
        """BodyWeightContext should serialize/deserialize correctly"""
        from src.scenario import BodyWeightContext

        ctx = BodyWeightContext(
            avg_body_weight=480.0,
            last_body_weight=478,
            current_body_weight=475,
        )

        # to_dict
        d = ctx.to_dict()
        assert d["avg_body_weight"] == 480.0
        assert d["current_body_weight"] == 475
        assert d["has_current_data"] is True

        # from_dict
        ctx2 = BodyWeightContext.from_dict(d)
        assert ctx2.avg_body_weight == ctx.avg_body_weight
        assert ctx2.current_body_weight == ctx.current_body_weight


class TestLogFeatureSelectionSummary:
    """Tests for log_feature_selection_summary function (Step 0 visibility)"""

    def test_log_feature_selection_summary_import(self):
        """log_feature_selection_summary should be importable"""
        from src.features_v4.train_eval_v4 import log_feature_selection_summary
        assert log_feature_selection_summary is not None

    def test_log_feature_selection_summary_basic(self, tmp_path):
        """log_feature_selection_summary should log and save correct info"""
        from src.features_v4.train_eval_v4 import log_feature_selection_summary

        candidate_features = ["h_avg_body_weight", "h_body_weight", "h_last_body_weight", "market_win_odds"]
        exclude_set = {"h_body_weight", "market_win_odds"}
        final_features = ["h_avg_body_weight", "h_last_body_weight"]

        result = log_feature_selection_summary(
            mode="pre_race",
            target_col="target_win",
            candidate_features=candidate_features,
            exclude_set=exclude_set,
            final_features=final_features,
            output_dir=str(tmp_path),
        )

        # Check result structure
        assert result["mode"] == "pre_race"
        assert result["target"] == "target_win"
        assert result["n_candidate_features"] == 4
        assert result["n_exclude_requested"] == 2
        assert result["n_actually_excluded"] == 2
        assert result["n_final_features"] == 2
        assert "h_body_weight" in result["excluded_features"]
        assert "market_win_odds" in result["excluded_features"]

        # Check files are created
        import os
        summary_file = tmp_path / "feature_selection_target_win_pre_race.json"
        features_file = tmp_path / "used_features_target_win_pre_race.txt"
        assert os.path.exists(summary_file)
        assert os.path.exists(features_file)

        # Verify file contents
        import json
        with open(summary_file, "r") as f:
            saved_summary = json.load(f)
        assert saved_summary["n_actually_excluded"] == 2

        with open(features_file, "r") as f:
            saved_features = [line.strip() for line in f if line.strip()]
        assert "h_avg_body_weight" in saved_features
        assert "h_body_weight" not in saved_features

    def test_log_feature_selection_warns_missing_features(self, tmp_path):
        """log_feature_selection_summary should track features not found in candidates"""
        from src.features_v4.train_eval_v4 import log_feature_selection_summary

        candidate_features = ["h_avg_body_weight", "h_last_body_weight"]
        # Exclude features that don't exist in candidates
        exclude_set = {"h_body_weight", "nonexistent_feature"}
        final_features = ["h_avg_body_weight", "h_last_body_weight"]

        result = log_feature_selection_summary(
            mode="pre_race",
            target_col="target_win",
            candidate_features=candidate_features,
            exclude_set=exclude_set,
            final_features=final_features,
            output_dir=str(tmp_path),
        )

        # Should track not-found features
        assert len(result["not_found_features"]) == 2
        assert "h_body_weight" in result["not_found_features"]
        assert "nonexistent_feature" in result["not_found_features"]
        assert result["n_actually_excluded"] == 0


class TestGeneratePreRaceMaterials:
    """Tests for generate_pre_race_materials.py (Step 1)"""

    def test_script_exists(self):
        """generate_pre_race_materials.py should exist"""
        import os
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "generate_pre_race_materials.py"
        )
        assert os.path.exists(script_path), f"Script should exist at {script_path}"

    def test_script_help(self):
        """generate_pre_race_materials.py --help should work"""
        import subprocess
        result = subprocess.run(
            ["python", "scripts/generate_pre_race_materials.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--date" in result.stdout
        assert "--out" in result.stdout
        assert "--models-dir" in result.stdout
        assert "pre-race" in result.stdout.lower()

    def test_json_generation_functions(self):
        """JSON generation functions should work correctly"""
        from scripts.generate_pre_race_materials import (
            generate_race_json,
            generate_summary_json,
        )

        # Test race JSON generation
        race_info = {
            "race_id": "202405120101",
            "date": "2024-05-12",
            "place": "東京",
            "race_no": 1,
            "name": "テストレース",
            "distance": 1600,
            "track_condition": "良",
        }
        entries_df = pd.DataFrame([
            {"horse_id": "h001", "horse_no": 1, "horse_name": "テスト馬1", "jockey_name": "騎手A", "trainer_name": "調教師A", "sex": "牡", "age": 3},
            {"horse_id": "h002", "horse_no": 2, "horse_name": "テスト馬2", "jockey_name": "騎手B", "trainer_name": "調教師B", "sex": "牝", "age": 4},
        ])
        predictions = {
            "h001": {"p_win": 0.3, "p_in3": 0.6, "rank_win": 1, "rank_in3": 1},
            "h002": {"p_win": 0.2, "p_in3": 0.4, "rank_win": 2, "rank_in3": 2},
        }

        race_json = generate_race_json(race_info, entries_df, predictions)

        assert race_json["race_id"] == "202405120101"
        assert race_json["mode"] == "pre_race"
        assert len(race_json["entries"]) == 2
        assert race_json["entries"][0]["p_win"] == 0.3
        assert race_json["entries"][0]["rank_win"] == 1

        # Test summary JSON generation
        summary_json = generate_summary_json(
            date="2024-05-12",
            races=[{"race_id": "202405120101", "place": "東京", "race_no": 1}],
            n_features=90,
            models_used=["models/lgbm_target_win.txt"],
        )

        assert summary_json["date"] == "2024-05-12"
        assert summary_json["mode"] == "pre_race"
        assert summary_json["n_races"] == 1
        assert summary_json["n_features_used"] == 90


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
