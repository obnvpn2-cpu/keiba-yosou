#!/usr/bin/env python3
"""
Tests for Feature Comparison Report Generator (Step E)

Tests for:
- Report generation logic
- CSV output format
- Cross-version comparison
- Safety summary
- Windows path compatibility
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.report_featurepacks import (
    VersionAuditData,
    generate_feature_compare,
    generate_top_features,
    generate_common_top_features,
    generate_diff_v4_vs_v3,
    generate_safety_summary,
    generate_index_md,
    load_audit_data,
    write_csv,
    write_json,
    run_report,
    _get_rank,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_audit_data():
    """Create sample audit data for testing."""
    return {
        "v4": {
            "target_win": VersionAuditData(
                version="v4",
                target="target_win",
                mode="pre_race",
                features=["h_career_win_rate", "j_win_rate", "race_distance", "new_v4_feature"],
                importance_gain={
                    "h_career_win_rate": 100.0,
                    "j_win_rate": 80.0,
                    "race_distance": 50.0,
                    "new_v4_feature": 30.0,
                },
                importance_split={
                    "h_career_win_rate": 200,
                    "j_win_rate": 150,
                    "race_distance": 100,
                    "new_v4_feature": 50,
                },
                summary={
                    "version": "v4",
                    "target": "target_win",
                    "n_features": 4,
                    "n_unsafe": 0,
                    "n_warn": 0,
                    "n_safe": 4,
                },
            ),
        },
        "v3": {
            "target_win": VersionAuditData(
                version="v3",
                target="target_win",
                mode="pre_race",
                features=["h_career_win_rate", "j_win_rate", "race_distance", "old_v3_feature"],
                importance_gain={
                    "h_career_win_rate": 90.0,
                    "j_win_rate": 85.0,
                    "race_distance": 40.0,
                    "old_v3_feature": 60.0,
                },
                importance_split={
                    "h_career_win_rate": 180,
                    "j_win_rate": 160,
                    "race_distance": 80,
                    "old_v3_feature": 120,
                },
                summary={
                    "version": "v3",
                    "target": "target_win",
                    "n_features": 4,
                    "n_unsafe": 0,
                    "n_warn": 0,
                    "n_safe": 4,
                },
            ),
        },
        "v2": {
            "target_win": VersionAuditData(
                version="v2",
                target="target_win",
                mode="pre_race",
                features=["h_career_win_rate", "j_win_rate"],
                importance_gain={
                    "h_career_win_rate": 70.0,
                    "j_win_rate": 60.0,
                },
                importance_split={
                    "h_career_win_rate": 140,
                    "j_win_rate": 120,
                },
                summary={
                    "version": "v2",
                    "target": "target_win",
                    "n_features": 2,
                    "n_unsafe": 0,
                    "n_warn": 0,
                    "n_safe": 2,
                },
            ),
        },
    }


@pytest.fixture
def sample_audit_data_with_unsafe():
    """Create sample audit data with unsafe features."""
    return {
        "v4": {
            "target_win": VersionAuditData(
                version="v4",
                target="target_win",
                mode="pre_race",
                features=["h_career_win_rate", "h_body_weight", "win_odds"],
                importance_gain={
                    "h_career_win_rate": 100.0,
                    "h_body_weight": 80.0,  # unsafe
                    "win_odds": 60.0,  # unsafe
                },
                importance_split={},
                summary={
                    "version": "v4",
                    "target": "target_win",
                    "n_features": 3,
                    "n_unsafe": 2,
                    "n_warn": 0,
                    "n_safe": 1,
                },
            ),
        },
    }


# =============================================================================
# Test: Feature Comparison
# =============================================================================

class TestFeatureCompare:
    """Tests for feature comparison generation."""

    def test_generate_feature_compare_basic(self, sample_audit_data):
        """Basic feature comparison should work."""
        result = generate_feature_compare(
            sample_audit_data,
            "target_win",
            ["v4", "v3", "v2"]
        )

        assert len(result) > 0
        assert all("feature" in row for row in result)
        assert all("safety_label" in row for row in result)

    def test_feature_compare_has_all_features(self, sample_audit_data):
        """Should include all features from all versions."""
        result = generate_feature_compare(
            sample_audit_data,
            "target_win",
            ["v4", "v3", "v2"]
        )

        features = [r["feature"] for r in result]
        assert "h_career_win_rate" in features
        assert "j_win_rate" in features
        assert "race_distance" in features
        assert "new_v4_feature" in features
        assert "old_v3_feature" in features

    def test_feature_compare_version_columns(self, sample_audit_data):
        """Should have columns for each version."""
        result = generate_feature_compare(
            sample_audit_data,
            "target_win",
            ["v4", "v3"]
        )

        first_row = result[0]
        assert "v4_rank" in first_row
        assert "v4_gain" in first_row
        assert "v4_gain_norm" in first_row
        assert "v3_rank" in first_row
        assert "v3_gain" in first_row

    def test_feature_compare_normalized_gain(self, sample_audit_data):
        """Normalized gain should be between 0 and 1."""
        result = generate_feature_compare(
            sample_audit_data,
            "target_win",
            ["v4"]
        )

        for row in result:
            if row.get("v4_gain_norm") is not None:
                assert 0 <= row["v4_gain_norm"] <= 1

    def test_feature_compare_sorted_by_v4_rank(self, sample_audit_data):
        """Results should be sorted by v4 rank."""
        result = generate_feature_compare(
            sample_audit_data,
            "target_win",
            ["v4", "v3"]
        )

        # First feature should be top v4 feature
        assert result[0]["feature"] == "h_career_win_rate"


class TestTopFeatures:
    """Tests for top features generation."""

    def test_generate_top_features_basic(self, sample_audit_data):
        """Basic top features should work."""
        result = generate_top_features(
            sample_audit_data,
            "target_win",
            "v4",
            top_n=10
        )

        assert len(result) > 0
        assert result[0]["rank"] == 1
        assert result[0]["feature"] == "h_career_win_rate"

    def test_top_features_has_cumulative(self, sample_audit_data):
        """Should have cumulative percentage."""
        result = generate_top_features(
            sample_audit_data,
            "target_win",
            "v4",
            top_n=10
        )

        for row in result:
            assert "cumulative_pct" in row
            assert 0 <= row["cumulative_pct"] <= 1

    def test_top_features_respects_limit(self, sample_audit_data):
        """Should respect top_n limit."""
        result = generate_top_features(
            sample_audit_data,
            "target_win",
            "v4",
            top_n=2
        )

        assert len(result) == 2


class TestCommonTopFeatures:
    """Tests for common top features generation."""

    def test_common_features_basic(self, sample_audit_data):
        """Common features should work."""
        result = generate_common_top_features(
            sample_audit_data,
            "target_win",
            ["v4", "v3", "v2"],
            top_n=10
        )

        assert len(result) > 0

    def test_common_features_n_versions(self, sample_audit_data):
        """Should count number of versions."""
        result = generate_common_top_features(
            sample_audit_data,
            "target_win",
            ["v4", "v3", "v2"],
            top_n=10
        )

        # h_career_win_rate and j_win_rate should be in all 3 versions
        common = [r for r in result if r["n_versions"] == 3]
        assert len(common) >= 2


class TestDiffV4VsV3:
    """Tests for V4 vs V3 diff generation."""

    def test_diff_basic(self, sample_audit_data):
        """Basic diff should work."""
        result = generate_diff_v4_vs_v3(
            sample_audit_data,
            "target_win"
        )

        assert len(result) > 0

    def test_diff_detects_new_features(self, sample_audit_data):
        """Should detect new features in v4."""
        result = generate_diff_v4_vs_v3(
            sample_audit_data,
            "target_win"
        )

        new_features = [r for r in result if r["change_type"] == "new_in_v4"]
        assert any(r["feature"] == "new_v4_feature" for r in new_features)

    def test_diff_detects_removed_features(self, sample_audit_data):
        """Should detect removed features in v4."""
        result = generate_diff_v4_vs_v3(
            sample_audit_data,
            "target_win"
        )

        removed = [r for r in result if r["change_type"] == "removed_in_v4"]
        assert any(r["feature"] == "old_v3_feature" for r in removed)

    def test_diff_has_gain_diff(self, sample_audit_data):
        """Should have gain diff for changed features."""
        result = generate_diff_v4_vs_v3(
            sample_audit_data,
            "target_win"
        )

        for row in result:
            assert "gain_diff" in row


class TestSafetySummary:
    """Tests for safety summary generation."""

    def test_safety_summary_basic(self, sample_audit_data):
        """Basic safety summary should work."""
        result = generate_safety_summary(
            sample_audit_data,
            "target_win",
            ["v4", "v3"]
        )

        assert len(result) == 2  # v4 and v3

    def test_safety_summary_counts(self, sample_audit_data):
        """Should have correct counts."""
        result = generate_safety_summary(
            sample_audit_data,
            "target_win",
            ["v4"]
        )

        assert result[0]["n_features"] == 4
        assert result[0]["n_safe"] == 4
        assert result[0]["n_unsafe"] == 0

    def test_safety_summary_detects_unsafe(self, sample_audit_data_with_unsafe):
        """Should detect unsafe features in top N."""
        result = generate_safety_summary(
            sample_audit_data_with_unsafe,
            "target_win",
            ["v4"],
            top_n=10
        )

        assert result[0]["n_unsafe"] == 2
        assert result[0]["unsafe_in_top_n"] == 2


# =============================================================================
# Test: Index Markdown Generation
# =============================================================================

class TestIndexMarkdown:
    """Tests for index.md generation."""

    def test_index_md_basic(self, sample_audit_data, tmp_path):
        """Basic index.md should be generated."""
        result = generate_index_md(
            sample_audit_data,
            ["target_win"],
            ["v4", "v3"],
            tmp_path
        )

        assert "# Feature Comparison Report" in result
        assert "## Overview" in result

    def test_index_md_has_tables(self, sample_audit_data, tmp_path):
        """Index should have markdown tables."""
        result = generate_index_md(
            sample_audit_data,
            ["target_win"],
            ["v4", "v3"],
            tmp_path
        )

        assert "|" in result
        assert "Version" in result or "Feature" in result

    def test_index_md_has_safety_warnings(self, sample_audit_data_with_unsafe, tmp_path):
        """Index should show safety warnings."""
        result = generate_index_md(
            sample_audit_data_with_unsafe,
            ["target_win"],
            ["v4"],
            tmp_path
        )

        assert "Safety" in result


# =============================================================================
# Test: File Writers
# =============================================================================

class TestFileWriters:
    """Tests for file output functions."""

    def test_write_csv(self, tmp_path):
        """write_csv should create valid CSV."""
        data = [
            {"feature": "a", "value": 1},
            {"feature": "b", "value": 2},
        ]
        path = tmp_path / "test.csv"

        write_csv(path, data)

        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["feature"] == "a"

    def test_write_csv_custom_fieldnames(self, tmp_path):
        """write_csv should respect custom fieldnames order."""
        data = [
            {"a": 1, "b": 2, "c": 3},
        ]
        path = tmp_path / "test.csv"

        # Custom order with all fields
        write_csv(path, data, fieldnames=["c", "b", "a"])

        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip()

        assert header == "c,b,a"

    def test_write_json(self, tmp_path):
        """write_json should create valid JSON."""
        data = {"key": "value", "list": [1, 2, 3]}
        path = tmp_path / "test.json"

        write_json(path, data)

        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded == data

    def test_write_csv_creates_parent_dirs(self, tmp_path):
        """write_csv should create parent directories."""
        path = tmp_path / "subdir" / "nested" / "test.csv"

        write_csv(path, [{"a": 1}])

        assert path.exists()


# =============================================================================
# Test: Data Loading
# =============================================================================

class TestDataLoading:
    """Tests for data loading functions."""

    def test_load_audit_data_missing_dir(self, tmp_path):
        """Should handle missing directories gracefully."""
        result = load_audit_data(
            tmp_path / "nonexistent",
            ["v4"],
            ["target_win"]
        )

        assert "v4" in result
        assert "target_win" not in result["v4"]  # No data loaded

    def test_load_audit_data_partial_files(self, tmp_path):
        """Should handle partial file availability."""
        # Create directory structure with only some files
        audit_dir = tmp_path / "v4" / "pre_race" / "target_win"
        audit_dir.mkdir(parents=True)

        # Only create used_features.txt
        (audit_dir / "used_features.txt").write_text("feature_a\nfeature_b\n")

        result = load_audit_data(
            tmp_path,
            ["v4"],
            ["target_win"]
        )

        assert "v4" in result
        assert "target_win" in result["v4"]
        assert result["v4"]["target_win"].features == ["feature_a", "feature_b"]
        assert result["v4"]["target_win"].importance_gain == {}  # No importance file


# =============================================================================
# Test: Full Report Generation
# =============================================================================

class TestRunReport:
    """Tests for full report generation."""

    def test_run_report_dry_run(self, tmp_path):
        """Dry run should not create files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        # Create minimal input
        audit_dir = input_dir / "v4" / "pre_race" / "target_win"
        audit_dir.mkdir(parents=True)
        (audit_dir / "used_features.txt").write_text("feature_a\n")
        (audit_dir / "importance_gain.csv").write_text("feature,importance\nfeature_a,100\n")

        exit_code = run_report(
            input_dir=input_dir,
            output_dir=output_dir,
            versions=["v4"],
            targets=["target_win"],
            dry_run=True,
        )

        assert exit_code == 0
        assert not output_dir.exists()  # Should not create output

    def test_run_report_creates_files(self, tmp_path):
        """Full run should create report files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        # Create input data
        audit_dir = input_dir / "v4" / "pre_race" / "target_win"
        audit_dir.mkdir(parents=True)
        (audit_dir / "used_features.txt").write_text("feature_a\nfeature_b\n")
        (audit_dir / "importance_gain.csv").write_text(
            "feature,importance\nfeature_a,100\nfeature_b,50\n"
        )
        (audit_dir / "summary.json").write_text(json.dumps({
            "version": "v4",
            "target": "target_win",
            "n_features": 2,
            "n_unsafe": 0,
            "n_warn": 0,
            "n_safe": 2,
        }))

        exit_code = run_report(
            input_dir=input_dir,
            output_dir=output_dir,
            versions=["v4"],
            targets=["target_win"],
            dry_run=False,
        )

        assert exit_code == 0
        assert output_dir.exists()
        assert (output_dir / "index.md").exists()
        assert (output_dir / "top_features_target_win_v4.csv").exists()

    def test_run_report_no_data(self, tmp_path):
        """Should fail with no data."""
        exit_code = run_report(
            input_dir=tmp_path / "nonexistent",
            output_dir=tmp_path / "output",
            versions=["v4"],
            targets=["target_win"],
            dry_run=False,
        )

        assert exit_code == 1


# =============================================================================
# Test: Windows Path Compatibility
# =============================================================================

class TestWindowsPathCompatibility:
    """Tests for Windows path handling."""

    def test_path_as_posix_in_output(self, tmp_path):
        """Output paths should use forward slashes."""
        # Create Path objects
        path1 = tmp_path / "subdir" / "file.csv"

        # Convert to POSIX
        posix_path = path1.as_posix()

        assert "\\" not in posix_path
        assert "/" in posix_path

    def test_path_separator_independence(self, tmp_path):
        """Path operations should work regardless of separator."""
        # Create using forward slashes (POSIX style)
        path1 = Path(str(tmp_path) + "/subdir/file.csv")

        # Create using Path operations
        path2 = tmp_path / "subdir" / "file.csv"

        # Should be equivalent
        assert path1 == path2

    def test_csv_paths_in_index(self, sample_audit_data, tmp_path):
        """Index.md should have platform-independent paths."""
        result = generate_index_md(
            sample_audit_data,
            ["target_win"],
            ["v4"],
            tmp_path
        )

        # File references should use forward slashes
        assert "\\" not in result


# =============================================================================
# Test: Helper Functions
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_rank(self):
        """_get_rank should return correct rank."""
        gain_dict = {"a": 100, "b": 80, "c": 60}

        assert _get_rank(gain_dict, "a") == 1
        assert _get_rank(gain_dict, "b") == 2
        assert _get_rank(gain_dict, "c") == 3
        assert _get_rank(gain_dict, "d") is None

    def test_get_rank_empty(self):
        """_get_rank should handle empty dict."""
        assert _get_rank({}, "a") is None


# =============================================================================
# Test: Target Coverage
# =============================================================================

class TestTargetCoverage:
    """Tests for target_win and target_in3 coverage."""

    def test_both_targets_generated(self, tmp_path):
        """Should generate reports for both targets."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        # Create data for both targets
        for target in ["target_win", "target_in3"]:
            audit_dir = input_dir / "v4" / "pre_race" / target
            audit_dir.mkdir(parents=True)
            (audit_dir / "used_features.txt").write_text("feature_a\n")
            (audit_dir / "importance_gain.csv").write_text(
                "feature,importance\nfeature_a,100\n"
            )
            (audit_dir / "summary.json").write_text(json.dumps({
                "version": "v4",
                "target": target,
                "n_features": 1,
                "n_unsafe": 0,
                "n_warn": 0,
                "n_safe": 1,
            }))

        exit_code = run_report(
            input_dir=input_dir,
            output_dir=output_dir,
            versions=["v4"],
            targets=["target_win", "target_in3"],
            dry_run=False,
        )

        assert exit_code == 0
        assert (output_dir / "top_features_target_win_v4.csv").exists()
        assert (output_dir / "top_features_target_in3_v4.csv").exists()


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
