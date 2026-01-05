#!/usr/bin/env python3
"""
Tests for Feature Audit Runner (Step C)

Tests for:
- Safety label classification
- Adapter registry
- V4 adapter detection
- Index JSON format
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_audit.safety import (
    classify_feature_safety,
    classify_features_batch,
    get_unsafe_features,
    get_warn_features,
    get_safe_features,
    summarize_safety,
)

from src.feature_audit.adapters import (
    get_available_adapters,
    get_adapter,
    detect_all_versions,
    ADAPTER_REGISTRY,
)


# =============================================================================
# Test: Safety Label Classification (C2)
# =============================================================================

class TestSafetyClassification:
    """Tests for safety label assignment"""

    def test_unsafe_body_weight(self):
        """Body weight features should be unsafe"""
        unsafe_features = [
            "h_body_weight",
            "h_body_weight_diff",
            "h_body_weight_dev",
            "horse_weight",
            "horse_weight_diff",
        ]
        for feat in unsafe_features:
            label, notes = classify_feature_safety(feat)
            assert label == "unsafe", f"{feat} should be unsafe, got {label}"
            assert notes, f"{feat} should have notes"

    def test_unsafe_market_features(self):
        """Market features should be unsafe"""
        unsafe_features = [
            "market_win_odds",
            "market_popularity",
            "win_odds",
            "popularity",
        ]
        for feat in unsafe_features:
            label, notes = classify_feature_safety(feat)
            assert label == "unsafe", f"{feat} should be unsafe, got {label}"

    def test_warn_result_patterns(self):
        """Result-related patterns should be warn"""
        warn_features = [
            "result_time",
            "finish_score",
            "target_win",
            "payout_amount",
        ]
        for feat in warn_features:
            label, notes = classify_feature_safety(feat)
            assert label == "warn", f"{feat} should be warn, got {label}"

    def test_safe_normal_features(self):
        """Normal features should be safe"""
        safe_features = [
            "h_career_win_rate",
            "j_win_rate",
            "t_recent_form",
            "race_distance",
            "surface_id",
        ]
        for feat in safe_features:
            label, notes = classify_feature_safety(feat)
            assert label == "safe", f"{feat} should be safe, got {label}"

    def test_batch_classification(self):
        """Batch classification should work correctly"""
        features = ["h_body_weight", "result_time", "race_distance"]
        results = classify_features_batch(features)

        assert len(results) == 3
        assert results[0]["safety_label"] == "unsafe"
        assert results[1]["safety_label"] == "warn"
        assert results[2]["safety_label"] == "safe"

    def test_get_unsafe_features(self):
        """get_unsafe_features should filter correctly"""
        features = ["h_body_weight", "market_odds", "race_distance", "win_odds"]
        unsafe = get_unsafe_features(features)

        assert "h_body_weight" in unsafe
        assert "win_odds" in unsafe
        assert "race_distance" not in unsafe

    def test_get_warn_features(self):
        """get_warn_features should filter correctly"""
        features = ["result_time", "finish_score", "race_distance"]
        warn = get_warn_features(features)

        assert "result_time" in warn
        assert "finish_score" in warn
        assert "race_distance" not in warn

    def test_get_safe_features(self):
        """get_safe_features should filter correctly"""
        features = ["h_body_weight", "result_time", "race_distance"]
        safe = get_safe_features(features)

        assert "race_distance" in safe
        assert "h_body_weight" not in safe
        assert "result_time" not in safe

    def test_summarize_safety(self):
        """summarize_safety should return correct counts"""
        features = [
            "h_body_weight",      # unsafe
            "market_win_odds",    # unsafe
            "result_time",        # warn
            "race_distance",      # safe
            "surface_id",         # safe
        ]
        summary = summarize_safety(features)

        assert summary["n_total"] == 5
        assert summary["n_unsafe"] == 2
        assert summary["n_warn"] == 1
        assert summary["n_safe"] == 2
        assert len(summary["unsafe_features"]) == 2
        assert len(summary["warn_features"]) == 1


# =============================================================================
# Test: Adapter Registry (C4)
# =============================================================================

class TestAdapterRegistry:
    """Tests for adapter registry"""

    def test_registry_not_empty(self):
        """Registry should have at least one adapter"""
        adapters = get_available_adapters()
        assert len(adapters) > 0, "Registry should not be empty"

    def test_v4_registered(self):
        """V4 adapter should be registered"""
        assert "v4" in ADAPTER_REGISTRY, "v4 should be registered"

    def test_legacy_registered(self):
        """Legacy adapter should be registered"""
        assert "legacy" in ADAPTER_REGISTRY, "legacy should be registered"

    def test_get_adapter_v4(self):
        """get_adapter should return V4 adapter"""
        adapter = get_adapter("v4")
        assert adapter is not None
        assert adapter.VERSION_NAME == "v4"

    def test_get_adapter_legacy(self):
        """get_adapter should return legacy adapter"""
        adapter = get_adapter("legacy")
        assert adapter is not None
        assert adapter.VERSION_NAME == "legacy"

    def test_get_adapter_unknown(self):
        """get_adapter should return None for unknown version"""
        adapter = get_adapter("unknown_version")
        assert adapter is None


# =============================================================================
# Test: V4 Adapter Detection
# =============================================================================

class TestV4AdapterDetection:
    """Tests for V4 adapter detection"""

    def test_v4_detect_without_db(self):
        """V4 adapter should detect when module is available"""
        adapter = get_adapter("v4")
        assert adapter is not None

        # Without DB path, should check module availability
        can_run, reason = adapter.detect()
        # May fail if features_v4 not available or no DB
        # Just check it doesn't crash
        assert isinstance(can_run, bool)
        assert isinstance(reason, str)

    def test_v4_adapter_has_required_methods(self):
        """V4 adapter should have required methods"""
        adapter = get_adapter("v4")
        assert adapter is not None

        assert hasattr(adapter, "detect")
        assert hasattr(adapter, "list_feature_columns")
        assert hasattr(adapter, "get_used_features")
        assert hasattr(adapter, "get_feature_matrix_sample")
        assert hasattr(adapter, "get_importance")
        assert hasattr(adapter, "run_audit")


# =============================================================================
# Test: Legacy Adapter Detection
# =============================================================================

class TestLegacyAdapterDetection:
    """Tests for legacy adapter detection"""

    def test_legacy_detect_without_db(self):
        """Legacy adapter should fail without DB"""
        adapter = get_adapter("legacy")
        assert adapter is not None

        # Without DB path, should fail
        can_run, reason = adapter.detect()
        assert can_run is False
        assert "not specified" in reason.lower() or "exist" in reason.lower()

    def test_legacy_adapter_has_required_methods(self):
        """Legacy adapter should have required methods"""
        adapter = get_adapter("legacy")
        assert adapter is not None

        assert hasattr(adapter, "detect")
        assert hasattr(adapter, "list_feature_columns")
        assert hasattr(adapter, "get_used_features")


# =============================================================================
# Test: Version Detection
# =============================================================================

class TestVersionDetection:
    """Tests for version detection"""

    def test_detect_all_versions(self):
        """detect_all_versions should return list of tuples"""
        results = detect_all_versions()

        assert isinstance(results, list)
        for item in results:
            assert len(item) == 3
            version, can_run, reason = item
            assert isinstance(version, str)
            assert isinstance(can_run, bool)
            assert isinstance(reason, str)


# =============================================================================
# Test: Index JSON Format
# =============================================================================

class TestIndexJsonFormat:
    """Tests for index.json output format"""

    def test_index_json_structure(self, tmp_path):
        """Index JSON should have correct structure"""
        from scripts.audit_featurepacks import write_index

        results = [
            {
                "version": "v4",
                "mode": "pre_race",
                "target": "target_win",
                "status": "OK",
                "reason": "",
                "n_features": 100,
                "output_path": "artifacts/feature_audit/v4/pre_race/target_win",
                "timestamp": "2024-01-01T00:00:00",
            },
            {
                "version": "legacy",
                "mode": "pre_race",
                "target": "target_win",
                "status": "SKIP",
                "reason": "No feature table found",
                "output_path": None,
                "timestamp": "2024-01-01T00:00:00",
            },
        ]

        write_index(tmp_path, results)

        index_path = tmp_path / "index.json"
        assert index_path.exists()

        with open(index_path, "r") as f:
            index = json.load(f)

        assert "generated_at" in index
        assert "n_runs" in index
        assert "runs" in index
        assert index["n_runs"] == 2

    def test_index_json_status_values(self, tmp_path):
        """Index should support OK/SKIP/FAIL statuses"""
        from scripts.audit_featurepacks import write_index

        results = [
            {"version": "v4", "mode": "pre_race", "target": "target_win",
             "status": "OK", "reason": "", "output_path": "path", "timestamp": "t"},
            {"version": "v3", "mode": "pre_race", "target": "target_win",
             "status": "SKIP", "reason": "Not found", "output_path": None, "timestamp": "t"},
            {"version": "v2", "mode": "pre_race", "target": "target_win",
             "status": "FAIL", "reason": "Error", "output_path": None, "timestamp": "t"},
        ]

        write_index(tmp_path, results)

        with open(tmp_path / "index.json", "r") as f:
            index = json.load(f)

        statuses = [r["status"] for r in index["runs"]]
        assert "OK" in statuses
        assert "SKIP" in statuses
        assert "FAIL" in statuses


# =============================================================================
# Test: Output Files Format
# =============================================================================

class TestOutputFilesFormat:
    """Tests for output file formats"""

    def test_used_features_txt(self, tmp_path):
        """used_features.txt should be one feature per line"""
        from scripts.audit_featurepacks import write_used_features

        features = ["feature_a", "feature_b", "feature_c"]
        write_used_features(tmp_path, features)

        path = tmp_path / "used_features.txt"
        assert path.exists()

        with open(path, "r") as f:
            lines = f.read().strip().split("\n")

        assert len(lines) == 3
        assert lines[0] == "feature_a"

    def test_feature_inventory_json(self, tmp_path):
        """feature_inventory.json should have safety labels"""
        from scripts.audit_featurepacks import write_feature_inventory

        features = ["h_body_weight", "race_distance"]
        feature_stats = {
            "h_body_weight": {"dtype": "float64", "missing_rate": 0.1},
            "race_distance": {"dtype": "int64", "missing_rate": 0.0},
        }

        write_feature_inventory(tmp_path, features, feature_stats)

        path = tmp_path / "feature_inventory.json"
        assert path.exists()

        with open(path, "r") as f:
            inventory = json.load(f)

        assert len(inventory) == 2

        # Check first feature (unsafe)
        assert inventory[0]["feature"] == "h_body_weight"
        assert inventory[0]["safety_label"] == "unsafe"

        # Check second feature (safe)
        assert inventory[1]["feature"] == "race_distance"
        assert inventory[1]["safety_label"] == "safe"

    def test_importance_csv(self, tmp_path):
        """importance CSV should be sorted by value"""
        from scripts.audit_featurepacks import write_importance_csv

        importance = {"feature_a": 100.0, "feature_b": 50.0, "feature_c": 200.0}
        write_importance_csv(tmp_path, importance, "importance_gain.csv")

        path = tmp_path / "importance_gain.csv"
        assert path.exists()

        with open(path, "r") as f:
            lines = f.read().strip().split("\n")

        assert lines[0] == "feature,importance"
        # Should be sorted descending
        assert lines[1].startswith("feature_c,200")
        assert lines[2].startswith("feature_a,100")
        assert lines[3].startswith("feature_b,50")


# =============================================================================
# Test: Path Handling (OS Independence)
# =============================================================================

class TestPathHandling:
    """Tests for OS-independent path handling"""

    def test_path_as_posix(self):
        """Paths should be converted to POSIX format for JSON"""
        path = Path("artifacts") / "feature_audit" / "v4"
        posix = path.as_posix()

        assert "\\" not in posix
        assert "/" in posix

    def test_path_comparison(self):
        """Path comparisons should work cross-platform"""
        path1 = Path("artifacts/feature_audit/v4")
        path2 = Path("artifacts") / "feature_audit" / "v4"

        assert path1 == path2

    def test_output_path_in_index(self, tmp_path):
        """output_path in index should use forward slashes"""
        from scripts.audit_featurepacks import write_index

        results = [
            {
                "version": "v4",
                "mode": "pre_race",
                "target": "target_win",
                "status": "OK",
                "reason": "",
                "output_path": "artifacts/feature_audit/v4/pre_race/target_win",
                "timestamp": "2024-01-01T00:00:00",
            },
        ]

        write_index(tmp_path, results)

        with open(tmp_path / "index.json", "r") as f:
            index = json.load(f)

        output_path = index["runs"][0]["output_path"]
        assert "\\" not in output_path


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
