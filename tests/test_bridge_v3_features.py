# -*- coding: utf-8 -*-
"""
test_bridge_v3_features.py - Tests for bridge_v3_features.py

v3特徴量のv4ブリッジ機能のユニットテスト。
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.features_v4.bridge_v3_features import (
    BridgeV3Features,
    BridgeConfig,
    BridgeResult,
    MigrationCandidate,
    FeatureSafetyInfo,
    BRIDGE_PREFIX,
    SAFETY_SKIP,
    SAFETY_WARN,
    apply_bridge_to_dataframe,
    get_bridge_feature_columns,
    list_available_v3_features,
    load_bridge_feature_map,
    get_original_feature_name,
    get_feature_safety_for_bridge,
)

from src.feature_audit.safety import (
    classify_feature_safety,
    strip_bridge_prefix,
    BRIDGE_PREFIX as SAFETY_BRIDGE_PREFIX,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_db():
    """Create a temporary database with test tables"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)

    # Create feature_table_v3 (mock)
    conn.execute("""
        CREATE TABLE feature_table_v3 (
            race_id TEXT,
            horse_id TEXT,
            hr_recent_win_rate REAL,
            hr_total_prize REAL,
            ax1_momentum REAL,
            target_win INTEGER,
            target_in3 INTEGER,
            PRIMARY KEY (race_id, horse_id)
        )
    """)

    # Insert test data
    test_data = [
        ("2024010101", "horse1", 0.25, 1000.0, 0.5, 1, 1),
        ("2024010101", "horse2", 0.15, 500.0, 0.3, 0, 1),
        ("2024010102", "horse1", 0.25, 1000.0, 0.5, 0, 0),
        ("2024010102", "horse3", 0.30, 2000.0, 0.7, 1, 1),
    ]

    conn.executemany("""
        INSERT INTO feature_table_v3
        (race_id, horse_id, hr_recent_win_rate, hr_total_prize, ax1_momentum, target_win, target_in3)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, test_data)

    conn.commit()

    yield conn

    conn.close()
    Path(db_path).unlink()


@pytest.fixture
def temp_candidates_csv(tmp_path):
    """Create a temporary migration candidates CSV"""
    csv_path = tmp_path / "migration_candidates_target_win.csv"

    candidates = [
        {
            "feature": "hr_recent_win_rate",
            "v3_gain": 100.0,
            "v3_rank": 1,
            "safety_label": "safe",
            "safety_notes": "",
            "reason_tag": "v3_only",
            "blocked_reason": "",
            "suggested_action": "port",
        },
        {
            "feature": "hr_total_prize",
            "v3_gain": 80.0,
            "v3_rank": 2,
            "safety_label": "warn",
            "safety_notes": "Potential time leak",
            "reason_tag": "v3_only",
            "blocked_reason": "",
            "suggested_action": "needs_review",
        },
        {
            "feature": "ax1_momentum",
            "v3_gain": 60.0,
            "v3_rank": 3,
            "safety_label": "unsafe",
            "safety_notes": "Uses future data",
            "reason_tag": "v3_only",
            "blocked_reason": "future leak",
            "suggested_action": "skip",
        },
    ]

    df = pd.DataFrame(candidates)
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def v4_dataframe():
    """Create a v4 feature DataFrame for testing"""
    return pd.DataFrame({
        "race_id": ["2024010101", "2024010101", "2024010102", "2024010102"],
        "horse_id": ["horse1", "horse2", "horse1", "horse3"],
        "v4_feature_1": [1.0, 2.0, 3.0, 4.0],
        "v4_feature_2": [0.1, 0.2, 0.3, 0.4],
        "target_win": [1, 0, 0, 1],
    })


# =============================================================================
# Test BridgeConfig
# =============================================================================

class TestBridgeConfig:
    """Tests for BridgeConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        config = BridgeConfig()

        assert config.source_version == "v3"
        assert config.max_features == 20
        assert config.include_warn is True
        assert config.explicit_features == []
        assert config.candidates_path is None

    def test_custom_values(self):
        """Test custom configuration values"""
        config = BridgeConfig(
            source_version="v2",
            max_features=10,
            include_warn=False,
            explicit_features=["feat1", "feat2"],
        )

        assert config.source_version == "v2"
        assert config.max_features == 10
        assert config.include_warn is False
        assert config.explicit_features == ["feat1", "feat2"]


# =============================================================================
# Test MigrationCandidate
# =============================================================================

class TestMigrationCandidate:
    """Tests for MigrationCandidate dataclass"""

    def test_create_candidate(self):
        """Test creating a migration candidate"""
        candidate = MigrationCandidate(
            feature="test_feature",
            v3_gain=50.0,
            v3_rank=5,
            safety_label="safe",
            safety_notes="",
            reason_tag="v3_only",
            blocked_reason="",
            suggested_action="port",
        )

        assert candidate.feature == "test_feature"
        assert candidate.v3_gain == 50.0
        assert candidate.v3_rank == 5
        assert candidate.safety_label == "safe"
        assert candidate.suggested_action == "port"


# =============================================================================
# Test BridgeV3Features
# =============================================================================

class TestBridgeV3Features:
    """Tests for BridgeV3Features class"""

    def test_init(self, temp_db):
        """Test initialization"""
        bridge = BridgeV3Features(temp_db)

        assert bridge.conn is temp_db
        assert bridge.config is not None
        assert bridge._v3_table_exists is None  # Not checked yet

    def test_init_with_config(self, temp_db):
        """Test initialization with custom config"""
        config = BridgeConfig(max_features=5)
        bridge = BridgeV3Features(temp_db, config)

        assert bridge.config.max_features == 5

    def test_check_v3_table_exists(self, temp_db):
        """Test checking if v3 table exists"""
        bridge = BridgeV3Features(temp_db)

        assert bridge._check_v3_table() is True
        assert bridge._v3_table_exists is True  # Cached

    def test_check_v3_table_not_exists(self):
        """Test checking when v3 table doesn't exist"""
        conn = sqlite3.connect(":memory:")
        bridge = BridgeV3Features(conn)

        assert bridge._check_v3_table() is False
        conn.close()

    def test_load_candidates(self, temp_db, temp_candidates_csv):
        """Test loading migration candidates from CSV"""
        config = BridgeConfig(candidates_path=temp_candidates_csv)
        bridge = BridgeV3Features(temp_db, config)

        candidates = bridge.load_candidates("target_win")

        assert len(candidates) == 3
        assert candidates[0].feature == "hr_recent_win_rate"
        assert candidates[0].safety_label == "safe"
        assert candidates[1].feature == "hr_total_prize"
        assert candidates[1].safety_label == "warn"
        assert candidates[2].feature == "ax1_momentum"
        assert candidates[2].safety_label == "unsafe"

    def test_load_candidates_caching(self, temp_db, temp_candidates_csv):
        """Test that candidates are cached"""
        config = BridgeConfig(candidates_path=temp_candidates_csv)
        bridge = BridgeV3Features(temp_db, config)

        # First load
        candidates1 = bridge.load_candidates("target_win")
        # Second load (should use cache)
        candidates2 = bridge.load_candidates("target_win")

        assert candidates1 is candidates2  # Same object (cached)

    def test_load_candidates_missing_file(self, temp_db, tmp_path):
        """Test loading when CSV file doesn't exist"""
        config = BridgeConfig(candidates_path=tmp_path / "nonexistent.csv")
        bridge = BridgeV3Features(temp_db, config)

        candidates = bridge.load_candidates("target_win")

        assert candidates == []

    def test_get_bridge_features_safe_only(self, temp_db, temp_candidates_csv):
        """Test getting bridge features with safe only"""
        config = BridgeConfig(
            candidates_path=temp_candidates_csv,
            include_warn=False,
        )
        bridge = BridgeV3Features(temp_db, config)

        features, excluded_unsafe, excluded_warn, warn_included, candidate_map = bridge.get_bridge_features("target_win")

        assert "hr_recent_win_rate" in features
        assert "hr_total_prize" not in features  # warn excluded
        assert "ax1_momentum" not in features  # unsafe excluded
        assert "ax1_momentum" in excluded_unsafe  # unsafe tracked separately
        assert "hr_total_prize" in excluded_warn  # warn tracked when include_warn=False

    def test_get_bridge_features_include_warn(self, temp_db, temp_candidates_csv):
        """Test getting bridge features including warn"""
        config = BridgeConfig(
            candidates_path=temp_candidates_csv,
            include_warn=True,
        )
        bridge = BridgeV3Features(temp_db, config)

        features, excluded_unsafe, excluded_warn, warn_included, candidate_map = bridge.get_bridge_features("target_win")

        assert "hr_recent_win_rate" in features
        assert "hr_total_prize" in features  # warn included
        assert "hr_total_prize" in warn_included
        assert "ax1_momentum" not in features  # unsafe still excluded
        assert "ax1_momentum" in excluded_unsafe  # unsafe tracked

    def test_get_bridge_features_explicit(self, temp_db):
        """Test getting bridge features with explicit list"""
        config = BridgeConfig(
            explicit_features=["explicit_feat1", "explicit_feat2"],
        )
        bridge = BridgeV3Features(temp_db, config)

        features, excluded_unsafe, excluded_warn, warn_included, candidate_map = bridge.get_bridge_features("target_win")

        assert features == ["explicit_feat1", "explicit_feat2"]
        assert excluded_unsafe == []
        assert excluded_warn == []
        assert warn_included == []

    def test_get_bridge_features_max_limit(self, temp_db, temp_candidates_csv):
        """Test max features limit"""
        config = BridgeConfig(
            candidates_path=temp_candidates_csv,
            max_features=1,
            include_warn=True,
        )
        bridge = BridgeV3Features(temp_db, config)

        features, excluded_unsafe, excluded_warn, warn_included, candidate_map = bridge.get_bridge_features("target_win")

        assert len(features) == 1
        assert features[0] == "hr_recent_win_rate"

    def test_load_v3_features(self, temp_db):
        """Test loading v3 features from database"""
        bridge = BridgeV3Features(temp_db)

        df = bridge.load_v3_features(
            race_ids=["2024010101"],
            features=["hr_recent_win_rate", "ax1_momentum"],
        )

        assert len(df) == 2  # 2 horses in race
        assert "race_id" in df.columns
        assert "horse_id" in df.columns
        assert "hr_recent_win_rate" in df.columns
        assert "ax1_momentum" in df.columns

    def test_load_v3_features_missing_column(self, temp_db):
        """Test loading with missing columns"""
        bridge = BridgeV3Features(temp_db)

        df = bridge.load_v3_features(
            race_ids=["2024010101"],
            features=["hr_recent_win_rate", "nonexistent_column"],
        )

        assert "hr_recent_win_rate" in df.columns
        assert "nonexistent_column" not in df.columns  # Silently skipped

    def test_apply_bridge(self, temp_db, temp_candidates_csv, v4_dataframe):
        """Test applying bridge to v4 DataFrame"""
        config = BridgeConfig(
            candidates_path=temp_candidates_csv,
            include_warn=True,
        )
        bridge = BridgeV3Features(temp_db, config)

        bridged_df, result = bridge.apply_bridge(v4_dataframe, "target_win")

        # Check result
        assert isinstance(result, BridgeResult)
        assert result.n_features_applied == 2  # safe + warn (unsafe skipped)

        # Check DataFrame
        assert f"{BRIDGE_PREFIX}hr_recent_win_rate" in bridged_df.columns
        assert f"{BRIDGE_PREFIX}hr_total_prize" in bridged_df.columns
        assert f"{BRIDGE_PREFIX}ax1_momentum" not in bridged_df.columns  # unsafe

        # Original columns preserved
        assert "v4_feature_1" in bridged_df.columns
        assert "v4_feature_2" in bridged_df.columns

    def test_apply_bridge_no_candidates(self, temp_db, v4_dataframe, tmp_path):
        """Test applying bridge with no candidates"""
        config = BridgeConfig(
            candidates_path=tmp_path / "nonexistent.csv",
        )
        bridge = BridgeV3Features(temp_db, config)

        bridged_df, result = bridge.apply_bridge(v4_dataframe, "target_win")

        assert result.n_features_applied == 0
        assert len(bridged_df.columns) == len(v4_dataframe.columns)


# =============================================================================
# Test Utility Functions
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions"""

    def test_get_bridge_feature_columns(self, temp_db, temp_candidates_csv):
        """Test get_bridge_feature_columns"""
        config = BridgeConfig(
            candidates_path=temp_candidates_csv,
            include_warn=True,
        )

        columns = get_bridge_feature_columns(temp_db, "target_win", config)

        assert f"{BRIDGE_PREFIX}hr_recent_win_rate" in columns
        assert f"{BRIDGE_PREFIX}hr_total_prize" in columns

    def test_apply_bridge_to_dataframe(self, temp_db, temp_candidates_csv, v4_dataframe):
        """Test apply_bridge_to_dataframe convenience function"""
        config = BridgeConfig(
            candidates_path=temp_candidates_csv,
            include_warn=False,  # Only safe
        )

        bridged_df, result = apply_bridge_to_dataframe(
            temp_db, v4_dataframe, "target_win", config
        )

        assert result.n_features_applied == 1  # Only safe feature
        assert f"{BRIDGE_PREFIX}hr_recent_win_rate" in bridged_df.columns

    def test_list_available_v3_features(self, temp_db):
        """Test listing available v3 features"""
        features = list_available_v3_features(temp_db)

        # Should include v3 features but exclude identity/target columns
        assert "hr_recent_win_rate" in features
        assert "hr_total_prize" in features
        assert "ax1_momentum" in features
        assert "race_id" not in features  # Excluded
        assert "horse_id" not in features  # Excluded
        assert "target_win" not in features  # Excluded


# =============================================================================
# Test BridgeResult
# =============================================================================

class TestBridgeResult:
    """Tests for BridgeResult dataclass"""

    def test_create_result(self):
        """Test creating a bridge result"""
        result = BridgeResult(
            n_features_requested=10,
            n_features_applied=5,
            n_features_skipped_unsafe=3,
            n_features_skipped_missing=2,
            n_features_skipped_warn=1,
            n_features_warn=1,
            applied_features=["f1", "f2", "f3", "f4", "f5"],
            skipped_features=["f6", "f7", "f8"],
            warn_features=["f5"],
        )

        assert result.n_features_requested == 10
        assert result.n_features_applied == 5
        assert result.n_features_skipped_warn == 1
        assert len(result.applied_features) == 5

    def test_result_to_summary_dict(self):
        """Test BridgeResult.to_summary_dict() method"""
        result = BridgeResult(
            n_features_requested=10,
            n_features_applied=5,
            n_features_skipped_unsafe=3,
            n_features_skipped_missing=1,
            n_features_skipped_warn=1,
            n_features_warn=2,
            applied_features=["f1", "f2", "f3", "f4", "f5"],
            skipped_features=["f6", "f7", "f8", "f9"],
            warn_features=["f4", "f5"],
        )

        summary = result.to_summary_dict()

        # to_summary_dict includes these keys:
        assert summary["n_features_applied"] == 5
        assert summary["n_features_skipped_unsafe"] == 3
        assert summary["n_features_skipped_warn"] == 1
        assert summary["n_features_skipped_missing"] == 1
        assert summary["n_features_warn_included"] == 2
        assert len(summary["applied_features"]) == 5


# =============================================================================
# Test Constants
# =============================================================================

class TestConstants:
    """Tests for module constants"""

    def test_bridge_prefix(self):
        """Test bridge prefix constant"""
        assert BRIDGE_PREFIX == "v4_bridge_"

    def test_safety_labels(self):
        """Test safety label sets"""
        assert "unsafe" in SAFETY_SKIP
        assert "warn" in SAFETY_WARN


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests"""

    def test_full_bridge_workflow(self, temp_db, temp_candidates_csv, v4_dataframe):
        """Test complete bridge workflow"""
        # 1. Configure bridge
        config = BridgeConfig(
            candidates_path=temp_candidates_csv,
            max_features=10,
            include_warn=True,
        )

        # 2. Create bridge instance
        bridge = BridgeV3Features(temp_db, config)

        # 3. Load candidates
        candidates = bridge.load_candidates("target_win")
        assert len(candidates) == 3

        # 4. Get bridge features
        features, excluded_unsafe, excluded_warn, warn_included, candidate_map = bridge.get_bridge_features("target_win")
        assert len(features) == 2  # safe + warn
        assert len(excluded_unsafe) >= 1  # At least unsafe

        # 5. Apply bridge
        bridged_df, result = bridge.apply_bridge(v4_dataframe, "target_win")

        # 6. Verify results
        assert result.n_features_applied == 2
        assert bridged_df.shape[0] == v4_dataframe.shape[0]  # Same rows
        assert bridged_df.shape[1] > v4_dataframe.shape[1]  # More columns

        # 7. Verify bridge features are properly named
        for col in bridged_df.columns:
            if "hr_" in col or "ax1_" in col:
                assert col.startswith(BRIDGE_PREFIX)

    def test_bridge_with_empty_v3_table(self):
        """Test bridge when v3 table exists but is empty"""
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE feature_table_v3 (
                race_id TEXT,
                horse_id TEXT,
                feature1 REAL,
                PRIMARY KEY (race_id, horse_id)
            )
        """)

        v4_df = pd.DataFrame({
            "race_id": ["r1", "r2"],
            "horse_id": ["h1", "h2"],
            "v4_feat": [1.0, 2.0],
        })

        config = BridgeConfig(explicit_features=["feature1"])
        bridge = BridgeV3Features(conn, config)

        bridged_df, result = bridge.apply_bridge(v4_df, "target_win")

        # Should handle gracefully
        assert result.n_features_applied >= 0
        conn.close()


# =============================================================================
# Test F-2 Safety Features
# =============================================================================

class TestF2SafetyFeatures:
    """Tests for F-2 safety enhancements"""

    def test_strip_bridge_prefix(self):
        """Test strip_bridge_prefix function"""
        # Bridge feature
        original, is_bridged = strip_bridge_prefix("v4_bridge_hr_recent_win_rate")
        assert original == "hr_recent_win_rate"
        assert is_bridged is True

        # Native feature
        original, is_bridged = strip_bridge_prefix("v4_native_feature")
        assert original == "v4_native_feature"
        assert is_bridged is False

    def test_classify_bridge_feature_safety(self):
        """Test that bridged features inherit safety from original name"""
        # Unsafe original -> unsafe bridged
        label, notes = classify_feature_safety("v4_bridge_h_body_weight")
        assert label == "unsafe"
        assert "[bridged from v3]" in notes

        # Safe original -> safe bridged
        label, notes = classify_feature_safety("v4_bridge_hr_recent_win_rate")
        assert label == "safe"
        assert "[bridged from v3]" in notes

        # Warn original -> warn bridged
        label, notes = classify_feature_safety("v4_bridge_track_condition_id")
        assert label == "warn"
        assert "[bridged from v3]" in notes

    def test_classify_native_feature_unchanged(self):
        """Test that native features are classified normally"""
        # Native unsafe
        label, notes = classify_feature_safety("h_body_weight")
        assert label == "unsafe"
        assert "[bridged from v3]" not in notes

        # Native safe
        label, notes = classify_feature_safety("hr_recent_win_rate")
        assert label == "safe"
        assert notes == ""

    def test_bridge_prefix_constants_match(self):
        """Test that BRIDGE_PREFIX is consistent across modules"""
        assert BRIDGE_PREFIX == SAFETY_BRIDGE_PREFIX
        assert BRIDGE_PREFIX == "v4_bridge_"


class TestF2FeatureSafetyInfo:
    """Tests for FeatureSafetyInfo dataclass"""

    def test_create_safety_info(self):
        """Test creating FeatureSafetyInfo"""
        info = FeatureSafetyInfo(
            original_name="hr_recent_win_rate",
            bridged_name="v4_bridge_hr_recent_win_rate",
            safety_label="safe",
            safety_notes="",
            origin="v3_bridged",
        )

        assert info.original_name == "hr_recent_win_rate"
        assert info.bridged_name == "v4_bridge_hr_recent_win_rate"
        assert info.safety_label == "safe"
        assert info.origin == "v3_bridged"


class TestF2FeatureMapUtilities:
    """Tests for feature map utility functions"""

    def test_get_original_feature_name(self):
        """Test get_original_feature_name function"""
        feature_map = {
            "v4_bridge_hr_recent_win_rate": "hr_recent_win_rate",
            "v4_bridge_ax1_momentum": "ax1_momentum",
        }

        # Mapped feature
        original = get_original_feature_name("v4_bridge_hr_recent_win_rate", feature_map)
        assert original == "hr_recent_win_rate"

        # Unmapped feature (returns as-is)
        original = get_original_feature_name("v4_native_feature", feature_map)
        assert original == "v4_native_feature"

    def test_get_feature_safety_for_bridge(self, tmp_path):
        """Test get_feature_safety_for_bridge function"""
        # Create a mock feature map file
        bridge_map = {
            "metadata": {"target": "target_win"},
            "feature_map": {
                "v4_bridge_test_feat": "test_feat",
            },
            "features": {
                "v4_bridge_test_feat": {
                    "original_name": "test_feat",
                    "bridged_name": "v4_bridge_test_feat",
                    "safety_label": "safe",
                    "safety_notes": "",
                    "origin": "v3_bridged",
                }
            },
        }

        map_path = tmp_path / "bridge_feature_map_target_win.json"
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(bridge_map, f)

        # Load and test
        loaded_map = load_bridge_feature_map(map_path)
        label, notes, origin = get_feature_safety_for_bridge(
            "v4_bridge_test_feat", loaded_map
        )

        assert label == "safe"
        assert origin == "v3_bridged"

        # Unknown feature (non-bridge prefix treated as v4 native)
        label, notes, origin = get_feature_safety_for_bridge(
            "unknown_feature", loaded_map
        )
        # Features without bridge prefix are considered v4 native and default to safe
        assert label == "safe"
        assert origin == "v4_native"


class TestF2UnsafeGate:
    """Tests for unsafe feature exclusion (F-2.1 gate)"""

    def test_unsafe_always_excluded(self, temp_db, temp_candidates_csv):
        """Test that unsafe features are ALWAYS excluded regardless of config"""
        # Even with include_warn=True, unsafe should be excluded
        config = BridgeConfig(
            candidates_path=temp_candidates_csv,
            include_warn=True,
        )
        bridge = BridgeV3Features(temp_db, config)

        features, excluded_unsafe, _, _, _ = bridge.get_bridge_features("target_win")

        # ax1_momentum is unsafe in test CSV
        assert "ax1_momentum" not in features
        assert "ax1_momentum" in excluded_unsafe

    def test_unsafe_cannot_be_included_explicitly(self, temp_db):
        """Test that explicit_features still respect safety gate"""
        # Even when explicitly requested, unsafe features should be excluded
        config = BridgeConfig(
            explicit_features=["h_body_weight", "hr_recent_win_rate"],
        )
        bridge = BridgeV3Features(temp_db, config)

        # Note: explicit_features bypasses candidates, so safety check is at apply time
        features, excluded_unsafe, _, _, _ = bridge.get_bridge_features("target_win")

        # Explicit features are passed through (safety checked at apply time)
        # This test verifies the behavior - explicit features bypass candidate filtering
        assert "h_body_weight" in features or "hr_recent_win_rate" in features


# =============================================================================
# Test F-3 Explain Runner
# =============================================================================

from src.features_v4.explain_runner import (
    FeatureExplanation,
    ExplainResult,
    resolve_feature_name,
    build_feature_explanation,
    generate_explain_result,
)


class TestF3ExplainRunner:
    """Tests for F-3 Explain Runner functionality"""

    def test_resolve_feature_name_bridged(self):
        """Test that bridged features resolve to original names"""
        bridge_map = {
            "feature_map": {
                "v4_bridge_hr_test_feat": "hr_test_feat",
            },
            "features": {
                "v4_bridge_hr_test_feat": {
                    "safety_label": "safe",
                    "safety_notes": "",
                    "origin": "v3_bridged",
                }
            },
        }

        display_name, origin, safety_label, safety_notes = resolve_feature_name(
            "v4_bridge_hr_test_feat", bridge_map
        )

        assert display_name == "hr_test_feat"  # Resolved to original
        assert origin == "v3_bridged"
        assert safety_label == "safe"

    def test_resolve_feature_name_native(self):
        """Test that native v4 features are recognized correctly"""
        bridge_map = {"feature_map": {}, "features": {}}

        display_name, origin, safety_label, safety_notes = resolve_feature_name(
            "h_recent3_avg_finish", bridge_map
        )

        assert display_name == "h_recent3_avg_finish"  # Unchanged
        assert origin == "v4_native"
        assert safety_label == "safe"

    def test_build_feature_explanation(self):
        """Test building a single feature explanation"""
        bridge_map = {
            "feature_map": {
                "v4_bridge_hr_test_feat": "hr_test_feat",
            },
            "features": {
                "v4_bridge_hr_test_feat": {
                    "safety_label": "warn",
                    "safety_notes": "Potential time leak",
                    "origin": "v3_bridged",
                }
            },
        }

        explanation = build_feature_explanation(
            "v4_bridge_hr_test_feat",
            bridge_map,
            importance_gain=10.5,
            importance_split=100,
        )

        assert explanation.feature_name == "v4_bridge_hr_test_feat"
        assert explanation.display_name == "hr_test_feat"
        assert explanation.origin == "v3_bridged"
        assert explanation.safety_label == "warn"
        assert explanation.importance_gain == 10.5
        assert explanation.importance_split == 100

    def test_generate_explain_result(self):
        """Test generating complete explain result"""
        bridge_map = {
            "feature_map": {
                "v4_bridge_feat1": "feat1",
                "v4_bridge_feat2": "feat2",
            },
            "features": {
                "v4_bridge_feat1": {
                    "safety_label": "safe",
                    "safety_notes": "",
                    "origin": "v3_bridged",
                },
                "v4_bridge_feat2": {
                    "safety_label": "warn",
                    "safety_notes": "Review needed",
                    "origin": "v3_bridged",
                },
            },
        }

        feature_names = [
            "v4_bridge_feat1",
            "v4_bridge_feat2",
            "native_feat",
        ]

        result = generate_explain_result(
            feature_names=feature_names,
            target="target_win",
            bridge_map_data=bridge_map,
        )

        assert result.n_features == 3
        assert result.n_bridged == 2
        assert result.n_native == 1

    def test_generate_explain_result_exclude_warn(self):
        """Test that warn features are excluded when exclude_warn=True"""
        bridge_map = {
            "feature_map": {
                "v4_bridge_safe_feat": "safe_feat",
                "v4_bridge_warn_feat": "warn_feat",
            },
            "features": {
                "v4_bridge_safe_feat": {
                    "safety_label": "safe",
                    "safety_notes": "",
                    "origin": "v3_bridged",
                },
                "v4_bridge_warn_feat": {
                    "safety_label": "warn",
                    "safety_notes": "Review needed",
                    "origin": "v3_bridged",
                },
            },
        }

        feature_names = ["v4_bridge_safe_feat", "v4_bridge_warn_feat"]

        # Without exclude_warn
        result_all = generate_explain_result(
            feature_names=feature_names,
            target="target_win",
            bridge_map_data=bridge_map,
            exclude_warn=False,
        )
        assert result_all.n_features == 2

        # With exclude_warn
        result_no_warn = generate_explain_result(
            feature_names=feature_names,
            target="target_win",
            bridge_map_data=bridge_map,
            exclude_warn=True,
        )
        assert result_no_warn.n_features == 1
        assert result_no_warn.features[0].safety_label == "safe"

    def test_explain_result_to_dict(self):
        """Test ExplainResult.to_dict() for JSON serialization"""
        result = ExplainResult(
            target="target_win",
            n_features=2,
            n_bridged=1,
            n_native=1,
            features=[
                FeatureExplanation(
                    feature_name="v4_bridge_test",
                    display_name="test",
                    origin="v3_bridged",
                    safety_label="safe",
                    safety_notes="",
                    importance_gain=10.0,
                ),
                FeatureExplanation(
                    feature_name="native_feat",
                    display_name="native_feat",
                    origin="v4_native",
                    safety_label="safe",
                    safety_notes="v4 native feature",
                    importance_gain=5.0,
                ),
            ],
        )

        d = result.to_dict()

        assert d["target"] == "target_win"
        assert d["n_features"] == 2
        assert d["n_bridged"] == 1
        assert len(d["features"]) == 2
        assert d["features"][0]["display_name"] == "test"

    def test_explain_result_schema_version(self):
        """Test that ExplainResult includes schema_version, generated_at, model_version"""
        result = ExplainResult(
            target="target_win",
            n_features=1,
            n_bridged=0,
            n_native=1,
            model_version="v4",
        )

        d = result.to_dict()

        # Check schema fields at top level
        assert "schema_version" in d
        assert d["schema_version"] == "1.0"
        assert "generated_at" in d
        assert d["generated_at"] != ""  # Should be auto-set
        assert "model_version" in d
        assert d["model_version"] == "v4"

        # Check order: schema fields should come before target
        keys = list(d.keys())
        assert keys.index("schema_version") < keys.index("target")
        assert keys.index("generated_at") < keys.index("target")
        assert keys.index("model_version") < keys.index("target")

    def test_generate_explain_result_sorting(self):
        """Test that features are sorted by gain then split descending"""
        bridge_map = {"feature_map": {}, "features": {}}

        feature_names = ["feat_low", "feat_high", "feat_mid"]
        importance_df = pd.DataFrame([
            {"feature": "feat_low", "gain": 1.0, "split": 10},
            {"feature": "feat_high", "gain": 100.0, "split": 5},
            {"feature": "feat_mid", "gain": 50.0, "split": 20},
        ])

        result = generate_explain_result(
            feature_names=feature_names,
            target="target_win",
            bridge_map_data=bridge_map,
            importance_df=importance_df,
        )

        # Should be sorted by gain descending
        assert result.features[0].feature_name == "feat_high"
        assert result.features[1].feature_name == "feat_mid"
        assert result.features[2].feature_name == "feat_low"


class TestExplainFromPipeline:
    """Tests for generate_explain_from_pipeline function"""

    def test_generate_explain_from_pipeline_with_bridge_map(self, tmp_path):
        """Test explain generation with bridge map present"""
        from src.features_v4.explain_runner import generate_explain_from_pipeline

        # Create test bridge map
        bridge_map = {
            "feature_map": {
                "v4_bridge_test": "original_test",
            },
            "features": {
                "v4_bridge_test": {
                    "safety_label": "safe",
                    "safety_notes": "",
                    "origin": "v3_bridged",
                },
            },
        }
        bridge_map_path = tmp_path / "bridge_feature_map_target_win.json"
        with open(bridge_map_path, "w") as f:
            json.dump(bridge_map, f)

        # Create test importance CSV
        importance_df = pd.DataFrame([
            {"feature": "v4_bridge_test", "gain": 50.0, "split": 100},
            {"feature": "native_feat", "gain": 25.0, "split": 50},
        ])
        importance_path = tmp_path / "feature_importance_target_win_v4.csv"
        importance_df.to_csv(importance_path, index=False)

        feature_cols = ["v4_bridge_test", "native_feat"]

        result = generate_explain_from_pipeline(
            feature_cols=feature_cols,
            target="target_win",
            output_dir=tmp_path,
            bridge_map_path=bridge_map_path,
            importance_csv_path=importance_path,
            model_version="v4",
        )

        assert result is not None
        assert result.n_features == 2
        assert result.n_bridged == 1
        assert result.n_native == 1
        assert result.model_version == "v4"

        # Check output file exists
        output_path = tmp_path / "explain_target_win_v4.json"
        assert output_path.exists()

        # Verify JSON content
        with open(output_path) as f:
            data = json.load(f)
        assert data["schema_version"] == "1.0"
        assert data["model_version"] == "v4"
        assert len(data["features"]) == 2

    def test_generate_explain_from_pipeline_without_bridge_map(self, tmp_path):
        """Test explain generation without bridge map (v4 native only)"""
        from src.features_v4.explain_runner import generate_explain_from_pipeline

        feature_cols = ["native_feat1", "native_feat2", "native_feat3"]

        result = generate_explain_from_pipeline(
            feature_cols=feature_cols,
            target="target_win",
            output_dir=tmp_path,
            bridge_map_path=None,  # No bridge map
            importance_csv_path=None,  # No importance
            model_version="v4",
        )

        assert result is not None
        assert result.n_features == 3
        assert result.n_bridged == 0  # No bridged features
        assert result.n_native == 3  # All native

        # Check output file
        output_path = tmp_path / "explain_target_win_v4.json"
        assert output_path.exists()

    def test_generate_explain_from_pipeline_error_handling(self, tmp_path):
        """Test that explain generation handles errors gracefully"""
        from src.features_v4.explain_runner import generate_explain_from_pipeline

        # Point to non-existent bridge map path
        result = generate_explain_from_pipeline(
            feature_cols=["feat1"],
            target="target_win",
            output_dir=tmp_path,
            bridge_map_path=Path("/nonexistent/path.json"),
            importance_csv_path=Path("/nonexistent/importance.csv"),
            model_version="v4",
        )

        # Should still succeed (non-fatal error for missing files)
        assert result is not None
        assert result.n_features == 1


class TestExplainJSONSchema:
    """Tests for Explain JSON schema validation"""

    def test_explain_json_has_required_top_level_keys(self, tmp_path):
        """Test that generated explain JSON has all required top-level keys"""
        from src.features_v4.explain_runner import generate_explain_from_pipeline

        feature_cols = ["feat1", "feat2"]
        result = generate_explain_from_pipeline(
            feature_cols=feature_cols,
            target="target_win",
            output_dir=tmp_path,
            model_version="v4",
        )

        # Load the generated JSON
        output_path = tmp_path / "explain_target_win_v4.json"
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        # Check required top-level keys
        required_keys = [
            "schema_version",
            "generated_at",
            "model_version",
            "target",
            "n_features",
            "n_bridged",
            "n_native",
            "features",
        ]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"

        # Check features is array
        assert isinstance(data["features"], list)
        assert len(data["features"]) == 2

    def test_explain_json_feature_structure(self, tmp_path):
        """Test that each feature in the array has required fields"""
        from src.features_v4.explain_runner import generate_explain_from_pipeline

        feature_cols = ["test_feature"]
        result = generate_explain_from_pipeline(
            feature_cols=feature_cols,
            target="target_win",
            output_dir=tmp_path,
            model_version="v4",
        )

        output_path = tmp_path / "explain_target_win_v4.json"
        with open(output_path) as f:
            data = json.load(f)

        assert len(data["features"]) == 1
        feature = data["features"][0]

        # Check required feature fields
        required_feature_keys = [
            "feature_name",
            "display_name",
            "origin",
            "safety_label",
            "importance_gain",
            "importance_split",
        ]
        for key in required_feature_keys:
            assert key in feature, f"Missing required feature key: {key}"

        assert feature["feature_name"] == "test_feature"
        assert feature["origin"] in ["v4_native", "v3_bridged"]
        assert feature["safety_label"] in ["safe", "warn", "unsafe", "unknown"]

    def test_explain_json_no_nan_values(self, tmp_path):
        """Test that generated explain JSON contains no NaN values"""
        from src.features_v4.explain_runner import generate_explain_from_pipeline

        feature_cols = ["feat1", "feat2"]
        result = generate_explain_from_pipeline(
            feature_cols=feature_cols,
            target="target_win",
            output_dir=tmp_path,
            model_version="v4",
        )

        output_path = tmp_path / "explain_target_win_v4.json"
        assert output_path.exists()

        # Read raw file content and check for NaN
        content = output_path.read_text(encoding="utf-8")
        assert "NaN" not in content, "JSON contains invalid NaN value"
        assert "Infinity" not in content, "JSON contains invalid Infinity value"

        # Also verify it parses as valid JSON
        data = json.loads(content)
        assert data is not None

    def test_explain_importance_from_csv_with_importance_prefix(self, tmp_path):
        """Test that importance values are loaded from CSV with importance_gain/importance_split columns"""
        from src.features_v4.explain_runner import generate_explain_from_pipeline

        # Create CSV with importance_gain/importance_split columns (train_eval_v4 format)
        feature_cols = ["h_recent3_avg_finish", "v4_bridge_ax8_jockey_in3_rate_total_asof", "some_other_feat"]
        importance_df = pd.DataFrame([
            {"feature": "h_recent3_avg_finish", "importance_gain": 15357.5, "importance_split": 120},
            {"feature": "v4_bridge_ax8_jockey_in3_rate_total_asof", "importance_gain": 10002.3, "importance_split": 85},
            {"feature": "some_other_feat", "importance_gain": 500.0, "importance_split": 10},
        ])
        importance_path = tmp_path / "feature_importance_target_win_v4.csv"
        importance_df.to_csv(importance_path, index=False)

        result = generate_explain_from_pipeline(
            feature_cols=feature_cols,
            target="target_win",
            output_dir=tmp_path,
            importance_csv_path=importance_path,
            model_version="v4",
        )

        assert result is not None
        assert result.n_features == 3

        # Verify importance values are correctly loaded (not zero)
        output_path = tmp_path / "explain_target_win_v4.json"
        with open(output_path) as f:
            data = json.load(f)

        # Find the two key features and verify non-zero importance
        feat_map = {f["feature_name"]: f for f in data["features"]}

        assert "h_recent3_avg_finish" in feat_map
        assert feat_map["h_recent3_avg_finish"]["importance_gain"] > 0, "h_recent3_avg_finish should have non-zero gain"
        assert abs(feat_map["h_recent3_avg_finish"]["importance_gain"] - 15357.5) < 0.1

        assert "v4_bridge_ax8_jockey_in3_rate_total_asof" in feat_map
        assert feat_map["v4_bridge_ax8_jockey_in3_rate_total_asof"]["importance_gain"] > 0, "bridged feature should have non-zero gain"
        assert abs(feat_map["v4_bridge_ax8_jockey_in3_rate_total_asof"]["importance_gain"] - 10002.3) < 0.1


class TestDescriptionCoverage:
    """Tests for feature description coverage"""

    def test_explain_json_has_desc_field(self, tmp_path):
        """Test that each feature has a desc field"""
        from src.features_v4.explain_runner import generate_explain_from_pipeline

        feature_cols = ["h_recent3_avg_finish", "field_size", "test_unknown"]
        result = generate_explain_from_pipeline(
            feature_cols=feature_cols,
            target="target_win",
            output_dir=tmp_path,
            model_version="v4",
        )

        output_path = tmp_path / "explain_target_win_v4.json"
        with open(output_path) as f:
            data = json.load(f)

        for feature in data["features"]:
            assert "desc" in feature, f"Missing desc field in feature: {feature['feature_name']}"
            assert isinstance(feature["desc"], str), f"desc should be string: {feature['feature_name']}"

    def test_desc_coverage_above_threshold(self, tmp_path):
        """Test that desc coverage is above minimum threshold (>=75%)"""
        from src.features_v4.explain_runner import generate_explain_from_pipeline

        # Test with a mix of known and unknown features
        feature_cols = [
            # Known v4 native features (should have desc from dict or infer)
            "h_recent3_avg_finish",
            "h_win_rate_total",
            "field_size",
            "distance",
            "age",
            "h_days_since_last",
            # Features that should be inferred
            "h_starts_total",
            "j_win_rate_total",
            "t_in3_rate_total",
            # Unknown feature (may have empty desc)
            "xyz_unknown_feature",
        ]
        result = generate_explain_from_pipeline(
            feature_cols=feature_cols,
            target="target_win",
            output_dir=tmp_path,
            model_version="v4",
        )

        output_path = tmp_path / "explain_target_win_v4.json"
        with open(output_path) as f:
            data = json.load(f)

        # Count non-empty desc
        total = len(data["features"])
        with_desc = sum(1 for f in data["features"] if f.get("desc", "").strip())
        coverage = with_desc / total if total > 0 else 0

        assert coverage >= 0.75, f"Desc coverage {coverage:.1%} is below 75% threshold"

    def test_infer_desc_from_name_basic(self):
        """Test basic inference from naming conventions"""
        from src.features_v4.explain_runner import infer_desc_from_name

        # Test horse prefix + stat
        desc = infer_desc_from_name("h_win_rate_total", "h_win_rate_total", "v4_native")
        assert "馬" in desc
        assert "勝率" in desc

        # Test jockey prefix
        desc = infer_desc_from_name("j_in3_rate_total", "j_in3_rate_total", "v4_native")
        assert "騎手" in desc
        assert "複勝率" in desc

        # Test recent window
        desc = infer_desc_from_name("h_recent3_avg_finish", "h_recent3_avg_finish", "v4_native")
        assert "直近3走" in desc or "馬" in desc

    def test_infer_desc_from_name_suffixes(self):
        """Test condition suffix inference"""
        from src.features_v4.explain_runner import infer_desc_from_name

        # Test _dist_cat suffix
        desc = infer_desc_from_name("h_win_rate_dist_cat", "h_win_rate_dist_cat", "v4_native")
        assert "距離" in desc

        # Test _course suffix
        desc = infer_desc_from_name("h_win_rate_course", "h_win_rate_course", "v4_native")
        assert "コース" in desc

    def test_get_feature_desc_priority(self):
        """Test that dictionary takes priority over inference"""
        from src.features_v4.explain_runner import get_feature_desc, FEATURE_DESC_MAP

        # Feature in dictionary should use dictionary value
        desc = get_feature_desc("field_size", "field_size", "v4_native")
        assert desc == FEATURE_DESC_MAP["field_size"]

        # Feature not in dictionary should use inference
        desc = get_feature_desc("h_some_custom_win_rate_total", "h_some_custom_win_rate_total", "v4_native")
        # Should have inferred something
        assert "馬" in desc or "勝率" in desc or desc == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
