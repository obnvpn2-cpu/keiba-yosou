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
    BRIDGE_PREFIX,
    SAFETY_SKIP,
    SAFETY_WARN,
    apply_bridge_to_dataframe,
    get_bridge_feature_columns,
    list_available_v3_features,
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

        features, skipped, warn = bridge.get_bridge_features("target_win")

        assert "hr_recent_win_rate" in features
        assert "hr_total_prize" not in features  # warn excluded
        assert "ax1_momentum" not in features  # unsafe excluded
        assert "hr_total_prize" in skipped or "ax1_momentum" in skipped

    def test_get_bridge_features_include_warn(self, temp_db, temp_candidates_csv):
        """Test getting bridge features including warn"""
        config = BridgeConfig(
            candidates_path=temp_candidates_csv,
            include_warn=True,
        )
        bridge = BridgeV3Features(temp_db, config)

        features, skipped, warn = bridge.get_bridge_features("target_win")

        assert "hr_recent_win_rate" in features
        assert "hr_total_prize" in features  # warn included
        assert "hr_total_prize" in warn
        assert "ax1_momentum" not in features  # unsafe still excluded

    def test_get_bridge_features_explicit(self, temp_db):
        """Test getting bridge features with explicit list"""
        config = BridgeConfig(
            explicit_features=["explicit_feat1", "explicit_feat2"],
        )
        bridge = BridgeV3Features(temp_db, config)

        features, skipped, warn = bridge.get_bridge_features("target_win")

        assert features == ["explicit_feat1", "explicit_feat2"]
        assert skipped == []
        assert warn == []

    def test_get_bridge_features_max_limit(self, temp_db, temp_candidates_csv):
        """Test max features limit"""
        config = BridgeConfig(
            candidates_path=temp_candidates_csv,
            max_features=1,
            include_warn=True,
        )
        bridge = BridgeV3Features(temp_db, config)

        features, skipped, warn = bridge.get_bridge_features("target_win")

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
            n_features_warn=1,
            applied_features=["f1", "f2", "f3", "f4", "f5"],
            skipped_features=["f6", "f7", "f8"],
            warn_features=["f5"],
        )

        assert result.n_features_requested == 10
        assert result.n_features_applied == 5
        assert len(result.applied_features) == 5


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
        features, skipped, warn = bridge.get_bridge_features("target_win")
        assert len(features) == 2  # safe + warn
        assert len(skipped) >= 1  # At least unsafe

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
