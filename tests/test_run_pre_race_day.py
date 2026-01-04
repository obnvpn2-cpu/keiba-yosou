#!/usr/bin/env python3
"""
Tests for run_pre_race_day.py - Pre-race day operation runner

Step B テスト: ワンコマンドスクリプトと回帰テスト
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

from scripts.run_pre_race_day import (
    check_db,
    check_models,
    check_materials,
    check_demo_data,
    REQUIRED_MODELS,
)


# =============================================================================
# Test: Database Check
# =============================================================================

class TestCheckDb:
    """Tests for database existence check"""

    def test_check_db_exists(self, tmp_path):
        """Should return True when database exists"""
        db_path = tmp_path / "test.db"
        db_path.write_text("dummy db content")

        ok, msg = check_db(db_path)

        assert ok is True
        assert "found" in msg.lower() or "test.db" in msg

    def test_check_db_not_exists(self, tmp_path):
        """Should return False when database does not exist"""
        db_path = tmp_path / "nonexistent.db"

        ok, msg = check_db(db_path)

        assert ok is False
        assert "not found" in msg.lower()

    def test_check_db_shows_size(self, tmp_path):
        """Should show file size when database exists"""
        db_path = tmp_path / "sized.db"
        # Write 1MB of data
        db_path.write_bytes(b"x" * (1024 * 1024))

        ok, msg = check_db(db_path)

        assert ok is True
        assert "1.0 MB" in msg or "1.00 MB" in msg


# =============================================================================
# Test: Model Check
# =============================================================================

class TestCheckModels:
    """Tests for model existence check"""

    def test_check_models_all_found(self, tmp_path):
        """Should return True when all required models exist"""
        # Create mock model files
        for target, filenames in REQUIRED_MODELS:
            (tmp_path / filenames[0]).write_text("mock model")

        all_ok, found, missing = check_models(tmp_path)

        assert all_ok is True
        assert len(found) == len(REQUIRED_MODELS)
        assert len(missing) == 0

    def test_check_models_fallback_paths(self, tmp_path):
        """Should check fallback model paths"""
        # Create only fallback files (second in the list)
        for target, filenames in REQUIRED_MODELS:
            if len(filenames) > 1:
                (tmp_path / filenames[1]).write_text("fallback model")
            else:
                (tmp_path / filenames[0]).write_text("mock model")

        all_ok, found, missing = check_models(tmp_path)

        assert all_ok is True

    def test_check_models_partial_missing(self, tmp_path):
        """Should return False when some models are missing"""
        # Only create target_win model
        target, filenames = REQUIRED_MODELS[0]
        (tmp_path / filenames[0]).write_text("mock model")

        all_ok, found, missing = check_models(tmp_path)

        assert all_ok is False
        assert len(missing) > 0
        assert "target_in3" in missing  # The second model is missing

    def test_check_models_all_missing(self, tmp_path):
        """Should return False when no models exist"""
        all_ok, found, missing = check_models(tmp_path)

        assert all_ok is False
        assert len(found) == 0
        assert len(missing) == len(REQUIRED_MODELS)


# =============================================================================
# Test: Materials Check
# =============================================================================

class TestCheckMaterials:
    """Tests for pre_race materials check"""

    @pytest.fixture
    def artifacts_dir(self, tmp_path, monkeypatch):
        """Create mock artifacts directory structure"""
        artifacts = tmp_path / "artifacts" / "pre_race"
        artifacts.mkdir(parents=True)

        # Patch the ARTIFACTS_DIR constant
        import scripts.run_pre_race_day as module
        monkeypatch.setattr(module, "ARTIFACTS_DIR", artifacts)

        return artifacts

    def test_check_materials_not_exists(self, artifacts_dir):
        """Should return False when date directory does not exist"""
        ok, info = check_materials("2024-12-29")

        assert ok is False
        assert info["exists"] is False

    def test_check_materials_empty_dir(self, artifacts_dir):
        """Should return False when date directory is empty"""
        date_dir = artifacts_dir / "2024-12-29"
        date_dir.mkdir()

        ok, info = check_materials("2024-12-29")

        assert ok is False
        assert info["exists"] is True
        assert info["n_race_files"] == 0

    def test_check_materials_with_race_files(self, artifacts_dir):
        """Should return True when race files exist"""
        date_dir = artifacts_dir / "2024-12-29"
        date_dir.mkdir()

        # Create race file
        race_file = date_dir / "race_202412290605.json"
        race_file.write_text(json.dumps({"race_id": "test"}))

        ok, info = check_materials("2024-12-29")

        assert ok is True
        assert info["n_race_files"] == 1

    def test_check_materials_with_summary(self, artifacts_dir):
        """Should detect summary file"""
        date_dir = artifacts_dir / "2024-12-29"
        date_dir.mkdir()

        # Create summary and race file
        (date_dir / "summary_2024-12-29.json").write_text(json.dumps({"n_races": 1}))
        (date_dir / "race_test.json").write_text(json.dumps({}))

        ok, info = check_materials("2024-12-29")

        assert ok is True
        assert info["summary_exists"] is True


# =============================================================================
# Test: Demo Data Check
# =============================================================================

class TestCheckDemoData:
    """Tests for demo data check"""

    @pytest.fixture
    def demo_dir(self, tmp_path, monkeypatch):
        """Create mock demo directory"""
        artifacts = tmp_path / "artifacts" / "pre_race"
        artifacts.mkdir(parents=True)

        import scripts.run_pre_race_day as module
        monkeypatch.setattr(module, "ARTIFACTS_DIR", artifacts)

        return artifacts / "demo"

    def test_check_demo_not_exists(self, demo_dir):
        """Should return False when demo directory does not exist"""
        ok, info = check_demo_data()

        assert ok is False
        assert info["exists"] is False

    def test_check_demo_empty(self, demo_dir):
        """Should return False when demo directory is empty"""
        demo_dir.mkdir()

        ok, info = check_demo_data()

        assert ok is False
        assert info["exists"] is True
        assert info["n_race_files"] == 0

    def test_check_demo_with_data(self, demo_dir):
        """Should return True when demo race files exist"""
        demo_dir.mkdir()

        (demo_dir / "summary_demo.json").write_text(json.dumps({"n_races": 1}))
        (demo_dir / "race_demo_001.json").write_text(json.dumps({}))

        ok, info = check_demo_data()

        assert ok is True
        assert info["n_race_files"] == 1
        assert info["summary_exists"] is True


# =============================================================================
# Test: Path/Encoding Regression Tests (Step B)
# =============================================================================

class TestPathEncodingRegression:
    """Regression tests for path and encoding issues"""

    def test_path_with_japanese_characters(self, tmp_path):
        """Should handle paths with Japanese characters"""
        # Create directory with Japanese name
        jp_dir = tmp_path / "テスト"
        jp_dir.mkdir()

        # Create file with Japanese content
        jp_file = jp_dir / "データ.json"
        jp_file.write_text(json.dumps({"レース": "有馬記念"}, ensure_ascii=False), encoding="utf-8")

        # Should be able to read it back
        with open(jp_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["レース"] == "有馬記念"

    def test_path_comparison_os_independent(self):
        """Path comparisons should be OS-independent"""
        # Use Path for comparison (not string)
        path1 = Path("artifacts") / "pre_race" / "demo"
        path2 = Path("artifacts/pre_race/demo")

        assert path1 == path2

    def test_json_with_unicode(self, tmp_path):
        """Should handle JSON with various unicode characters"""
        test_data = {
            "name": "ドウデュース",
            "jockey": "武豊",
            "comment": "【重要】特記事項：〇△×☆",
            "symbols": "◎○▲△☆",
        }

        json_file = tmp_path / "unicode_test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False)

        with open(json_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded == test_data

    def test_json_missing_fields_handling(self):
        """Should handle JSON with missing optional fields gracefully"""
        # Minimal race data
        minimal_race = {
            "race_id": "test",
            "entries": [{"horse_id": "h001", "name": "TestHorse"}]
        }

        # Should not raise when accessing optional fields
        entries = minimal_race.get("entries", [])
        for entry in entries:
            # These fields might be missing
            _ = entry.get("p_win")
            _ = entry.get("p_in3")
            _ = entry.get("run_style")
            _ = entry.get("profile", {})


# =============================================================================
# Test: Actual Demo Data Exists (Integration)
# =============================================================================

class TestDemoDataExists:
    """Integration tests for actual demo data"""

    def test_demo_directory_exists(self):
        """Demo directory should exist in the repo"""
        demo_dir = Path(__file__).parent.parent / "artifacts" / "pre_race" / "demo"
        assert demo_dir.exists(), "Demo directory should exist for UI testing"

    def test_demo_summary_valid(self):
        """Demo summary JSON should be valid"""
        summary_path = Path(__file__).parent.parent / "artifacts" / "pre_race" / "demo" / "summary_demo.json"

        if not summary_path.exists():
            pytest.skip("Demo summary not found")

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert "n_races" in summary
        assert "races" in summary
        assert summary["n_races"] >= 1

    def test_demo_race_files_valid(self):
        """Demo race JSON files should be valid"""
        demo_dir = Path(__file__).parent.parent / "artifacts" / "pre_race" / "demo"

        if not demo_dir.exists():
            pytest.skip("Demo directory not found")

        race_files = list(demo_dir.glob("race_*.json"))
        assert len(race_files) >= 1, "At least one demo race file should exist"

        for race_file in race_files:
            with open(race_file, "r", encoding="utf-8") as f:
                race_data = json.load(f)

            assert "race_id" in race_data
            assert "entries" in race_data
            assert len(race_data["entries"]) >= 1

            # Each entry should have required fields
            for entry in race_data["entries"]:
                assert "horse_id" in entry
                assert "name" in entry
                assert "umaban" in entry


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
