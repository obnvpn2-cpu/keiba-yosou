# -*- coding: utf-8 -*-
"""
Tests for src/ui_pre_race/horse_profile.py
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui_pre_race.horse_profile import (
    generate_horse_profile,
    generate_profiles_for_race,
    PRE_RACE_UNSAFE_FEATURES,
)


# =============================================================================
# Test Basic Functionality
# =============================================================================

class TestHorseProfileBasics:
    """Basic functionality tests"""

    def test_empty_entry(self):
        """Empty entry should return minimal profile"""
        profile = generate_horse_profile({})
        assert "overview" in profile
        assert "supplement" in profile
        assert "comments" in profile
        assert "evidence" in profile
        assert isinstance(profile["comments"], list)
        assert isinstance(profile["evidence"], list)

    def test_profile_structure(self):
        """Profile should have correct structure"""
        entry = {
            "horse_id": "test001",
            "name": "TestHorse",
            "h_recent3_avg_finish": 2.0,
        }
        profile = generate_horse_profile(entry)

        assert isinstance(profile["overview"], str)
        assert isinstance(profile["supplement"], str)
        assert isinstance(profile["comments"], list)
        assert isinstance(profile["evidence"], list)

    def test_evidence_structure(self):
        """Evidence items should have correct structure"""
        entry = {
            "horse_id": "test001",
            "h_recent3_avg_finish": 1.5,
        }
        profile = generate_horse_profile(entry)

        assert len(profile["evidence"]) > 0
        for e in profile["evidence"]:
            assert "key" in e
            assert "value" in e
            assert "reason" in e


# =============================================================================
# Test Recent Form Detection
# =============================================================================

class TestRecentFormDetection:
    """Recent form feature detection tests"""

    def test_recent_good_form(self):
        """Should detect good recent form"""
        entry = {"h_recent3_avg_finish": 1.5}
        profile = generate_horse_profile(entry)

        assert "近走好調" in profile["comments"]
        evidence_keys = [e["key"] for e in profile["evidence"]]
        assert "h_recent3_avg_finish" in evidence_keys

    def test_recent_stable_form(self):
        """Should detect stable recent form"""
        entry = {"h_recent3_avg_finish": 3.5}
        profile = generate_horse_profile(entry)

        assert "近走安定" in profile["comments"]

    def test_recent_poor_form(self):
        """Should detect poor recent form"""
        entry = {"h_recent3_avg_finish": 12.0}
        profile = generate_horse_profile(entry)

        assert "近走不振" in profile["comments"]

    def test_recent_winner(self):
        """Should detect recent winner"""
        entry = {"h_recent3_best_finish": 1}
        profile = generate_horse_profile(entry)

        assert "近走勝ち馬" in profile["comments"]


# =============================================================================
# Test Last 3F (Closing Speed) Detection
# =============================================================================

class TestLast3FDetection:
    """Last 3F feature detection tests"""

    def test_sharp_closing_speed(self):
        """Should detect sharp closing speed"""
        entry = {"h_recent3_avg_last3f": 33.5}
        profile = generate_horse_profile(entry)

        assert "末脚鋭い" in profile["comments"]
        assert "末脚型" in profile["overview"]

    def test_slow_closing_speed(self):
        """Should detect slow closing speed"""
        entry = {"h_recent3_avg_last3f": 38.0}
        profile = generate_horse_profile(entry)

        assert "上がり平凡" in profile["comments"]

    def test_explosive_speed(self):
        """Should detect explosive best last 3F"""
        entry = {"h_best_last3f": 32.5}
        profile = generate_horse_profile(entry)

        assert "瞬発力あり" in profile["comments"]


# =============================================================================
# Test Rest Days Detection
# =============================================================================

class TestRestDaysDetection:
    """Rest days feature detection tests"""

    def test_long_layoff(self):
        """Should detect long layoff (180+ days)"""
        entry = {"h_days_since_last": 200}
        profile = generate_horse_profile(entry)

        assert "長期休み明け" in profile["comments"]

    def test_medium_layoff(self):
        """Should detect medium layoff (90+ days)"""
        entry = {"h_days_since_last": 100}
        profile = generate_horse_profile(entry)

        assert "休み明け" in profile["comments"]

    def test_quick_turnaround(self):
        """Should detect quick turnaround (14 days or less)"""
        entry = {"h_days_since_last": 7}
        profile = generate_horse_profile(entry)

        assert "連闘" in profile["comments"]


# =============================================================================
# Test Distance/Course Aptitude Detection
# =============================================================================

class TestAptitudeDetection:
    """Distance and course aptitude detection tests"""

    def test_strong_distance_aptitude(self):
        """Should detect strong distance aptitude"""
        entry = {
            "h_win_rate_dist": 0.30,
            "h_n_starts_dist": 5,
        }
        profile = generate_horse_profile(entry)

        assert "距離◎" in profile["comments"]

    def test_good_distance_aptitude(self):
        """Should detect good distance aptitude"""
        entry = {
            "h_in3_rate_dist": 0.55,
            "h_n_starts_dist": 5,
        }
        profile = generate_horse_profile(entry)

        assert "距離○" in profile["comments"]

    def test_poor_distance_aptitude(self):
        """Should detect poor distance aptitude"""
        entry = {
            "h_in3_rate_dist": 0.10,
            "h_n_starts_dist": 6,
        }
        profile = generate_horse_profile(entry)

        assert "距離△" in profile["comments"]

    def test_strong_course_aptitude(self):
        """Should detect strong course aptitude"""
        entry = {
            "h_win_rate_course": 0.35,
            "h_n_starts_course": 3,
        }
        profile = generate_horse_profile(entry)

        assert "コース◎" in profile["comments"]


# =============================================================================
# Test Running Style Detection
# =============================================================================

class TestRunningStyleDetection:
    """Running style detection tests"""

    def test_running_style_nige(self):
        """Should detect 逃げ running style"""
        entry = {"h_running_style_id": 0}
        profile = generate_horse_profile(entry)

        assert "逃げ脚質" in profile["comments"]
        assert "逃げ脚質" in profile["overview"]

    def test_running_style_senkou(self):
        """Should detect 先行 running style"""
        entry = {"h_running_style_id": 1}
        profile = generate_horse_profile(entry)

        assert "先行脚質" in profile["comments"]

    def test_running_style_sashi(self):
        """Should detect 差し running style"""
        entry = {"h_running_style_id": 2}
        profile = generate_horse_profile(entry)

        assert "差し脚質" in profile["comments"]

    def test_running_style_oikomi(self):
        """Should detect 追込 running style"""
        entry = {"h_running_style_id": 3}
        profile = generate_horse_profile(entry)

        assert "追込脚質" in profile["comments"]

    def test_makuri_tendency(self):
        """Should detect 捲り (position-gaining) tendency"""
        entry = {"h_pos_change_tendency": -4.0}
        profile = generate_horse_profile(entry)

        assert "捲り脚" in profile["comments"]


# =============================================================================
# Test Jockey/Trainer Stats Detection
# =============================================================================

class TestJockeyTrainerDetection:
    """Jockey and trainer stats detection tests"""

    def test_high_jockey_win_rate(self):
        """Should detect high jockey win rate"""
        entry = {"j_win_rate": 0.18}
        profile = generate_horse_profile(entry)

        assert "J高勝率" in profile["comments"]

    def test_stable_jockey(self):
        """Should detect stable jockey"""
        entry = {"j_in3_rate": 0.40}
        profile = generate_horse_profile(entry)

        assert "J安定" in profile["comments"]

    def test_good_combo(self):
        """Should detect good jockey-horse combo"""
        entry = {
            "jh_combo_win_rate": 0.30,
            "jh_n_combos": 4,
        }
        profile = generate_horse_profile(entry)

        assert "コンビ◎" in profile["comments"]

    def test_high_trainer_win_rate(self):
        """Should detect high trainer win rate"""
        entry = {"t_win_rate": 0.15}
        profile = generate_horse_profile(entry)

        assert "T高勝率" in profile["comments"]


# =============================================================================
# Test Graded Race Experience Detection
# =============================================================================

class TestGradedRaceDetection:
    """Graded race experience detection tests"""

    def test_g1_experience(self):
        """Should detect G1 experience"""
        entry = {"h_n_g1_runs": 2}
        profile = generate_horse_profile(entry)

        assert "G1経験" in profile["comments"]

    def test_good_graded_performance(self):
        """Should detect good graded race performance"""
        entry = {"h_graded_in3_rate": 0.35}
        profile = generate_horse_profile(entry)

        assert "重賞実績○" in profile["comments"]


# =============================================================================
# Test First Run Flag
# =============================================================================

class TestFirstRunDetection:
    """First run flag detection tests"""

    def test_first_run(self):
        """Should detect first run"""
        entry = {"h_is_first_run": 1}
        profile = generate_horse_profile(entry)

        assert "初出走" in profile["comments"]
        assert "初出走馬" in profile["overview"]


# =============================================================================
# Test Pre-race Unsafe Features Exclusion
# =============================================================================

class TestPreRaceUnsafeFeaturesExclusion:
    """Test that pre-race unsafe features are not used in profile"""

    def test_body_weight_not_used(self):
        """Profile should not reference body weight features"""
        entry = {
            "h_body_weight": 500,
            "h_body_weight_diff": 10,
            "h_body_weight_dev": 5.0,
        }
        profile = generate_horse_profile(entry)

        # Evidence should not contain body weight keys
        evidence_keys = [e["key"] for e in profile["evidence"]]
        for key in PRE_RACE_UNSAFE_FEATURES:
            assert key not in evidence_keys, f"{key} should not be in evidence"

    def test_market_info_not_used(self):
        """Profile should not reference market info features"""
        entry = {
            "market_win_odds": 5.0,
            "market_popularity": 3,
        }
        profile = generate_horse_profile(entry)

        evidence_keys = [e["key"] for e in profile["evidence"]]
        assert "market_win_odds" not in evidence_keys
        assert "market_popularity" not in evidence_keys


# =============================================================================
# Test Batch Processing
# =============================================================================

class TestBatchProcessing:
    """Batch processing tests"""

    def test_generate_profiles_for_race(self):
        """Should generate profiles for all entries"""
        entries = [
            {"horse_id": "h001", "name": "Horse1", "h_recent3_avg_finish": 2.0},
            {"horse_id": "h002", "name": "Horse2", "h_recent3_avg_finish": 5.0},
            {"horse_id": "h003", "name": "Horse3", "h_running_style_id": 0},
        ]

        results = generate_profiles_for_race(entries)

        assert len(results) == 3
        for entry in results:
            assert "profile" in entry
            assert "overview" in entry["profile"]

    def test_generate_profiles_with_feature_data(self):
        """Should merge feature data for profile generation"""
        entries = [
            {"horse_id": "h001", "name": "Horse1"},
        ]
        feature_data_map = {
            "h001": {
                "h_recent3_avg_finish": 1.5,
                "h_running_style_id": 2,
            }
        }

        results = generate_profiles_for_race(entries, feature_data_map)

        assert len(results) == 1
        profile = results[0]["profile"]
        assert "近走好調" in profile["comments"]
        assert "差し脚質" in profile["comments"]


# =============================================================================
# Test Complex Entry
# =============================================================================

class TestComplexEntry:
    """Complex entry with multiple features"""

    def test_complex_entry(self):
        """Should handle entry with many features"""
        entry = {
            "horse_id": "test001",
            "name": "ComplexHorse",
            "h_recent3_avg_finish": 2.0,
            "h_recent3_avg_last3f": 33.8,
            "h_days_since_last": 28,
            "h_win_rate_dist": 0.25,
            "h_n_starts_dist": 4,
            "h_running_style_id": 2,
            "j_win_rate": 0.12,
            "h_n_g1_runs": 1,
        }

        profile = generate_horse_profile(entry)

        # Should have multiple comments and evidence
        assert len(profile["comments"]) >= 3
        assert len(profile["evidence"]) >= 3

        # Overview should not be empty
        assert profile["overview"]

        # Supplement should have data
        assert profile["supplement"]


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
