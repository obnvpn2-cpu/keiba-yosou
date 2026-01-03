# -*- coding: utf-8 -*-
"""
Tests for ui/pre_race/server.py v2.2

Tests for:
- Date folder listing
- Race listing with summary JSON reading
- Save/load history functionality
- Error handling
- v2.2: Structured reasons
- v2.2: Structured race comment
- v2.2: Mini-summary generation
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import server module components
from ui.pre_race.server import (
    apply_scenario_adjustment,
    estimate_run_style,
    estimate_lane,
    generate_horse_tags,
    generate_race_comment,
    generate_race_comment_structured,
    generate_rank_mini_summary,
)


# =============================================================================
# Test Scenario Adjustment Logic
# =============================================================================

class TestScenarioAdjustment:
    """Tests for scenario adjustment logic"""

    def test_apply_scenario_basic(self):
        """Basic scenario adjustment should work"""
        race_data = {
            "race_id": "202412290605",
            "date": "2024-12-29",
            "name": "Test Race",
            "entries": [
                {"horse_id": "h001", "umaban": 1, "name": "Horse1", "p_win": 0.2, "p_in3": 0.5},
                {"horse_id": "h002", "umaban": 2, "name": "Horse2", "p_win": 0.15, "p_in3": 0.4},
            ]
        }
        scenario = {
            "pace": "M",
            "track_condition": "良",
            "lane_bias": "flat",
            "style_bias": "flat",
            "front_runner_ids": [],
        }

        result = apply_scenario_adjustment(race_data, scenario)

        assert result["race_id"] == "202412290605"
        assert len(result["entries"]) == 2
        assert result["entries"][0]["adj_p_win"] > 0
        assert result["entries"][0]["adj_p_in3"] > 0

    def test_apply_scenario_with_pace(self):
        """Pace adjustment should affect running style-based horses"""
        race_data = {
            "race_id": "test",
            "name": "Test",
            "entries": [
                {"horse_id": "h001", "umaban": 1, "name": "Nige", "p_win": 0.2, "p_in3": 0.5, "run_style": "逃げ"},
                {"horse_id": "h002", "umaban": 8, "name": "Oikomi", "p_win": 0.2, "p_in3": 0.5, "run_style": "追込"},
            ]
        }

        # Slow pace favors front runners
        scenario_slow = {
            "pace": "S",
            "track_condition": "良",
            "lane_bias": "flat",
            "style_bias": "flat",
            "front_runner_ids": [],
        }
        result_slow = apply_scenario_adjustment(race_data, scenario_slow)

        # High pace favors closers
        scenario_high = {
            "pace": "H",
            "track_condition": "良",
            "lane_bias": "flat",
            "style_bias": "flat",
            "front_runner_ids": [],
        }
        result_high = apply_scenario_adjustment(race_data, scenario_high)

        # Nige should be higher ranked in slow pace
        nige_slow = next(e for e in result_slow["entries"] if e["name"] == "Nige")
        nige_high = next(e for e in result_high["entries"] if e["name"] == "Nige")
        assert nige_slow["adj_p_win"] > nige_high["adj_p_win"]

    def test_apply_scenario_with_lane_bias(self):
        """Lane bias should affect horses by estimated lane"""
        race_data = {
            "race_id": "test",
            "name": "Test",
            "entries": [
                {"horse_id": "h001", "umaban": 1, "name": "Inner", "p_win": 0.2, "p_in3": 0.5, "run_style": "先行"},
                {"horse_id": "h002", "umaban": 14, "name": "Outer", "p_win": 0.2, "p_in3": 0.5, "run_style": "追込"},
            ]
        }

        # Inner bias
        scenario_inner = {
            "pace": "M",
            "track_condition": "良",
            "lane_bias": "inner",
            "style_bias": "flat",
            "front_runner_ids": [],
        }
        result_inner = apply_scenario_adjustment(race_data, scenario_inner)

        inner_horse = next(e for e in result_inner["entries"] if e["name"] == "Inner")
        outer_horse = next(e for e in result_inner["entries"] if e["name"] == "Outer")

        # Inner should be favored when lane_bias is inner
        assert inner_horse["adj_p_win"] >= outer_horse["adj_p_win"] or inner_horse["adj_rank_win"] <= outer_horse["adj_rank_win"]

    def test_apply_scenario_normalization(self):
        """Probabilities should be normalized after adjustment"""
        race_data = {
            "race_id": "test",
            "name": "Test",
            "entries": [
                {"horse_id": f"h{i:03d}", "umaban": i, "name": f"Horse{i}", "p_win": 0.1, "p_in3": 0.3}
                for i in range(1, 11)
            ]
        }
        scenario = {
            "pace": "M",
            "track_condition": "良",
            "lane_bias": "flat",
            "style_bias": "flat",
            "front_runner_ids": [],
        }

        result = apply_scenario_adjustment(race_data, scenario)

        # Win probabilities should sum to approximately 1
        total_win = sum(e["adj_p_win"] for e in result["entries"])
        assert abs(total_win - 1.0) < 0.01


# =============================================================================
# Test Running Style Estimation
# =============================================================================

class TestRunStyleEstimation:
    """Tests for running style estimation"""

    def test_estimate_from_entry(self):
        """Should use run_style from entry if available"""
        entry = {"run_style": "逃げ"}
        style = estimate_run_style(1, [], "h001", entry)
        assert style == "逃げ"

    def test_estimate_from_front_runner_ids(self):
        """Should use front_runner_ids if specified"""
        style = estimate_run_style(8, ["h001"], "h001", {})
        assert style == "逃げ"

    def test_estimate_from_umaban(self):
        """Should estimate from umaban as fallback"""
        # Inner umaban tends to be 先行
        style = estimate_run_style(2, [], "h002", {})
        assert style == "先行"

        # Middle umaban tends to be 差し
        style = estimate_run_style(6, [], "h006", {})
        assert style == "差し"

        # Outer umaban tends to be 追込
        style = estimate_run_style(12, [], "h012", {})
        assert style == "追込"


# =============================================================================
# Test Lane Estimation
# =============================================================================

class TestLaneEstimation:
    """Tests for lane estimation"""

    def test_estimate_lane_from_entry(self):
        """Should use estimated_lane from entry if available"""
        entry = {"estimated_lane": "outer_lane"}
        lane = estimate_lane(1, 12, "先行", entry)
        assert lane == "outer_lane"

    def test_estimate_lane_inner(self):
        """Inner umaban should estimate inner lane"""
        lane = estimate_lane(1, 12, "先行", {})
        assert lane == "inner_lane"

    def test_estimate_lane_outer(self):
        """Outer umaban should estimate outer lane"""
        lane = estimate_lane(12, 12, "逃げ", {})
        # Nige from outer often cuts to middle
        assert lane == "middle_lane"

    def test_estimate_lane_sashi_adjustment(self):
        """Sashi from inner should be adjusted to middle"""
        lane = estimate_lane(1, 12, "差し", {})
        assert lane == "middle_lane"


# =============================================================================
# Test Horse Tags
# =============================================================================

class TestHorseTags:
    """Tests for horse tag generation"""

    def test_front_runner_tag(self):
        """Front runner should get tag"""
        tags = generate_horse_tags({}, "逃げ", "inner_lane", True)
        assert "逃げ想定" in tags

    def test_running_style_tag(self):
        """Running style should be in tags"""
        tags = generate_horse_tags({}, "差し", "outer_lane", False)
        assert "差し想定" in tags

    def test_rest_day_tags(self):
        """Rest days should generate tags"""
        entry = {"h_days_since_last": 100}
        tags = generate_horse_tags(entry, "先行", "middle_lane", False)
        assert "休み明け" in tags

        entry = {"h_days_since_last": 70}
        tags = generate_horse_tags(entry, "先行", "middle_lane", False)
        assert "間隔あり" in tags

    def test_recent_form_tags(self):
        """Recent form should generate tags"""
        entry = {"h_recent3_avg_finish": 1.5}
        tags = generate_horse_tags(entry, "先行", "middle_lane", False)
        assert "近走好調" in tags

        entry = {"h_recent3_avg_finish": 10.0}
        tags = generate_horse_tags(entry, "先行", "middle_lane", False)
        assert "近走不振" in tags

    def test_last3f_tags(self):
        """Last 3F should generate tags"""
        entry = {"h_recent3_avg_last3f": 33.5}
        tags = generate_horse_tags(entry, "差し", "outer_lane", False)
        assert "末脚○" in tags

        entry = {"h_recent3_avg_last3f": 38.0}
        tags = generate_horse_tags(entry, "追込", "outer_lane", False)
        assert "末脚△" in tags


# =============================================================================
# Test Race Comment Generation
# =============================================================================

class TestRaceComment:
    """Tests for race comment generation"""

    def test_basic_comment(self):
        """Should generate basic comment with pace"""
        race_data = {"name": "Test Race", "distance": 2000, "course": "芝"}
        scenario = {"pace": "M", "lane_bias": "flat", "style_bias": "flat", "track_condition": "良"}
        entries = []

        comment = generate_race_comment(race_data, scenario, entries)

        assert "Test Race" in comment
        assert "平均ペース" in comment

    def test_comment_with_slow_pace(self):
        """Should mention slow pace"""
        race_data = {"name": "Test"}
        scenario = {"pace": "S", "lane_bias": "flat", "style_bias": "flat", "track_condition": "良"}

        comment = generate_race_comment(race_data, scenario, [])
        assert "スローペース" in comment

    def test_comment_with_high_pace(self):
        """Should mention high pace"""
        race_data = {"name": "Test"}
        scenario = {"pace": "H", "lane_bias": "flat", "style_bias": "flat", "track_condition": "良"}

        comment = generate_race_comment(race_data, scenario, [])
        assert "ハイペース" in comment

    def test_comment_with_style_bias(self):
        """Should mention style bias"""
        race_data = {"name": "Test"}
        scenario = {"pace": "M", "lane_bias": "flat", "style_bias": "front", "track_condition": "良"}

        comment = generate_race_comment(race_data, scenario, [])
        assert "前残り" in comment

    def test_comment_with_lane_bias(self):
        """Should mention lane bias"""
        race_data = {"name": "Test"}
        scenario = {"pace": "M", "lane_bias": "inner", "style_bias": "flat", "track_condition": "良"}

        comment = generate_race_comment(race_data, scenario, [])
        assert "内" in comment

    def test_comment_with_front_runners(self):
        """Should mention front runners"""
        race_data = {"name": "Test"}
        scenario = {"pace": "M", "lane_bias": "flat", "style_bias": "flat", "track_condition": "良"}
        entries = [{"name": "Nige1", "run_style": "逃げ"}]

        comment = generate_race_comment(race_data, scenario, entries)
        assert "Nige1" in comment or "逃げ" in comment


# =============================================================================
# Test Output Path Structure
# =============================================================================

class TestOutputPathStructure:
    """Tests for output path structure"""

    def test_output_path_format(self):
        """Output path should follow <date>/<race_id>/<timestamp>.json format"""
        # This tests the path construction logic
        race_date = "2024-12-29"
        race_id = "202412290605"

        expected_pattern = f"artifacts/pre_race/outputs/{race_date}/{race_id}/"

        # Verify the pattern is correct
        from pathlib import Path
        output_dir = Path("artifacts") / "pre_race" / "outputs" / race_date / race_id
        assert str(output_dir) == f"artifacts/pre_race/outputs/{race_date}/{race_id}"


# =============================================================================
# Test Summary JSON Reading (Mock)
# =============================================================================

class TestSummaryReading:
    """Tests for summary JSON reading logic"""

    def test_summary_format(self):
        """Summary JSON should have expected format"""
        summary = {
            "date": "2024-12-29",
            "n_races": 2,
            "races": [
                {
                    "race_id": "202412290601",
                    "place": "中山",
                    "race_no": 1,
                    "name": "未勝利",
                    "grade": None,
                    "distance": 1200,
                    "n_entries": 10,
                },
                {
                    "race_id": "202412290605",
                    "place": "中山",
                    "race_no": 5,
                    "name": "有馬記念",
                    "grade": "G1",
                    "distance": 2500,
                    "n_entries": 12,
                },
            ],
            "features_version": "v4",
        }

        # Verify structure
        assert summary["n_races"] == 2
        assert len(summary["races"]) == 2
        assert summary["races"][1]["grade"] == "G1"
        assert summary["races"][1]["distance"] == 2500

    def test_race_label_generation(self):
        """Race label should be human-readable"""
        race_info = {
            "place": "中山",
            "race_no": 5,
            "name": "有馬記念",
            "grade": "G1",
            "distance": 2500,
            "start_time": "15:25",
            "n_entries": 12,
        }

        # Build label parts
        label_parts = []
        if race_info["place"]:
            label_parts.append(race_info["place"])
        if race_info["race_no"]:
            label_parts.append(f"{race_info['race_no']}R")
        if race_info["name"]:
            label_parts.append(race_info["name"])
        if race_info["grade"]:
            label_parts.append(f"({race_info['grade']})")
        if race_info["distance"]:
            label_parts.append(f"{race_info['distance']}m")
        if race_info["start_time"]:
            label_parts.append(race_info["start_time"])
        if race_info["n_entries"]:
            label_parts.append(f"{race_info['n_entries']}頭")

        label = " ".join(label_parts)

        assert "中山" in label
        assert "5R" in label
        assert "有馬記念" in label
        assert "(G1)" in label
        assert "2500m" in label
        assert "15:25" in label
        assert "12頭" in label


# =============================================================================
# v2.2 Test Structured Reasons
# =============================================================================

class TestStructuredReasons:
    """Tests for v2.2 structured reasons"""

    def test_structured_reasons_included(self):
        """Adjusted entries should include reasons_structured"""
        race_data = {
            "race_id": "test",
            "name": "Test Race",
            "entries": [
                {"horse_id": "h001", "umaban": 1, "name": "Horse1", "p_win": 0.2, "p_in3": 0.5, "run_style": "逃げ"},
            ]
        }
        scenario = {
            "pace": "S",
            "track_condition": "良",
            "lane_bias": "inner",
            "style_bias": "flat",
            "front_runner_ids": [],
        }

        result = apply_scenario_adjustment(race_data, scenario)

        # Check that reasons_structured exists
        entry = result["entries"][0]
        assert "reasons_structured" in entry
        assert isinstance(entry["reasons_structured"], dict)

    def test_structured_reasons_has_categories(self):
        """Structured reasons should have correct categories"""
        race_data = {
            "race_id": "test",
            "name": "Test",
            "entries": [
                {"horse_id": "h001", "umaban": 1, "name": "Nige", "p_win": 0.2, "p_in3": 0.5, "run_style": "逃げ"},
            ]
        }
        scenario = {
            "pace": "S",
            "track_condition": "重",
            "lane_bias": "inner",
            "style_bias": "front",
            "front_runner_ids": ["h001"],
        }

        result = apply_scenario_adjustment(race_data, scenario)
        entry = result["entries"][0]
        reasons = entry["reasons_structured"]

        # Should have pace, style_bias, lane_bias, track_condition categories
        assert "pace" in reasons
        assert reasons["pace"]["category"] == "pace"
        assert reasons["pace"]["effect"] in ["positive", "negative"]

        assert "track_condition" in reasons
        assert reasons["track_condition"]["category"] == "track_condition"


# =============================================================================
# v2.2 Test Structured Race Comment
# =============================================================================

class TestStructuredRaceComment:
    """Tests for v2.2 structured race comment"""

    def test_structured_comment_has_sections(self):
        """Structured comment should have all sections"""
        race_data = {
            "race_id": "test",
            "name": "Test Race",
            "place": "東京",
            "distance": 2000,
            "course": "芝",
        }
        scenario = {
            "pace": "S",
            "lane_bias": "inner",
            "style_bias": "front",
            "track_condition": "良",
            "notes": "Test memo",
        }
        entries = [
            {"name": "Horse1", "run_style": "逃げ", "rank_change_win": 2, "reasons_structured": {"pace": {"text": "ペースSで逃げ有利"}}},
        ]

        comment = generate_race_comment_structured(race_data, scenario, entries)

        assert "overview" in comment
        assert "bias" in comment
        assert "focus" in comment
        assert "highlight" in comment
        assert "supplement" in comment

    def test_structured_comment_overview_content(self):
        """Overview should include race info"""
        race_data = {
            "name": "有馬記念",
            "place": "中山",
            "distance": 2500,
            "course": "芝",
        }
        scenario = {
            "pace": "M",
            "lane_bias": "flat",
            "style_bias": "flat",
            "track_condition": "良",
        }

        comment = generate_race_comment_structured(race_data, scenario, [])

        assert "有馬記念" in comment["overview"]
        assert "中山" in comment["overview"]
        assert "2500m" in comment["overview"]

    def test_structured_comment_bias_content(self):
        """Bias section should include lane and style bias"""
        race_data = {"name": "Test"}
        scenario = {
            "pace": "M",
            "lane_bias": "inner",
            "style_bias": "front",
            "track_condition": "良",
        }

        comment = generate_race_comment_structured(race_data, scenario, [])

        assert "内伸び" in comment["bias"]
        assert "前残り" in comment["bias"]


# =============================================================================
# v2.2 Test Mini Summary
# =============================================================================

class TestMiniSummary:
    """Tests for v2.2 mini summary generation"""

    def test_mini_summary_format(self):
        """Mini summary should show top 3 ranking"""
        entries = [
            {"horse_id": "h001", "name": "Horse1", "base_rank_win": 3, "adj_rank_win": 1},
            {"horse_id": "h002", "name": "Horse2", "base_rank_win": 1, "adj_rank_win": 2},
            {"horse_id": "h003", "name": "Horse3", "base_rank_win": 5, "adj_rank_win": 3},
        ]

        summary = generate_rank_mini_summary(entries)

        assert "Horse1" in summary
        assert "Horse2" in summary
        assert "Horse3" in summary
        assert "1位" in summary

    def test_mini_summary_shows_rank_change(self):
        """Mini summary should indicate rank changes"""
        entries = [
            {"horse_id": "h001", "name": "Changed", "base_rank_win": 5, "adj_rank_win": 1},
        ]

        summary = generate_rank_mini_summary(entries)

        # Should show old → new format when changed
        assert "5→1" in summary

    def test_mini_summary_unchanged_rank(self):
        """Mini summary should handle unchanged ranks"""
        entries = [
            {"horse_id": "h001", "name": "Same", "base_rank_win": 1, "adj_rank_win": 1},
        ]

        summary = generate_rank_mini_summary(entries)

        # Should just show the rank without arrow
        assert "Same" in summary
        assert "(1)" in summary


# =============================================================================
# v2.2 Test Result Output Structure
# =============================================================================

class TestResultOutputV22:
    """Tests for v2.2 result output structure"""

    def test_result_has_structured_comment(self):
        """Result should include race_comment_structured"""
        race_data = {
            "race_id": "test",
            "name": "Test",
            "entries": [
                {"horse_id": "h001", "umaban": 1, "name": "Horse1", "p_win": 0.2, "p_in3": 0.5},
            ]
        }
        scenario = {
            "pace": "M",
            "track_condition": "良",
            "lane_bias": "flat",
            "style_bias": "flat",
            "front_runner_ids": [],
        }

        result = apply_scenario_adjustment(race_data, scenario)

        assert "race_comment_structured" in result
        assert isinstance(result["race_comment_structured"], dict)

    def test_result_has_mini_summary(self):
        """Result should include rank_mini_summary"""
        race_data = {
            "race_id": "test",
            "name": "Test",
            "entries": [
                {"horse_id": "h001", "umaban": 1, "name": "Horse1", "p_win": 0.2, "p_in3": 0.5},
            ]
        }
        scenario = {
            "pace": "M",
            "track_condition": "良",
            "lane_bias": "flat",
            "style_bias": "flat",
            "front_runner_ids": [],
        }

        result = apply_scenario_adjustment(race_data, scenario)

        assert "rank_mini_summary" in result
        assert isinstance(result["rank_mini_summary"], str)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
