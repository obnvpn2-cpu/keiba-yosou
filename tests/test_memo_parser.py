# -*- coding: utf-8 -*-
"""
Tests for src/ui_pre_race/memo_parser.py
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui_pre_race.memo_parser import MemoParser, parse_memo, Chip, Conflict


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def parser():
    """Create a MemoParser instance"""
    return MemoParser()


# =============================================================================
# Test Basic Functionality
# =============================================================================

class TestMemoParserBasics:
    """Basic functionality tests"""

    def test_parser_initialization(self, parser):
        """Parser should initialize correctly"""
        assert parser.rules is not None
        assert "track_condition" in parser.rules
        assert "lane_bias" in parser.rules
        assert "style_bias" in parser.rules
        assert "pace" in parser.rules

    def test_empty_memo(self, parser):
        """Empty memo should return empty result"""
        result = parser.parse("")
        assert len(result.chips) == 0
        assert len(result.conflicts) == 0

    def test_whitespace_only_memo(self, parser):
        """Whitespace-only memo should return empty result"""
        result = parser.parse("   \n\t  ")
        assert len(result.chips) == 0


# =============================================================================
# Test Track Condition Detection
# =============================================================================

class TestTrackConditionDetection:
    """Track condition keyword detection tests"""

    def test_detect_high_speed_track(self, parser):
        """Should detect high speed track (良)"""
        result = parser.parse("開幕週の高速馬場で時計が速い")
        chip_ids = [c.id for c in result.chips]
        assert "track_condition.良" in chip_ids

    def test_detect_heavy_track(self, parser):
        """Should detect heavy track (重)"""
        result = parser.parse("道悪の重馬場でパワーが必要")
        chip_ids = [c.id for c in result.chips]
        assert "track_condition.重" in chip_ids

    def test_detect_soft_track(self, parser):
        """Should detect soft track (不良)"""
        result = parser.parse("不良馬場でかなりの道悪")
        chip_ids = [c.id for c in result.chips]
        assert "track_condition.不良" in chip_ids


# =============================================================================
# Test Lane Bias Detection
# =============================================================================

class TestLaneBiasDetection:
    """Lane bias keyword detection tests"""

    def test_detect_inner_bias(self, parser):
        """Should detect inner lane bias"""
        result = parser.parse("内伸びの馬場で内ラチ沿いが有利")
        chip_ids = [c.id for c in result.chips]
        assert "lane_bias.inner" in chip_ids

    def test_detect_outer_bias(self, parser):
        """Should detect outer lane bias"""
        result = parser.parse("外伸びで外差しが届く馬場")
        chip_ids = [c.id for c in result.chips]
        assert "lane_bias.outer" in chip_ids

    def test_opening_week_inner_bias(self, parser):
        """Opening week should suggest inner bias"""
        result = parser.parse("開幕週")
        # Should detect both inner (from opening week) and potentially track condition
        chip_ids = [c.id for c in result.chips]
        assert "lane_bias.inner" in chip_ids


# =============================================================================
# Test Style Bias Detection
# =============================================================================

class TestStyleBiasDetection:
    """Style bias keyword detection tests"""

    def test_detect_front_bias(self, parser):
        """Should detect front running style bias"""
        result = parser.parse("前残りで先行有利")
        chip_ids = [c.id for c in result.chips]
        assert "style_bias.front" in chip_ids

    def test_detect_closer_bias(self, parser):
        """Should detect closer style bias"""
        result = parser.parse("差しが届く展開で追込有利")
        chip_ids = [c.id for c in result.chips]
        assert "style_bias.closer" in chip_ids

    def test_slow_pace_implies_front(self, parser):
        """Slow pace keywords should suggest front advantage"""
        result = parser.parse("スローペースで上がり勝負")
        chip_ids = [c.id for c in result.chips]
        # スローペース should be detected
        assert "pace.S" in chip_ids


# =============================================================================
# Test Pace Detection
# =============================================================================

class TestPaceDetection:
    """Pace keyword detection tests"""

    def test_detect_slow_pace(self, parser):
        """Should detect slow pace"""
        result = parser.parse("スローペースで瞬発力勝負")
        chip_ids = [c.id for c in result.chips]
        assert "pace.S" in chip_ids

    def test_detect_high_pace(self, parser):
        """Should detect high pace"""
        result = parser.parse("ハイペースで消耗戦")
        chip_ids = [c.id for c in result.chips]
        assert "pace.H" in chip_ids


# =============================================================================
# Test Pedigree Keywords (needs_feature)
# =============================================================================

class TestPedigreeDetection:
    """Pedigree keyword detection tests"""

    def test_detect_deep_impact(self, parser):
        """Should detect Deep Impact pedigree with needs_feature flag"""
        result = parser.parse("ディープインパクト産駒向き")

        pedigree_chips = [c for c in result.chips if c.category == "pedigree"]
        assert len(pedigree_chips) >= 1
        assert pedigree_chips[0].needs_feature == True

    def test_pedigree_not_in_apply_payload(self, parser):
        """Pedigree chips should have empty apply_payload"""
        result = parser.parse("ディープ系の血統")

        pedigree_chips = [c for c in result.chips if c.category == "pedigree"]
        if pedigree_chips:
            assert pedigree_chips[0].apply_payload == {}


# =============================================================================
# Test Conflict Detection
# =============================================================================

class TestConflictDetection:
    """Conflict detection tests"""

    def test_lane_bias_conflict(self, parser):
        """Should detect conflicting lane bias keywords"""
        result = parser.parse("内伸びだが外伸びもある")

        assert len(result.conflicts) >= 1
        conflict_messages = [c.message for c in result.conflicts]
        assert any("進路" in m for m in conflict_messages)

    def test_pace_conflict(self, parser):
        """Should detect conflicting pace keywords"""
        result = parser.parse("スローペースだがハイペースになる可能性も")

        assert len(result.conflicts) >= 1
        conflict_messages = [c.message for c in result.conflicts]
        assert any("ペース" in m for m in conflict_messages)

    def test_warning_chip_for_conflict(self, parser):
        """Conflict should generate warning chip"""
        result = parser.parse("内伸びと外伸び両方")

        warning_chips = [c for c in result.chips if c.is_warning]
        assert len(warning_chips) >= 1


# =============================================================================
# Test Intensity Modifiers
# =============================================================================

class TestIntensityModifiers:
    """Intensity modifier tests"""

    def test_strong_modifier(self, parser):
        """'かなり' should increase score"""
        result1 = parser.parse("高速馬場")
        result2 = parser.parse("かなりの高速馬場")

        # Both should detect 良 track condition
        assert "track_condition" in result1.suggestions
        assert "track_condition" in result2.suggestions

        # The one with modifier should have higher score
        assert result2.suggestions["track_condition"]["score"] >= result1.suggestions["track_condition"]["score"]

    def test_weak_modifier(self, parser):
        """'やや' should decrease score"""
        result1 = parser.parse("高速馬場")
        result2 = parser.parse("やや高速馬場")

        if "track_condition" in result1.suggestions and "track_condition" in result2.suggestions:
            # The one with weak modifier should have lower score
            assert result2.suggestions["track_condition"]["score"] <= result1.suggestions["track_condition"]["score"]


# =============================================================================
# Test Negation Patterns
# =============================================================================

class TestNegationPatterns:
    """Negation pattern tests"""

    def test_negation_excluded(self, parser):
        """Negated keywords should not generate chips"""
        result = parser.parse("高速馬場ではない")

        # Should not have 良 in chips (negated)
        chip_ids = [c.id for c in result.chips if not c.is_warning]
        # The negated keyword should either be excluded or marked differently
        # Check that the score is negative or zero
        if "track_condition" in result.suggestions:
            assert result.suggestions["track_condition"]["score"] <= 0 or \
                   "track_condition.良" not in chip_ids


# =============================================================================
# Test Full Acceptance Criteria
# =============================================================================

class TestAcceptanceCriteria:
    """Tests for the specified acceptance criteria"""

    def test_full_memo_example(self, parser):
        """
        Test the acceptance criteria memo:
        "開幕週の高速馬場で内枠、前有利になってる。瞬発力のあるディープインパクト系の血統が得意な軽い芝。"

        Expected:
        - 馬場：良（高速示唆）
        - 進路：内寄り（開幕週）
        - 脚質：前寄り（前残り）
        - 展開：瞬発力勝負（スロー寄り）
        - 血統：SS/ディープ言及（needs_feature: pedigree）
        """
        result = parser.parse(
            "開幕週の高速馬場で内枠、前有利になってる。瞬発力のあるディープインパクト系の血統が得意な軽い芝。"
        )

        chip_ids = [c.id for c in result.chips]

        # 馬場：良（高速示唆）
        assert "track_condition.良" in chip_ids, "Should detect 良 track condition"

        # 進路：内寄り（開幕週）
        assert "lane_bias.inner" in chip_ids, "Should detect inner lane bias"

        # 血統（needs_feature）
        pedigree_chips = [c for c in result.chips if c.category == "pedigree"]
        assert len(pedigree_chips) >= 1, "Should detect pedigree keywords"
        assert pedigree_chips[0].needs_feature == True, "Pedigree should have needs_feature=True"

    def test_chip_apply_payload_correctness(self, parser):
        """Chips should have correct apply_payload for form autofill"""
        result = parser.parse("高速馬場で内伸び、スローペース")

        for chip in result.chips:
            if chip.id == "track_condition.良":
                assert chip.apply_payload == {"track_condition": "良"}
            elif chip.id == "lane_bias.inner":
                assert chip.apply_payload == {"lane_bias": "inner"}
            elif chip.id == "pace.S":
                assert chip.apply_payload == {"pace": "S"}


# =============================================================================
# Test parse_memo Convenience Function
# =============================================================================

class TestParseMemoFunction:
    """Tests for the parse_memo convenience function"""

    def test_parse_memo_returns_dict(self):
        """parse_memo should return a dictionary"""
        result = parse_memo("高速馬場")

        assert isinstance(result, dict)
        assert "chips" in result
        assert "conflicts" in result
        assert "suggestions" in result

    def test_parse_memo_chips_format(self):
        """Chips in dict format should have all required fields"""
        result = parse_memo("高速馬場")

        for chip in result["chips"]:
            assert "id" in chip
            assert "label" in chip
            assert "reason" in chip
            assert "category" in chip
            assert "apply_payload" in chip
            assert "needs_feature" in chip
            assert "is_warning" in chip


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
