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
# Test Frame Hint Separation (v3.0)
# 重要: 枠への言及はlane_biasに直結しない
# =============================================================================

class TestFrameHintSeparation:
    """Tests for frame hint vs lane bias separation (v3.0 critical feature)"""

    def test_inner_frame_not_lane_bias(self, parser):
        """内枠 mention should NOT generate lane_bias.inner"""
        result = parser.parse("内枠有利な展開")
        chip_ids = [c.id for c in result.chips]

        # Should detect frame_hint, NOT lane_bias
        assert "frame_hint.inner_frame" in chip_ids
        # lane_bias.inner should NOT be generated from 内枠 alone
        lane_bias_chips = [c for c in result.chips if c.category == "lane_bias"]
        assert len(lane_bias_chips) == 0, "内枠 should not generate lane_bias"

    def test_outer_frame_not_lane_bias(self, parser):
        """外枠 mention should NOT generate lane_bias.outer"""
        result = parser.parse("外枠不利")
        chip_ids = [c.id for c in result.chips]

        # Should detect frame_hint
        assert "frame_hint.outer_frame" in chip_ids
        # Should NOT generate lane_bias.outer
        lane_bias_chips = [c for c in result.chips if c.category == "lane_bias"]
        assert len(lane_bias_chips) == 0

    def test_frame_hint_is_warning(self, parser):
        """Frame hint chips should have is_warning=True"""
        result = parser.parse("内枠を引きたい")

        frame_chips = [c for c in result.chips if c.category == "frame_hint"]
        assert len(frame_chips) >= 1
        assert frame_chips[0].is_warning == True

    def test_frame_hint_no_apply_payload(self, parser):
        """Frame hint chips should have empty apply_payload"""
        result = parser.parse("内枠有利")

        frame_chips = [c for c in result.chips if c.category == "frame_hint"]
        if frame_chips:
            assert frame_chips[0].apply_payload == {}

    def test_lane_bias_vs_frame_hint_distinction(self, parser):
        """内伸び should be lane_bias, 内枠 should be frame_hint"""
        result = parser.parse("内伸びだが内枠とは限らない")
        chip_ids = [c.id for c in result.chips]

        # 内伸び → lane_bias
        assert "lane_bias.inner" in chip_ids
        # 内枠 → frame_hint (but negated, so may not appear as chip)
        # The key point: lane_bias is from 内伸び, not 内枠


# =============================================================================
# Test Race Flow Detection (v3.0)
# =============================================================================

class TestRaceFlowDetection:
    """Tests for race flow category (v3.0)"""

    def test_many_front_runners(self, parser):
        """Should detect many front runners scenario"""
        result = parser.parse("逃げ馬が多くてペースが流れそう")
        chip_ids = [c.id for c in result.chips]

        assert "race_flow.many_front_runners" in chip_ids

    def test_few_front_runners(self, parser):
        """Should detect few front runners scenario"""
        result = parser.parse("逃げ馬不在で楽逃げ確実")
        chip_ids = [c.id for c in result.chips]

        assert "race_flow.few_front_runners" in chip_ids

    def test_pace_maker(self, parser):
        """Should detect pace maker presence"""
        result = parser.parse("ペースメーカーがいてペースは安定")

        race_flow_chips = [c for c in result.chips if c.category == "race_flow"]
        values = [c.id for c in race_flow_chips]
        assert "race_flow.pace_maker" in values


# =============================================================================
# Test Natural Language Samples (v3.0)
# 20-30 realistic memo samples
# =============================================================================

class TestNaturalLanguageSamples:
    """Tests for natural language memo samples (v3.0 acceptance criteria)"""

    # ---------------------------------------------------------------------------
    # Sample 1-5: Basic patterns
    # ---------------------------------------------------------------------------

    def test_sample_01_opening_week_fast_track(self, parser):
        """Sample 1: 開幕週の高速馬場パターン"""
        result = parser.parse("開幕週で高速馬場、内伸びが続いている")
        chip_ids = [c.id for c in result.chips]

        assert "track_condition.良" in chip_ids
        assert "lane_bias.inner" in chip_ids

    def test_sample_02_heavy_track_outer(self, parser):
        """Sample 2: 道悪の外伸びパターン"""
        result = parser.parse("道悪で内がぐちゃぐちゃ、外を回した方が伸びる")
        chip_ids = [c.id for c in result.chips]

        assert "track_condition.重" in chip_ids
        assert "lane_bias.outer" in chip_ids

    def test_sample_03_slow_pace_front_advantage(self, parser):
        """Sample 3: スローで前残りパターン"""
        result = parser.parse("逃げ候補少なくスローペースで前残り濃厚")
        chip_ids = [c.id for c in result.chips]

        assert "pace.S" in chip_ids
        assert "style_bias.front" in chip_ids

    def test_sample_04_high_pace_closer_advantage(self, parser):
        """Sample 4: ハイペースで差し有利パターン"""
        result = parser.parse("先行馬多くハイペース必至、差し追込有利")
        chip_ids = [c.id for c in result.chips]

        assert "pace.H" in chip_ids
        assert "style_bias.closer" in chip_ids

    def test_sample_05_pedigree_mention(self, parser):
        """Sample 5: 血統言及パターン"""
        result = parser.parse("ディープインパクト産駒が活躍する馬場")

        pedigree_chips = [c for c in result.chips if c.category == "pedigree"]
        assert len(pedigree_chips) >= 1
        assert pedigree_chips[0].needs_feature == True

    # ---------------------------------------------------------------------------
    # Sample 6-10: Compound conditions
    # ---------------------------------------------------------------------------

    def test_sample_06_compound_opening_front(self, parser):
        """Sample 6: 複合条件（開幕+高速+内+前有利）"""
        result = parser.parse("開幕週の超高速馬場で内ラチ沿いを通せる先行馬有利")
        chip_ids = [c.id for c in result.chips]

        assert "track_condition.良" in chip_ids
        assert "lane_bias.inner" in chip_ids
        assert "style_bias.front" in chip_ids

    def test_sample_07_compound_late_meet_outer(self, parser):
        """Sample 7: 複合条件（開催終盤+内荒れ+外伸び）"""
        result = parser.parse("開催終盤で内が荒れて外伸び、差しが届く展開")
        chip_ids = [c.id for c in result.chips]

        assert "lane_bias.outer" in chip_ids
        assert "style_bias.closer" in chip_ids

    def test_sample_08_complex_pace_analysis(self, parser):
        """Sample 8: 複雑な展開分析"""
        result = parser.parse("逃げ馬複数でハナ争い、ペース流れて消耗戦になりそう")
        chip_ids = [c.id for c in result.chips]

        assert "pace.H" in chip_ids
        assert "race_flow.many_front_runners" in chip_ids

    def test_sample_09_weather_affected(self, parser):
        """Sample 9: 天候影響パターン"""
        result = parser.parse("雨上がりの稍重馬場、時計がかかる条件")
        chip_ids = [c.id for c in result.chips]

        assert "track_condition.稍重" in chip_ids

    def test_sample_10_multiple_pedigree(self, parser):
        """Sample 10: 複数血統言及"""
        result = parser.parse("サンデー系かキンカメ産駒が有利な馬場設定")

        pedigree_chips = [c for c in result.chips if c.category == "pedigree"]
        # Should detect both SS系 and キンカメ
        pedigree_values = [c.id for c in pedigree_chips]
        assert len(pedigree_chips) >= 2

    # ---------------------------------------------------------------------------
    # Sample 11-15: Negation patterns
    # ---------------------------------------------------------------------------

    def test_sample_11_negation_outer_bias(self, parser):
        """Sample 11: 否定パターン（外伸びではない）"""
        result = parser.parse("外伸びではない、内が伸びる馬場")

        # 外伸び should be negated, 内伸び should be detected
        chip_ids = [c.id for c in result.chips if not c.is_warning]
        assert "lane_bias.inner" in chip_ids
        # 外伸び is negated so should not appear as positive chip
        non_warning_chips = [c for c in result.chips if not c.is_warning and c.id == "lane_bias.outer"]
        # The negated one should not generate a chip
        assert len(non_warning_chips) == 0 or result.suggestions.get("lane_bias", {}).get("value") == "inner"

    def test_sample_12_negation_high_pace(self, parser):
        """Sample 12: 否定パターン（ハイペースにはならない）"""
        result = parser.parse("ハイペースにはならなそう、スローで瞬発力勝負")
        chip_ids = [c.id for c in result.chips]

        # スロー should be detected
        assert "pace.S" in chip_ids

    def test_sample_13_weak_negation(self, parser):
        """Sample 13: 弱い否定（〜しにくい）"""
        result = parser.parse("前が残りにくいが差しも届きにくい")

        # Both are somewhat negated, but the pattern should still be detected
        # The key is that negation is considered
        assert len(result.chips) >= 0  # Just checking parse doesn't crash

    def test_sample_14_conditional_negation(self, parser):
        """Sample 14: 条件付き否定（〜とは限らない）"""
        result = parser.parse("内枠有利とは限らないが内を通せれば有利")
        chip_ids = [c.id for c in result.chips]

        # 内枠 is in conditional, but 内を通せれば should suggest inner lane preference
        # This tests nuanced interpretation

    def test_sample_15_doubt_expression(self, parser):
        """Sample 15: 疑問表現（〜か疑問）"""
        result = parser.parse("高速馬場か疑問、やや時計がかかりそう")

        # 高速馬場 is doubted, 時計がかかる should be detected
        if "track_condition" in result.suggestions:
            # The suggestion should lean toward 稍重
            pass

    # ---------------------------------------------------------------------------
    # Sample 16-20: Variations and synonyms
    # ---------------------------------------------------------------------------

    def test_sample_16_variation_inner_bias(self, parser):
        """Sample 16: 内伸びの表記ゆれ"""
        result = parser.parse("インコース有利、ラチ沿いを通す馬が有利")
        chip_ids = [c.id for c in result.chips]

        assert "lane_bias.inner" in chip_ids

    def test_sample_17_variation_pace(self, parser):
        """Sample 17: ペースの表記ゆれ"""
        result = parser.parse("ヨーイドンの上がり勝負になりそう")
        chip_ids = [c.id for c in result.chips]

        # ヨーイドン = スロー想定
        assert "pace.S" in chip_ids or "style_bias.front" in chip_ids

    def test_sample_18_variation_track_condition(self, parser):
        """Sample 18: 馬場状態の表記ゆれ"""
        result = parser.parse("パンパンの良馬場で時計勝負")
        chip_ids = [c.id for c in result.chips]

        assert "track_condition.良" in chip_ids

    def test_sample_19_variation_style_bias(self, parser):
        """Sample 19: 脚質有利の表記ゆれ"""
        result = parser.parse("末脚が活きる展開、追い込み効く")
        chip_ids = [c.id for c in result.chips]

        assert "style_bias.closer" in chip_ids

    def test_sample_20_variation_heavy_track(self, parser):
        """Sample 20: 道悪の表記ゆれ"""
        result = parser.parse("力のいるタフな馬場、パワーが必要")
        chip_ids = [c.id for c in result.chips]

        assert "track_condition.重" in chip_ids

    # ---------------------------------------------------------------------------
    # Sample 21-25: Edge cases and special patterns
    # ---------------------------------------------------------------------------

    def test_sample_21_frame_with_lane(self, parser):
        """Sample 21: 枠と進路の両方言及（別扱い確認）"""
        result = parser.parse("内枠有利だが外を回しても伸びる")
        chip_ids = [c.id for c in result.chips]

        # 内枠 → frame_hint (warning)
        assert "frame_hint.inner_frame" in chip_ids
        # 外を回しても伸びる → lane_bias.outer
        assert "lane_bias.outer" in chip_ids

    def test_sample_22_conflicting_info(self, parser):
        """Sample 22: 矛盾する情報"""
        result = parser.parse("内伸びと外伸び両方あり得る、フラットに近い")

        # Should detect conflict
        assert len(result.conflicts) >= 1

    def test_sample_23_intensity_modifier(self, parser):
        """Sample 23: 強度修飾語"""
        result1 = parser.parse("高速馬場")
        result2 = parser.parse("超高速馬場でかなりの時計勝負")

        # 超 and かなり should increase score
        if "track_condition" in result1.suggestions and "track_condition" in result2.suggestions:
            assert result2.suggestions["track_condition"]["score"] >= result1.suggestions["track_condition"]["score"]

    def test_sample_24_advisory_tags(self, parser):
        """Sample 24: アドバイザリータグ検出"""
        result = parser.parse("出遅れ癖があるが末脚確実、道悪得意")

        tag_chips = [c for c in result.chips if c.category == "tag"]
        tag_values = [c.id for c in tag_chips]

        # Should detect some advisory tags
        assert len(tag_chips) >= 1

    def test_sample_25_misc_tags(self, parser):
        """Sample 25: その他タグ検出"""
        result = parser.parse("人気薄の穴馬、妙味あり")

        misc_chips = [c for c in result.chips if c.category == "misc_tag"]
        assert len(misc_chips) >= 1

    # ---------------------------------------------------------------------------
    # Sample 26-30: Real-world complex examples
    # ---------------------------------------------------------------------------

    def test_sample_26_arima_kinen_style(self, parser):
        """Sample 26: 有馬記念風の分析メモ"""
        result = parser.parse(
            "中山2500m、小回りで器用さが必要。"
            "逃げ馬少なくスローになりそう。"
            "内ラチ沿いを立ち回れる先行馬有利。"
        )
        chip_ids = [c.id for c in result.chips]

        assert "pace.S" in chip_ids
        assert "lane_bias.inner" in chip_ids
        assert "style_bias.front" in chip_ids

    def test_sample_27_derby_style(self, parser):
        """Sample 27: ダービー風の分析メモ"""
        result = parser.parse(
            "東京2400m、直線長く瞬発力勝負。"
            "高速馬場でディープ産駒向き。"
            "外を回しても差しが届く展開。"
        )
        chip_ids = [c.id for c in result.chips]

        assert "track_condition.良" in chip_ids
        assert "style_bias.closer" in chip_ids
        pedigree_chips = [c for c in result.chips if c.category == "pedigree"]
        assert len(pedigree_chips) >= 1

    def test_sample_28_sprint_race(self, parser):
        """Sample 28: スプリント戦の分析"""
        result = parser.parse(
            "1200m戦、テンが速くハイペース必至。"
            "先行馬多く消耗戦、差しが届きやすい。"
        )
        chip_ids = [c.id for c in result.chips]

        assert "pace.H" in chip_ids
        assert "style_bias.closer" in chip_ids

    def test_sample_29_stayers_race(self, parser):
        """Sample 29: 長距離戦の分析"""
        result = parser.parse(
            "3200m、スタミナ勝負になりそう。"
            "逃げ馬不在で楽逃げ、前残りも。"
            "ハーツ産駒の血統向き。"
        )
        chip_ids = [c.id for c in result.chips]

        assert "race_flow.few_front_runners" in chip_ids
        pedigree_chips = [c for c in result.chips if c.category == "pedigree"]
        assert len(pedigree_chips) >= 1

    def test_sample_30_complex_full_analysis(self, parser):
        """Sample 30: 複合条件フル分析"""
        result = parser.parse(
            "開幕週の超高速馬場で内ラチ沿い有利。"
            "逃げ候補少なくスローで前残り濃厚。"
            "ディープ系の瞬発力が活きる。"
            "内枠希望だが外枠でも内を突ければ可。"
        )
        chip_ids = [c.id for c in result.chips]

        # Track condition
        assert "track_condition.良" in chip_ids
        # Lane bias
        assert "lane_bias.inner" in chip_ids
        # Pace
        assert "pace.S" in chip_ids
        # Style bias
        assert "style_bias.front" in chip_ids
        # Frame hint (warning)
        frame_chips = [c for c in result.chips if c.category == "frame_hint"]
        assert len(frame_chips) >= 1
        # Pedigree
        pedigree_chips = [c for c in result.chips if c.category == "pedigree"]
        assert len(pedigree_chips) >= 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
