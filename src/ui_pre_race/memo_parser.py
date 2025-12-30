"""
Memo Parser Engine for Pre-race Scenario UI

メモテキストからキーワードを抽出し、提案チップを生成する。
自動反映はせず、ユーザーがクリックしたものだけフォームに反映する設計。

Usage:
    from src.ui_pre_race.memo_parser import MemoParser

    parser = MemoParser()
    result = parser.parse("開幕週の高速馬場で内枠、前有利になってる。")

    # result.chips - 提案チップのリスト
    # result.conflicts - 矛盾の警告リスト
    # result.suggestions - 数値スコアによる推定値
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Default rules path
DEFAULT_RULES_PATH = Path(__file__).parent.parent.parent / "config" / "memo_rules" / "pre_race_rules.yaml"


@dataclass
class Chip:
    """提案チップ"""
    id: str                          # 一意識別子
    label: str                       # 表示ラベル
    reason: str                      # 提案理由
    category: str                    # カテゴリ（track_condition, lane_bias, style_bias, pace, pedigree, tag）
    apply_payload: Dict[str, Any]    # 適用時のペイロード（フォームへの反映値）
    confidence: float = 1.0          # 信頼度（0.0〜1.0）
    needs_feature: bool = False      # 特徴量化が必要な場合True
    is_warning: bool = False         # 警告チップの場合True


@dataclass
class Conflict:
    """矛盾警告"""
    pair: Tuple[str, str]            # 矛盾するカテゴリペア
    message: str                     # 警告メッセージ
    chips: List[str]                 # 関連するチップID


@dataclass
class ParseResult:
    """パース結果"""
    chips: List[Chip] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    suggestions: Dict[str, Any] = field(default_factory=dict)
    raw_matches: List[Dict[str, Any]] = field(default_factory=list)


class MemoParser:
    """メモパーサー"""

    def __init__(self, rules_path: Optional[Path] = None):
        """
        初期化

        Args:
            rules_path: YAMLルールファイルのパス（デフォルトはconfig/memo_rules/pre_race_rules.yaml）
        """
        self.rules_path = rules_path or DEFAULT_RULES_PATH
        self.rules = self._load_rules()

        # 否定パターンをコンパイル
        self.negation_patterns = self.rules.get("negation_patterns", [])

        # 強度修飾語を展開
        self.intensity_modifiers = self._build_intensity_map()

    def _load_rules(self) -> Dict[str, Any]:
        """ルールファイルを読み込む"""
        if not self.rules_path.exists():
            logger.warning(f"Rules file not found: {self.rules_path}")
            return {}

        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            return {}

    def _build_intensity_map(self) -> Dict[str, float]:
        """強度修飾語 → 倍率のマップを構築"""
        result = {}
        for level, config in self.rules.get("intensity_modifiers", {}).items():
            multiplier = config.get("multiplier", 1.0)
            for word in config.get("words", []):
                result[word] = multiplier
        return result

    def _check_negation(self, text: str, keyword: str, match_pos: int) -> bool:
        """
        キーワード周辺に否定パターンがあるかチェック

        Args:
            text: 全文
            keyword: マッチしたキーワード
            match_pos: マッチ位置

        Returns:
            True if negated
        """
        keyword_end = match_pos + len(keyword)

        # キーワードの直後をチェック（「〜ではない」「〜じゃない」等）
        # 直後5文字程度をチェック
        after_context = text[keyword_end:min(len(text), keyword_end + 6)]

        # 直後パターン（キーワードの直後に来る否定）
        after_negation_patterns = [
            "ではない", "じゃない", "でない",
            "なさそう", "ないだろう", "ないかも",
            "にくい", "しにくい", "づらい", "しづらい",
            "難しい", "厳しい",
            "ほどでは", "とは言え", "とは思え", "とは考え",
        ]

        for pattern in after_negation_patterns:
            if after_context.startswith(pattern) or pattern in after_context[:5]:
                return True

        # キーワードの直前をチェック（ごく限定的）
        before_context = text[max(0, match_pos - 5):match_pos]

        # 直前パターン（まれ）
        before_negation_patterns = [
            "ない", "無い",  # 「ない内伸び」のようなケース（まれ）
        ]

        for pattern in before_negation_patterns:
            if before_context.endswith(pattern):
                return True

        # 「〜か疑問」「〜か微妙」のような曖昧表現をチェック（直後2-5文字）
        ambiguous_patterns = ["疑問", "微妙", "怪しい", "不透明"]
        extended_after = text[keyword_end:min(len(text), keyword_end + 8)]
        for pattern in ambiguous_patterns:
            if pattern in extended_after:
                return True

        return False

    def _get_intensity_multiplier(self, text: str, keyword: str, match_pos: int) -> float:
        """
        キーワード直前の強度修飾語を検出

        Returns:
            強度倍率（デフォルト1.0）
        """
        # キーワードの直前5文字程度を確認
        start = max(0, match_pos - 5)
        prefix = text[start:match_pos]

        for word, multiplier in self.intensity_modifiers.items():
            if prefix.endswith(word):
                return multiplier

        return 1.0

    def _extract_category_matches(
        self,
        text: str,
        category: str,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        特定カテゴリのキーワードマッチを抽出

        Args:
            text: メモテキスト
            category: カテゴリ名（track_condition, lane_bias等）
            config: カテゴリの設定辞書

        Returns:
            マッチ情報のリスト
        """
        matches = []

        for value, value_config in config.items():
            keywords = value_config.get("keywords", [])
            label = value_config.get("label", value)
            reason = value_config.get("reason", "")

            for keyword in keywords:
                # 全てのマッチ位置を検索
                pos = 0
                while True:
                    idx = text.find(keyword, pos)
                    if idx == -1:
                        break

                    # 否定チェック
                    is_negated = self._check_negation(text, keyword, idx)

                    # 強度チェック
                    intensity = self._get_intensity_multiplier(text, keyword, idx)

                    matches.append({
                        "category": category,
                        "value": value,
                        "keyword": keyword,
                        "position": idx,
                        "is_negated": is_negated,
                        "intensity": intensity,
                        "label": label,
                        "reason": reason,
                    })

                    pos = idx + len(keyword)

        return matches

    def _extract_pedigree_matches(self, text: str) -> List[Dict[str, Any]]:
        """血統キーワードを抽出（needs_feature付き）"""
        matches = []
        pedigree_config = self.rules.get("pedigree", {})
        keywords_by_type = pedigree_config.get("keywords", {})

        for pedigree_type, keywords in keywords_by_type.items():
            for keyword in keywords:
                if keyword in text:
                    idx = text.find(keyword)
                    is_negated = self._check_negation(text, keyword, idx)

                    matches.append({
                        "category": "pedigree",
                        "value": pedigree_type,
                        "keyword": keyword,
                        "position": idx,
                        "is_negated": is_negated,
                        "intensity": 1.0,
                        "label": f"血統：{pedigree_type}系言及",
                        "reason": f"血統キーワード「{keyword}」を検出",
                        "needs_feature": True,
                    })

        return matches

    def _extract_frame_hint_matches(self, text: str) -> List[Dict[str, Any]]:
        """枠番ヒントを抽出（lane_biasとは別扱い、警告付き）"""
        matches = []
        frame_hint_config = self.rules.get("frame_hint", {})

        for value, value_config in frame_hint_config.items():
            keywords = value_config.get("keywords", [])
            label = value_config.get("label", value)
            reason = value_config.get("reason", "")
            warning_message = value_config.get("warning_message", "")

            for keyword in keywords:
                if keyword in text:
                    idx = text.find(keyword)
                    is_negated = self._check_negation(text, keyword, idx)

                    matches.append({
                        "category": "frame_hint",
                        "value": value,
                        "keyword": keyword,
                        "position": idx,
                        "is_negated": is_negated,
                        "intensity": 1.0,
                        "label": label,
                        "reason": reason,
                        "is_warning": True,
                        "warning_message": warning_message,
                    })

        return matches

    def _extract_race_flow_matches(self, text: str) -> List[Dict[str, Any]]:
        """レースフロー（展開予測）を抽出"""
        matches = []
        race_flow_config = self.rules.get("race_flow", {})

        for value, value_config in race_flow_config.items():
            keywords = value_config.get("keywords", [])
            label = value_config.get("label", value)
            reason = value_config.get("reason", "")

            for keyword in keywords:
                if keyword in text:
                    idx = text.find(keyword)
                    is_negated = self._check_negation(text, keyword, idx)
                    intensity = self._get_intensity_multiplier(text, keyword, idx)

                    matches.append({
                        "category": "race_flow",
                        "value": value,
                        "keyword": keyword,
                        "position": idx,
                        "is_negated": is_negated,
                        "intensity": intensity,
                        "label": label,
                        "reason": reason,
                    })

        return matches

    def _extract_misc_tags(self, text: str) -> List[Dict[str, Any]]:
        """その他タグを抽出"""
        matches = []
        misc_config = self.rules.get("misc_tags", {})

        for polarity in ["positive", "negative", "neutral"]:
            for item in misc_config.get(polarity, []):
                pattern = item.get("pattern", "")
                tag = item.get("tag", "")

                if pattern and pattern in text:
                    idx = text.find(pattern)
                    is_negated = self._check_negation(text, pattern, idx)

                    # 否定されている場合、正負を反転
                    actual_polarity = polarity
                    if is_negated and polarity != "neutral":
                        actual_polarity = "negative" if polarity == "positive" else "positive"

                    matches.append({
                        "category": "misc_tag",
                        "value": tag,
                        "keyword": pattern,
                        "position": idx,
                        "is_negated": is_negated,
                        "intensity": 1.0,
                        "label": f"情報：{tag}",
                        "reason": f"パターン「{pattern}」を検出",
                        "polarity": actual_polarity,
                    })

        return matches

    def _extract_advisory_tags(self, text: str) -> List[Dict[str, Any]]:
        """アドバイザリータグを抽出"""
        matches = []
        tags_config = self.rules.get("advisory_tags", {})

        for polarity in ["positive", "negative"]:
            for item in tags_config.get(polarity, []):
                pattern = item.get("pattern", "")
                tag = item.get("tag", "")

                if pattern and pattern in text:
                    idx = text.find(pattern)
                    is_negated = self._check_negation(text, pattern, idx)

                    # 否定されている場合、正負を反転
                    actual_polarity = polarity
                    if is_negated:
                        actual_polarity = "negative" if polarity == "positive" else "positive"

                    matches.append({
                        "category": "tag",
                        "value": tag,
                        "keyword": pattern,
                        "position": idx,
                        "is_negated": is_negated,
                        "intensity": 1.0,
                        "label": f"タグ：{tag}",
                        "reason": f"パターン「{pattern}」を検出",
                        "polarity": actual_polarity,
                    })

        # 騎手関連タグ
        jockey_config = self.rules.get("jockey", {})
        for item in jockey_config.get("advisory_tags", []):
            pattern = item.get("pattern", "")
            tag = item.get("tag", "")

            if pattern and pattern in text:
                idx = text.find(pattern)
                matches.append({
                    "category": "tag",
                    "value": tag,
                    "keyword": pattern,
                    "position": idx,
                    "is_negated": False,
                    "intensity": 1.0,
                    "label": f"タグ：{tag}",
                    "reason": f"パターン「{pattern}」を検出",
                    "polarity": "neutral",
                })

        return matches

    def _get_negation_label(self, category: str, value: str, original_label: str) -> str:
        """
        否定チップ用のラベルを生成

        Args:
            category: カテゴリ名
            value: 値
            original_label: 元のラベル

        Returns:
            否定ラベル（例: "外伸び否定"）
        """
        # カテゴリ別の日本語ラベル
        category_labels = {
            "lane_bias": {
                "inner": "内伸び",
                "middle": "中央",
                "outer": "外伸び",
            },
            "style_bias": {
                "front": "前残り",
                "closer": "差し届き",
            },
            "pace": {
                "S": "スロー",
                "M": "ミドル",
                "H": "ハイペース",
            },
            "track_condition": {
                "良": "良馬場",
                "稍重": "稍重",
                "重": "重馬場",
                "不良": "不良馬場",
            },
        }

        if category in category_labels and value in category_labels[category]:
            base_label = category_labels[category][value]
        elif original_label:
            base_label = original_label
        else:
            base_label = f"{category}.{value}"

        return f"{base_label}否定"

    def _detect_conflicts(self, matches: List[Dict[str, Any]]) -> List[Conflict]:
        """矛盾を検出"""
        conflicts = []
        conflict_rules = self.rules.get("conflict_rules", [])

        # カテゴリ.値 のセットを作成
        matched_keys = set()
        for m in matches:
            if not m.get("is_negated", False):
                key = f"{m['category']}.{m['value']}"
                matched_keys.add(key)

        for rule in conflict_rules:
            pair = tuple(rule.get("pair", []))
            if len(pair) == 2 and pair[0] in matched_keys and pair[1] in matched_keys:
                conflicts.append(Conflict(
                    pair=pair,
                    message=rule.get("message", "矛盾が検出されました"),
                    chips=[pair[0], pair[1]],
                ))

        return conflicts

    def _calculate_suggestions(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        マッチ結果から推定スコアを計算

        Returns:
            カテゴリごとの推定値とスコア
        """
        scores = {
            "lane_bias": {"inner": 0.0, "middle": 0.0, "outer": 0.0},
            "style_bias": {"front": 0.0, "closer": 0.0},
            "pace": {"S": 0.0, "M": 0.0, "H": 0.0},
            "track_condition": {"良": 0.0, "稍重": 0.0, "重": 0.0, "不良": 0.0},
        }

        for m in matches:
            category = m["category"]
            value = m["value"]

            if category in scores and value in scores[category]:
                # 否定されていたらスコアを減らす（反対側に加算でもよいが、今は0にする）
                if m.get("is_negated", False):
                    scores[category][value] -= m["intensity"]
                else:
                    scores[category][value] += m["intensity"]

        # 各カテゴリで最大スコアの値を推定値とする
        suggestions = {}
        for category, category_scores in scores.items():
            if any(v != 0 for v in category_scores.values()):
                best_value = max(category_scores.items(), key=lambda x: x[1])
                if best_value[1] > 0:
                    suggestions[category] = {
                        "value": best_value[0],
                        "score": best_value[1],
                        "all_scores": category_scores,
                    }

        return suggestions

    def _build_chips(self, matches: List[Dict[str, Any]], conflicts: List[Conflict]) -> List[Chip]:
        """マッチ結果からチップを生成"""
        chips = []
        seen_ids = set()

        for m in matches:
            category = m["category"]
            value = m["value"]
            chip_id = f"{category}.{value}"

            # 重複を除外
            if chip_id in seen_ids:
                continue
            seen_ids.add(chip_id)

            # 否定されている場合は「否定タグ」として警告チップを生成
            # 重要: 否定から反対側を推論しない（例：外伸びではない → inner にしない）
            if m.get("is_negated", False):
                # 否定チップを生成（apply_payloadは空 = 自動反映しない）
                negation_chip_id = f"negated.{category}.{value}"
                if negation_chip_id not in seen_ids:
                    seen_ids.add(negation_chip_id)
                    # ラベル生成
                    negation_label = self._get_negation_label(category, value, m.get("label", ""))
                    chips.append(Chip(
                        id=negation_chip_id,
                        label=negation_label,
                        reason=f"「{m.get('keyword', '')}」が否定されています",
                        category="negation",
                        apply_payload={},  # 空 = 自動適用しない
                        confidence=m.get("intensity", 1.0),
                        needs_feature=False,
                        is_warning=True,
                    ))
                continue

            # apply_payload を構築
            apply_payload = {}
            if category == "track_condition":
                apply_payload = {"track_condition": value}
            elif category == "lane_bias":
                apply_payload = {"lane_bias": value}
            elif category == "style_bias":
                apply_payload = {"style_bias": value}
            elif category == "pace":
                apply_payload = {"pace": value}
            elif category == "pedigree":
                apply_payload = {}  # 血統は適用しない
            elif category == "frame_hint":
                apply_payload = {}  # 枠ヒントはlane_biasに直結させない
            elif category == "race_flow":
                apply_payload = {}  # レースフローは参考情報
            elif category == "tag":
                apply_payload = {"tag": value}
            elif category == "misc_tag":
                apply_payload = {}  # その他タグは参考情報

            chips.append(Chip(
                id=chip_id,
                label=m["label"],
                reason=m["reason"],
                category=category,
                apply_payload=apply_payload,
                confidence=min(1.0, m.get("intensity", 1.0)),
                needs_feature=m.get("needs_feature", False),
                is_warning=m.get("is_warning", False),
            ))

        # 矛盾がある場合は警告チップを追加
        for conflict in conflicts:
            chips.append(Chip(
                id=f"conflict.{conflict.pair[0]}_{conflict.pair[1]}",
                label="矛盾注意",
                reason=conflict.message,
                category="conflict",
                apply_payload={},
                confidence=1.0,
                is_warning=True,
            ))

        return chips

    def parse(self, memo_text: str) -> ParseResult:
        """
        メモテキストをパースして提案チップを生成

        Args:
            memo_text: メモテキスト

        Returns:
            ParseResult（chips, conflicts, suggestions, raw_matches）
        """
        if not memo_text or not memo_text.strip():
            return ParseResult()

        text = memo_text.strip()
        all_matches = []

        # 各カテゴリのマッチを抽出
        for category in ["track_condition", "lane_bias", "style_bias", "pace"]:
            config = self.rules.get(category, {})
            matches = self._extract_category_matches(text, category, config)
            all_matches.extend(matches)

        # 枠番ヒント（lane_biasとは別扱い）
        frame_hint_matches = self._extract_frame_hint_matches(text)
        all_matches.extend(frame_hint_matches)

        # レースフロー
        race_flow_matches = self._extract_race_flow_matches(text)
        all_matches.extend(race_flow_matches)

        # 血統マッチ
        pedigree_matches = self._extract_pedigree_matches(text)
        all_matches.extend(pedigree_matches)

        # アドバイザリータグ
        tag_matches = self._extract_advisory_tags(text)
        all_matches.extend(tag_matches)

        # その他タグ
        misc_tag_matches = self._extract_misc_tags(text)
        all_matches.extend(misc_tag_matches)

        # 矛盾検出
        conflicts = self._detect_conflicts(all_matches)

        # スコア計算
        suggestions = self._calculate_suggestions(all_matches)

        # チップ生成
        chips = self._build_chips(all_matches, conflicts)

        return ParseResult(
            chips=chips,
            conflicts=conflicts,
            suggestions=suggestions,
            raw_matches=all_matches,
        )

    def to_dict(self, result: ParseResult) -> Dict[str, Any]:
        """ParseResultを辞書に変換（JSON出力用）"""
        return {
            "chips": [
                {
                    "id": c.id,
                    "label": c.label,
                    "reason": c.reason,
                    "category": c.category,
                    "apply_payload": c.apply_payload,
                    "confidence": c.confidence,
                    "needs_feature": c.needs_feature,
                    "is_warning": c.is_warning,
                }
                for c in result.chips
            ],
            "conflicts": [
                {
                    "pair": list(c.pair),
                    "message": c.message,
                    "chips": c.chips,
                }
                for c in result.conflicts
            ],
            "suggestions": result.suggestions,
        }


# Convenience function
def parse_memo(memo_text: str, rules_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    メモをパースして辞書を返すユーティリティ関数

    Args:
        memo_text: メモテキスト
        rules_path: ルールファイルパス（オプション）

    Returns:
        パース結果の辞書
    """
    parser = MemoParser(rules_path)
    result = parser.parse(memo_text)
    return parser.to_dict(result)


if __name__ == "__main__":
    # テスト実行
    test_memos = [
        "開幕週の高速馬場で内枠、前有利になってる。瞬発力のあるディープインパクト系の血統が得意な軽い芝。",
        "ハイペースで差しが届きそう。外伸びの馬場。",
        "スローペースで前残り。ただし内も外も伸びる。",  # 矛盾例
    ]

    parser = MemoParser()

    for memo in test_memos:
        print(f"\n{'='*60}")
        print(f"メモ: {memo}")
        print(f"{'='*60}")

        result = parser.parse(memo)

        print("\n[チップ]")
        for chip in result.chips:
            prefix = "!" if chip.is_warning else ("*" if chip.needs_feature else "-")
            print(f"  {prefix} {chip.label}")
            print(f"    理由: {chip.reason}")
            if chip.apply_payload:
                print(f"    適用: {chip.apply_payload}")

        if result.conflicts:
            print("\n[矛盾]")
            for conflict in result.conflicts:
                print(f"  ! {conflict.message}")

        print("\n[推定値]")
        for category, suggestion in result.suggestions.items():
            print(f"  {category}: {suggestion['value']} (score={suggestion['score']:.1f})")
