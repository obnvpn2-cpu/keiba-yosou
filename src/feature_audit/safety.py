# -*- coding: utf-8 -*-
"""
safety.py - Pre-race Safety Label Assignment for Features

pre_race運用時に「使ってはいけない/警戒すべき」特徴量を判定するモジュール。
棚卸し（audit）用であり、自動除外は行わない。可視化のためのラベル付けのみ。

【ラベル定義】
- unsafe: pre_race運用で使用禁止（当日情報、オッズ等）
- warn: 警戒が必要（結果情報っぽい名前、確定情報）
- safe: 上記以外

【使い方】
>>> from src.feature_audit.safety import classify_feature_safety
>>> label, notes = classify_feature_safety("h_body_weight")
>>> print(label)  # "unsafe"
>>> print(notes)  # "race-day body weight information"
"""

import re
from typing import Tuple, List, Optional

# =============================================================================
# Bridge Feature Prefix (F-2)
# =============================================================================

# Bridge features from v3 have this prefix
BRIDGE_PREFIX = "v4_bridge_"


# =============================================================================
# Safety Rules Definition
# =============================================================================

# 完全一致で unsafe 判定
UNSAFE_EXACT = {
    # 当日体重情報（レース当日にしか分からない）
    "h_body_weight",
    "h_body_weight_diff",
    "h_body_weight_dev",
    "horse_weight",
    "horse_weight_diff",
    "body_weight",
    "body_weight_diff",
    "avg_horse_weight",  # Legacy: 当日平均体重
    # 市場情報（オッズ・人気）
    "market_win_odds",
    "market_popularity",
    "win_odds",
    "popularity",
    "odds",
    # 結果情報
    "finish_order",
    "finish_position",
    "time_sec",
    "last_3f",
    # Legacy: 払戻・着順関連
    "fukusho_payout",
    "payout_count",
    "paid_places",
    "should_have_payout",
}

# プレフィックス・パターンで unsafe 判定
UNSAFE_PATTERNS = [
    (r"^market_", "market information (odds/popularity)"),
    (r"^h_body_weight", "race-day body weight information"),
    (r"^odds_", "odds information"),
    (r"_weight_diff$", "weight difference (race-day info)"),
    (r"^paid_", "payout/paid information"),
]

# warn 判定用パターン（結果情報っぽい名前）
WARN_PATTERNS = [
    (r"result", "contains 'result'"),
    (r"finish", "contains 'finish'"),
    (r"着順", "contains '着順' (finish order)"),
    (r"払戻", "contains '払戻' (payout)"),
    (r"payout", "contains 'payout'"),
    (r"確定", "contains '確定' (finalized)"),
    (r"after", "contains 'after'"),
    (r"post", "contains 'post'"),
    (r"outcome", "contains 'outcome'"),
    (r"target_", "contains 'target_' (label column)"),
    (r"^y_", "starts with 'y_' (possible label)"),
    # Legacy: 馬場・コンディション（シナリオレイヤーで処理）
    (r"^track_condition", "track condition (handled by scenario layer)"),
    (r"^baba_", "track condition (baba, handled by scenario layer)"),
    # Legacy: ID列（特徴量としては不適切）
    (r"_id$", "ID column (should not be feature)"),
    (r"^horse_no$", "horse number (post-draw information)"),
    (r"^umaban$", "horse number (post-draw information)"),
    (r"^waku$", "frame number (post-draw information)"),
]


def strip_bridge_prefix(feature_name: str) -> Tuple[str, bool]:
    """
    Bridge prefix を除去して元の特徴量名を返す

    Args:
        feature_name: 特徴量名（v4_bridge_* or native）

    Returns:
        (original_name, is_bridged)
        - original_name: prefix を除去した名前
        - is_bridged: bridge 特徴量だった場合 True
    """
    if feature_name.startswith(BRIDGE_PREFIX):
        return feature_name[len(BRIDGE_PREFIX):], True
    return feature_name, False


def classify_feature_safety(
    feature_name: str,
    additional_unsafe: Optional[List[str]] = None,
    additional_warn_patterns: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[str, str]:
    """
    特徴量名から安全性ラベルを判定する

    Args:
        feature_name: 特徴量名
        additional_unsafe: 追加のunsafe完全一致リスト
        additional_warn_patterns: 追加のwarnパターン [(pattern, reason), ...]

    Returns:
        (label, notes) のタプル
        - label: "unsafe", "warn", "safe" のいずれか
        - notes: 判定理由（空文字列の場合もある）

    Note:
        F-2: v4_bridge_* 特徴量は prefix を除去してから元の v3 名で判定する。
        これにより bridge 特徴量の safety が元の v3 特徴量から継承される。
    """
    # F-2: Strip bridge prefix if present to classify based on original name
    original_name, is_bridged = strip_bridge_prefix(feature_name)
    name_lower = original_name.lower()

    # 1. 完全一致でunsafe判定
    unsafe_set = UNSAFE_EXACT.copy()
    if additional_unsafe:
        unsafe_set.update(additional_unsafe)

    if original_name in unsafe_set or name_lower in unsafe_set:
        reason = _get_unsafe_reason(original_name)
        if is_bridged:
            reason = f"[bridged from v3] {reason}"
        return "unsafe", reason

    # 2. パターンでunsafe判定
    for pattern, reason in UNSAFE_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            if is_bridged:
                reason = f"[bridged from v3] {reason}"
            return "unsafe", reason

    # 3. パターンでwarn判定
    warn_patterns = WARN_PATTERNS.copy()
    if additional_warn_patterns:
        warn_patterns.extend(additional_warn_patterns)

    for pattern, reason in warn_patterns:
        if re.search(pattern, name_lower, re.IGNORECASE):
            if is_bridged:
                reason = f"[bridged from v3] {reason}"
            return "warn", reason

    # 4. それ以外はsafe
    notes = ""
    if is_bridged:
        notes = "[bridged from v3]"
    return "safe", notes


def _get_unsafe_reason(feature_name: str) -> str:
    """unsafe特徴量の理由を返す"""
    name_lower = feature_name.lower()

    if "body_weight" in name_lower or "horse_weight" in name_lower:
        return "race-day body weight information"
    if "market" in name_lower or "odds" in name_lower:
        return "market information (odds/popularity)"
    if "popularity" in name_lower:
        return "market popularity information"
    if "finish" in name_lower:
        return "finish result information"
    if "fukusho" in name_lower or "payout" in name_lower:
        return "payout information (post-race)"
    if "paid_" in name_lower:
        return "paid/payout information (post-race)"
    if "avg_horse_weight" in name_lower:
        return "average horse weight (race-day)"

    return "pre_race unsafe feature"


def classify_features_batch(
    feature_names: List[str],
    additional_unsafe: Optional[List[str]] = None,
    additional_warn_patterns: Optional[List[Tuple[str, str]]] = None,
) -> List[dict]:
    """
    複数の特徴量を一括で判定する

    Args:
        feature_names: 特徴量名のリスト
        additional_unsafe: 追加のunsafe完全一致リスト
        additional_warn_patterns: 追加のwarnパターン

    Returns:
        [{"feature": name, "safety_label": label, "notes": notes}, ...]
    """
    results = []
    for name in feature_names:
        label, notes = classify_feature_safety(
            name,
            additional_unsafe=additional_unsafe,
            additional_warn_patterns=additional_warn_patterns,
        )
        results.append({
            "feature": name,
            "safety_label": label,
            "notes": notes,
        })
    return results


def get_unsafe_features(feature_names: List[str]) -> List[str]:
    """unsafe判定された特徴量のみを返す"""
    return [
        name for name in feature_names
        if classify_feature_safety(name)[0] == "unsafe"
    ]


def get_warn_features(feature_names: List[str]) -> List[str]:
    """warn判定された特徴量のみを返す"""
    return [
        name for name in feature_names
        if classify_feature_safety(name)[0] == "warn"
    ]


def get_safe_features(feature_names: List[str]) -> List[str]:
    """safe判定された特徴量のみを返す"""
    return [
        name for name in feature_names
        if classify_feature_safety(name)[0] == "safe"
    ]


def summarize_safety(feature_names: List[str]) -> dict:
    """
    安全性サマリーを返す

    Returns:
        {
            "n_total": int,
            "n_unsafe": int,
            "n_warn": int,
            "n_safe": int,
            "unsafe_features": [str, ...],
            "warn_features": [str, ...],
        }
    """
    unsafe = []
    warn = []
    safe = []

    for name in feature_names:
        label, _ = classify_feature_safety(name)
        if label == "unsafe":
            unsafe.append(name)
        elif label == "warn":
            warn.append(name)
        else:
            safe.append(name)

    return {
        "n_total": len(feature_names),
        "n_unsafe": len(unsafe),
        "n_warn": len(warn),
        "n_safe": len(safe),
        "unsafe_features": unsafe,
        "warn_features": warn,
    }
