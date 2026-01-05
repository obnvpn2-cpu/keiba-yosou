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
}

# プレフィックス・パターンで unsafe 判定
UNSAFE_PATTERNS = [
    (r"^market_", "market information (odds/popularity)"),
    (r"^h_body_weight", "race-day body weight information"),
    (r"^odds_", "odds information"),
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
]


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
    """
    name_lower = feature_name.lower()

    # 1. 完全一致でunsafe判定
    unsafe_set = UNSAFE_EXACT.copy()
    if additional_unsafe:
        unsafe_set.update(additional_unsafe)

    if feature_name in unsafe_set or name_lower in unsafe_set:
        return "unsafe", _get_unsafe_reason(feature_name)

    # 2. パターンでunsafe判定
    for pattern, reason in UNSAFE_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return "unsafe", reason

    # 3. パターンでwarn判定
    warn_patterns = WARN_PATTERNS.copy()
    if additional_warn_patterns:
        warn_patterns.extend(additional_warn_patterns)

    for pattern, reason in warn_patterns:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return "warn", reason

    # 4. それ以外はsafe
    return "safe", ""


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
