# -*- coding: utf-8 -*-
"""
src/feature_audit - Feature Audit Module

全バージョン横断の特徴量棚卸し（audit）機能を提供するモジュール。

【機能】
1. 全バージョン自動検出 + 実行
2. 共通フォーマットでの成果物出力
3. pre_race安全性ラベル付け
4. adapter方式による既存コードとの統合
"""

from .safety import (
    classify_feature_safety,
    classify_features_batch,
    get_unsafe_features,
    get_warn_features,
    get_safe_features,
    summarize_safety,
    strip_bridge_prefix,
    BRIDGE_PREFIX,
    UNSAFE_EXACT,
    UNSAFE_PATTERNS,
    WARN_PATTERNS,
)

from .adapters import (
    BaseFeatureAdapter,
    AdapterResult,
    get_available_adapters,
    get_adapter,
    register_adapter,
    detect_all_versions,
    ADAPTER_REGISTRY,
)

__all__ = [
    # Safety
    "classify_feature_safety",
    "classify_features_batch",
    "get_unsafe_features",
    "get_warn_features",
    "get_safe_features",
    "summarize_safety",
    "strip_bridge_prefix",
    "BRIDGE_PREFIX",
    "UNSAFE_EXACT",
    "UNSAFE_PATTERNS",
    "WARN_PATTERNS",
    # Adapters
    "BaseFeatureAdapter",
    "AdapterResult",
    "get_available_adapters",
    "get_adapter",
    "register_adapter",
    "detect_all_versions",
    "ADAPTER_REGISTRY",
]
