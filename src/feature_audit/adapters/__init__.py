# -*- coding: utf-8 -*-
"""
Feature Audit Adapters

各バージョンの特徴量パックを統一インターフェースで扱うためのアダプター群。
"""

from .base import BaseFeatureAdapter, AdapterResult
from .registry import (
    get_available_adapters,
    get_adapter,
    register_adapter,
    detect_all_versions,
    ADAPTER_REGISTRY,
)

__all__ = [
    "BaseFeatureAdapter",
    "AdapterResult",
    "get_available_adapters",
    "get_adapter",
    "register_adapter",
    "detect_all_versions",
    "ADAPTER_REGISTRY",
]
