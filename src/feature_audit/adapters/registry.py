# -*- coding: utf-8 -*-
"""
registry.py - Adapter Registry for Feature Audit

利用可能なアダプターを管理・検出するレジストリ。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

from .base import BaseFeatureAdapter


# グローバルレジストリ
ADAPTER_REGISTRY: Dict[str, Type[BaseFeatureAdapter]] = {}


def register_adapter(version_name: str):
    """
    アダプタークラスをレジストリに登録するデコレータ

    Usage:
        @register_adapter("v4")
        class V4Adapter(BaseFeatureAdapter):
            ...
    """
    def decorator(cls: Type[BaseFeatureAdapter]):
        ADAPTER_REGISTRY[version_name] = cls
        return cls
    return decorator


def get_available_adapters() -> List[str]:
    """
    登録されているアダプターのバージョン名一覧を取得

    Returns:
        バージョン名のリスト (e.g., ["v4", "legacy"])
    """
    return list(ADAPTER_REGISTRY.keys())


def get_adapter(
    version_name: str,
    db_path: Optional[Path] = None,
    models_dir: Optional[Path] = None,
) -> Optional[BaseFeatureAdapter]:
    """
    指定バージョンのアダプターインスタンスを取得

    Args:
        version_name: バージョン名
        db_path: データベースパス
        models_dir: モデルディレクトリ

    Returns:
        アダプターインスタンス、未登録の場合はNone
    """
    adapter_cls = ADAPTER_REGISTRY.get(version_name)
    if adapter_cls is None:
        return None
    return adapter_cls(db_path=db_path, models_dir=models_dir)


def detect_all_versions(
    db_path: Optional[Path] = None,
    models_dir: Optional[Path] = None,
) -> List[Tuple[str, bool, str]]:
    """
    全登録アダプターについて実行可能性を検出

    Args:
        db_path: データベースパス
        models_dir: モデルディレクトリ

    Returns:
        [(version_name, can_run, reason), ...]
    """
    results = []
    for version_name in ADAPTER_REGISTRY:
        adapter = get_adapter(version_name, db_path, models_dir)
        if adapter is None:
            results.append((version_name, False, "Adapter instantiation failed"))
            continue

        can_run, reason = adapter.detect()
        results.append((version_name, can_run, reason))

    return results


# =============================================================================
# Auto-import adapters to trigger registration
# =============================================================================

def _auto_register_adapters():
    """
    アダプターモジュールを自動インポートしてレジストリに登録する
    """
    # v4 adapter
    try:
        from . import v4_adapter  # noqa: F401
    except ImportError:
        pass

    # legacy adapter
    try:
        from . import legacy_adapter  # noqa: F401
    except ImportError:
        pass


# モジュール読み込み時に自動登録
_auto_register_adapters()
