# -*- coding: utf-8 -*-
"""
base.py - Base Adapter Interface for Feature Audit

全バージョン共通の抽象インターフェースを定義。
各バージョン固有のアダプターはこのクラスを継承して実装する。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import sqlite3

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore

if TYPE_CHECKING:
    import pandas as pd  # noqa: F811


@dataclass
class AdapterResult:
    """
    アダプター実行結果

    Attributes:
        success: 成功したかどうか
        error_message: 失敗時のエラーメッセージ
        used_features: 使用された特徴量リスト
        feature_stats: 特徴量ごとの統計情報
        importance_gain: gain重要度（feature -> value）
        importance_split: split重要度（feature -> value）
        group_importance: グループ別重要度
        warnings: 警告メッセージリスト
    """
    success: bool = True
    error_message: str = ""
    used_features: List[str] = field(default_factory=list)
    feature_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    importance_gain: Dict[str, float] = field(default_factory=dict)
    importance_split: Dict[str, int] = field(default_factory=dict)
    group_importance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "n_features": len(self.used_features),
            "has_importance": bool(self.importance_gain),
            "warnings": self.warnings,
        }


class BaseFeatureAdapter(ABC):
    """
    特徴量パックアダプターの基底クラス

    各バージョン固有のアダプターはこのクラスを継承し、
    必要なメソッドを実装する。実装できないメソッドは
    NotImplementedError を raise してよい（runner側がSKIPに落とす）。
    """

    # アダプター識別子（サブクラスでオーバーライド）
    VERSION_NAME: str = "unknown"
    DESCRIPTION: str = "Base adapter"

    def __init__(
        self,
        db_path: Optional[Path] = None,
        models_dir: Optional[Path] = None,
    ):
        """
        Args:
            db_path: SQLiteデータベースパス
            models_dir: モデルファイルディレクトリ
        """
        self.db_path = db_path
        self.models_dir = models_dir

    @abstractmethod
    def detect(self) -> Tuple[bool, str]:
        """
        この環境で実行可能かを検出する

        Returns:
            (can_run, reason) - can_runがFalseの場合、reasonに理由
        """
        raise NotImplementedError

    @abstractmethod
    def list_feature_columns(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
    ) -> List[str]:
        """
        使用可能な特徴量カラム一覧を取得

        Args:
            conn: DB接続
            mode: 実行モード ("pre_race", "default")
            target: ターゲットカラム ("target_win", "target_in3")

        Returns:
            特徴量名のリスト
        """
        raise NotImplementedError

    @abstractmethod
    def get_used_features(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
    ) -> List[str]:
        """
        実際にモデルで使用される特徴量リストを取得

        Args:
            conn: DB接続
            mode: 実行モード
            target: ターゲットカラム

        Returns:
            特徴量名のリスト
        """
        raise NotImplementedError

    def get_feature_matrix_sample(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
        sample_size: int = 1000,
    ) -> "pd.DataFrame":
        """
        特徴量マトリクスのサンプルを取得（統計計算用）

        Args:
            conn: DB接続
            mode: 実行モード
            target: ターゲットカラム
            sample_size: サンプルサイズ

        Returns:
            サンプルDataFrame

        Raises:
            NotImplementedError: サポートしていない場合
        """
        raise NotImplementedError(
            f"{self.VERSION_NAME} does not support get_feature_matrix_sample"
        )

    def get_importance(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """
        モデルの特徴量重要度を取得

        Args:
            conn: DB接続
            mode: 実行モード
            target: ターゲットカラム

        Returns:
            (gain_dict, split_dict) - 各特徴量の重要度

        Raises:
            NotImplementedError: サポートしていない場合
        """
        raise NotImplementedError(
            f"{self.VERSION_NAME} does not support get_importance"
        )

    def compute_feature_stats(
        self,
        df: "pd.DataFrame",
        feature_cols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        特徴量の統計情報を計算する（共通実装）

        Args:
            df: データフレーム
            feature_cols: 特徴量カラムリスト

        Returns:
            {feature_name: {dtype, missing_rate, n_unique, example_values}, ...}
        """
        stats = {}
        for col in feature_cols:
            if col not in df.columns:
                stats[col] = {
                    "dtype": "unknown",
                    "missing_rate": None,
                    "n_unique": None,
                    "example_values": [],
                    "error": "column not found in data",
                }
                continue

            series = df[col]
            n_total = len(series)
            n_missing = series.isna().sum()
            missing_rate = n_missing / n_total if n_total > 0 else 0.0

            # dtype
            dtype_str = str(series.dtype)

            # n_unique（カテゴリ的なカラムのみ）
            try:
                n_unique = series.nunique()
            except Exception:
                n_unique = None

            # example_values（最大3つ、数値は丸める）
            try:
                non_null = series.dropna()
                if len(non_null) > 0:
                    examples = non_null.head(3).tolist()
                    # 数値は丸める
                    examples = [
                        round(v, 4) if isinstance(v, float) else v
                        for v in examples
                    ]
                else:
                    examples = []
            except Exception:
                examples = []

            stats[col] = {
                "dtype": dtype_str,
                "missing_rate": round(missing_rate, 4),
                "n_unique": n_unique,
                "example_values": examples,
            }

        return stats

    def run_audit(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
        fast: bool = True,
    ) -> AdapterResult:
        """
        棚卸しを実行する（共通フロー）

        Args:
            conn: DB接続
            mode: 実行モード
            target: ターゲットカラム
            fast: 高速モード（重い処理をスキップ）

        Returns:
            AdapterResult
        """
        result = AdapterResult()

        try:
            # 1. 使用特徴量を取得
            result.used_features = self.get_used_features(conn, mode, target)
        except NotImplementedError as e:
            result.success = False
            result.error_message = str(e)
            return result
        except Exception as e:
            result.success = False
            result.error_message = f"Failed to get used features: {e}"
            return result

        # 2. 特徴量統計を取得（fast=Falseの場合のみ）
        if not fast:
            try:
                df_sample = self.get_feature_matrix_sample(
                    conn, mode, target, sample_size=5000
                )
                result.feature_stats = self.compute_feature_stats(
                    df_sample, result.used_features
                )
            except NotImplementedError:
                result.warnings.append("Feature statistics not available (not implemented)")
            except Exception as e:
                result.warnings.append(f"Failed to compute feature stats: {e}")

        # 3. 重要度を取得
        try:
            gain, split = self.get_importance(conn, mode, target)
            result.importance_gain = gain
            result.importance_split = split
        except NotImplementedError:
            result.warnings.append("Importance not available (not implemented)")
        except Exception as e:
            result.warnings.append(f"Failed to get importance: {e}")

        return result
