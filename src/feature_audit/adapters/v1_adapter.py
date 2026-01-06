# -*- coding: utf-8 -*-
"""
v1_adapter.py - Adapter for feature_table (v1, original)

元のfeature_tableに対応するアダプター。
v1は基本的な特徴量のみ（hr_*やax*_は含まない）。
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore

if TYPE_CHECKING:
    import pandas as pd  # noqa: F811

from .base import BaseFeatureAdapter
from .registry import register_adapter

logger = logging.getLogger(__name__)


# 禁止カラム（pre_raceモード）
PROHIBITED_EXACT = {
    "race_id",
    "horse_id",
    "horse_no",
    "target_win",
    "target_in3",
    "target_value",
    "finish_order",
    "finish_position",
    "paid_places",
    "payout_count",
    "should_have_payout",
    "fukusho_payout",
    "track_condition",
    "track_condition_id",
    "horse_weight",
    "horse_weight_diff",
}

PROHIBITED_PATTERNS = (
    "target",
    "payout",
    "paid_",
    "should_have",
)


def _is_prohibited(col: str) -> bool:
    """カラムが禁止リストに該当するか"""
    if col in PROHIBITED_EXACT:
        return True
    col_lower = col.lower()
    for pattern in PROHIBITED_PATTERNS:
        if pattern in col_lower:
            return True
    return False


@register_adapter("v1")
class V1Adapter(BaseFeatureAdapter):
    """
    feature_table (v1) 用アダプター

    v1は基本的な特徴量のみを含むオリジナルのテーブル。
    """

    VERSION_NAME = "v1"
    DESCRIPTION = "feature_table (original, basic features)"
    TABLE_NAME = "feature_table"

    def __init__(
        self,
        db_path: Optional[Path] = None,
        models_dir: Optional[Path] = None,
    ):
        super().__init__(db_path, models_dir)

    def detect(self) -> Tuple[bool, str]:
        """
        v1が実行可能かを検出
        """
        if not self.db_path or not self.db_path.exists():
            return False, "Database path not specified or does not exist"

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (self.TABLE_NAME,)
            )
            result = cursor.fetchone()
            conn.close()

            if result is None:
                return False, f"{self.TABLE_NAME} table not found"

            return True, f"OK (using {self.TABLE_NAME})"

        except Exception as e:
            return False, f"Database check failed: {e}"

    def list_feature_columns(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
    ) -> List[str]:
        """
        v1 table の全カラムから特徴量を抽出
        """
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({self.TABLE_NAME})")
        columns = [row[1] for row in cursor.fetchall()]

        # 禁止カラムを除外
        feature_cols = [c for c in columns if not _is_prohibited(c)]

        return feature_cols

    def get_used_features(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
    ) -> List[str]:
        """
        実際に使用される特徴量リストを取得

        Note: version無しファイルへのfallbackは禁止。
        必ずversion付きファイル (feature_columns_{target}_v1.json) を参照する。
        """
        # 1. version付きfeature_columns JSONファイルを探す（fallback禁止）
        if self.models_dir:
            json_path = self.models_dir / f"feature_columns_{target}_v1.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    features = json.load(f)
                logger.info(f"Loaded feature columns from: {json_path}")
                return features
            else:
                # version無しファイルがあっても使わない
                fallback_path = self.models_dir / f"feature_columns_{target}.json"
                if fallback_path.exists():
                    logger.warning(
                        f"[v1] Found version-less file {fallback_path.name} but NOT using it. "
                        f"Please run train_eval_legacy.py --version v1 to generate v1-specific files."
                    )

        # 2. テーブルから特徴量を取得（モデルファイルがない場合のみ）
        logger.warning(f"[v1] No feature_columns_{target}_v1.json found, using table columns")
        return self.list_feature_columns(conn, mode, target)

    def get_feature_matrix_sample(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
        sample_size: int = 1000,
    ) -> "pd.DataFrame":
        """
        v1 table からサンプルを取得
        """
        query = f"""
            SELECT *
            FROM {self.TABLE_NAME}
            ORDER BY RANDOM()
            LIMIT {sample_size}
        """
        df = pd.read_sql_query(query, conn)
        return df

    def get_importance(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """
        モデルの特徴量重要度を取得
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise NotImplementedError("lightgbm not installed")

        gain_dict: Dict[str, float] = {}
        split_dict: Dict[str, int] = {}

        # 1. importance CSVを探す
        if self.models_dir:
            csv_paths = [
                self.models_dir / f"feature_importance_{target}_v1.csv",
            ]
            for csv_path in csv_paths:
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    if "feature" in df.columns:
                        if "gain" in df.columns:
                            gain_dict = dict(zip(df["feature"], df["gain"]))
                        if "split" in df.columns:
                            split_dict = dict(zip(df["feature"], df["split"].astype(int)))
                        if "importance" in df.columns and not gain_dict:
                            gain_dict = dict(zip(df["feature"], df["importance"]))
                        logger.info(f"Loaded importance from: {csv_path}")
                        return gain_dict, split_dict

        # 2. モデルファイルから直接読み込む
        if self.models_dir:
            model_paths = [
                self.models_dir / f"lgbm_{target}_v1.txt",
            ]
            for model_path in model_paths:
                if model_path.exists():
                    try:
                        try:
                            model = lgb.Booster(model_file=str(model_path))
                        except Exception:
                            model_str = model_path.read_text(encoding="utf-8")
                            model = lgb.Booster(model_str=model_str)

                        feature_names = model.feature_name()
                        gain_values = model.feature_importance(importance_type="gain")
                        split_values = model.feature_importance(importance_type="split")

                        gain_dict = dict(zip(feature_names, gain_values.tolist()))
                        split_dict = dict(zip(feature_names, split_values.astype(int).tolist()))

                        logger.info(f"Loaded importance from model: {model_path}")
                        return gain_dict, split_dict

                    except Exception as e:
                        logger.warning(f"Failed to load model {model_path}: {e}")
                        continue

        raise NotImplementedError(
            f"No model file or importance CSV found for {self.VERSION_NAME}"
        )
