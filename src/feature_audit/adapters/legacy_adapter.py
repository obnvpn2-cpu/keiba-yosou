# -*- coding: utf-8 -*-
"""
legacy_adapter.py - Adapter for Legacy Feature Tables (v3/v2/v1)

feature_table_v3, feature_table_v2, feature_table (v1) に対応するアダプター。
train_eval_lgbm.py のロジックを参考に実装。
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


# 禁止カラム（train_eval_lgbm.py から）
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


@register_adapter("legacy")
class LegacyAdapter(BaseFeatureAdapter):
    """
    Legacy Feature Tables (v3/v2/v1) 用アダプター

    優先順位: feature_table_v3 > feature_table_v2 > feature_table
    """

    VERSION_NAME = "legacy"
    DESCRIPTION = "Legacy feature tables (v3/v2/v1)"

    def __init__(
        self,
        db_path: Optional[Path] = None,
        models_dir: Optional[Path] = None,
    ):
        super().__init__(db_path, models_dir)
        self._detected_table: Optional[str] = None

    def _detect_table(self, conn: sqlite3.Connection) -> Optional[str]:
        """利用可能なfeature tableを検出"""
        if self._detected_table:
            return self._detected_table

        cursor = conn.cursor()

        # 優先順位で検索
        for table_name in ["feature_table_v3", "feature_table_v2", "feature_table"]:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if cursor.fetchone():
                self._detected_table = table_name
                return table_name

        return None

    def detect(self) -> Tuple[bool, str]:
        """
        legacyが実行可能かを検出
        """
        if not self.db_path or not self.db_path.exists():
            return False, "Database path not specified or does not exist"

        try:
            conn = sqlite3.connect(str(self.db_path))
            table = self._detect_table(conn)
            conn.close()

            if table is None:
                return False, "No legacy feature table found (v3/v2/v1)"

            return True, f"OK (using {table})"

        except Exception as e:
            return False, f"Database check failed: {e}"

    def list_feature_columns(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
    ) -> List[str]:
        """
        legacy table の全カラムから特徴量を抽出
        """
        table = self._detect_table(conn)
        if table is None:
            raise NotImplementedError("No legacy feature table found")

        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
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
        """
        # 1. feature_columns JSONファイルを探す
        if self.models_dir:
            json_paths = [
                self.models_dir / f"feature_columns_{target}.json",
            ]
            for json_path in json_paths:
                if json_path.exists():
                    with open(json_path, "r") as f:
                        features = json.load(f)
                    logger.info(f"Loaded feature columns from: {json_path}")
                    return features

        # 2. テーブルから特徴量を取得
        return self.list_feature_columns(conn, mode, target)

    def get_feature_matrix_sample(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
        sample_size: int = 1000,
    ) -> "pd.DataFrame":
        """
        legacy table からサンプルを取得
        """
        table = self._detect_table(conn)
        if table is None:
            raise NotImplementedError("No legacy feature table found")

        query = f"""
            SELECT *
            FROM {table}
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

        # モデルファイルを探す
        if self.models_dir:
            model_paths = [
                self.models_dir / f"lgbm_{target}.txt",
                self.models_dir / f"lgbm_{target}.pkl",
            ]
            for model_path in model_paths:
                if model_path.exists():
                    try:
                        # .txt の場合
                        if model_path.suffix == ".txt":
                            try:
                                model = lgb.Booster(model_file=str(model_path))
                            except Exception:
                                model_str = model_path.read_text(encoding="utf-8")
                                model = lgb.Booster(model_str=model_str)
                        else:
                            # .pkl の場合
                            import joblib
                            model = joblib.load(model_path)

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
            "No model file found for legacy"
        )
