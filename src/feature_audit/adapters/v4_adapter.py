# -*- coding: utf-8 -*-
"""
v4_adapter.py - Adapter for FeaturePack v4

src/features_v4/ に対応するアダプター。
既存の train_eval_v4.py のロジックを流用して特徴量情報を取得する。
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


# pre_race モードで除外する特徴量
PRE_RACE_EXCLUDE_FEATURES = {
    "h_body_weight",
    "h_body_weight_diff",
    "h_body_weight_dev",
    "market_win_odds",
    "market_popularity",
}


@register_adapter("v4")
class V4Adapter(BaseFeatureAdapter):
    """
    FeaturePack v4 (src/features_v4/) 用アダプター
    """

    VERSION_NAME = "v4"
    DESCRIPTION = "FeaturePack v4 (200+ features, LightGBM)"

    def __init__(
        self,
        db_path: Optional[Path] = None,
        models_dir: Optional[Path] = None,
    ):
        super().__init__(db_path, models_dir)
        self._features_v4_available = None
        self._lightgbm_available = None

    def _check_features_v4(self) -> bool:
        """features_v4モジュールが利用可能か確認"""
        if self._features_v4_available is None:
            try:
                from src.features_v4 import get_feature_v4_columns
                self._features_v4_available = True
            except ImportError:
                self._features_v4_available = False
        return self._features_v4_available

    def _check_lightgbm(self) -> bool:
        """LightGBMが利用可能か確認"""
        if self._lightgbm_available is None:
            try:
                import lightgbm  # noqa: F401
                self._lightgbm_available = True
            except ImportError:
                self._lightgbm_available = False
        return self._lightgbm_available

    def detect(self) -> Tuple[bool, str]:
        """
        v4が実行可能かを検出

        Returns:
            (can_run, reason)
        """
        # 1. features_v4 モジュールの確認
        if not self._check_features_v4():
            return False, "src.features_v4 module not available"

        # 2. LightGBM の確認
        if not self._check_lightgbm():
            return False, "lightgbm not installed"

        # 3. feature_table_v4 テーブルの確認（DBがある場合）
        if self.db_path and self.db_path.exists():
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='feature_table_v4'"
                )
                if cursor.fetchone() is None:
                    conn.close()
                    return False, "feature_table_v4 table not found in database"
                conn.close()
            except Exception as e:
                return False, f"Database check failed: {e}"

        return True, "OK"

    def list_feature_columns(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
    ) -> List[str]:
        """
        v4の全特徴量カラム一覧を取得
        """
        from src.features_v4 import get_feature_v4_columns

        all_cols = get_feature_v4_columns()

        # ID/target系を除外
        exclude_cols = {
            "race_id", "horse_id", "race_date",
            "target_win", "target_in3", "target_value",
        }
        feature_cols = [c for c in all_cols if c not in exclude_cols]

        return feature_cols

    def get_used_features(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
    ) -> List[str]:
        """
        実際にモデルで使用される特徴量リストを取得

        1. feature_columns JSONファイルがあればそれを使う
        2. なければ list_feature_columns から除外適用
        """
        # 1. feature_columns JSONファイルを探す
        if self.models_dir:
            json_paths = [
                self.models_dir / f"feature_columns_{target}_v4.json",
                self.models_dir / f"feature_columns_{target}.json",
            ]
            for json_path in json_paths:
                if json_path.exists():
                    with open(json_path, "r") as f:
                        features = json.load(f)
                    logger.info(f"Loaded feature columns from: {json_path}")

                    # mode=pre_race の場合、除外を適用
                    if mode == "pre_race":
                        features = [
                            f for f in features
                            if f not in PRE_RACE_EXCLUDE_FEATURES
                        ]
                    return features

        # 2. JSONがなければ list_feature_columns から構築
        feature_cols = self.list_feature_columns(conn, mode, target)

        # mode=pre_race の場合、除外を適用
        if mode == "pre_race":
            feature_cols = [
                f for f in feature_cols
                if f not in PRE_RACE_EXCLUDE_FEATURES
            ]

        return feature_cols

    def get_feature_matrix_sample(
        self,
        conn: sqlite3.Connection,
        mode: str = "pre_race",
        target: str = "target_win",
        sample_size: int = 1000,
    ) -> "pd.DataFrame":
        """
        feature_table_v4 からサンプルを取得
        """
        query = f"""
            SELECT *
            FROM feature_table_v4
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

        1. 既存モデルファイルからロード
        2. importance CSVがあればそれを使う
        """
        import lightgbm as lgb

        gain_dict: Dict[str, float] = {}
        split_dict: Dict[str, int] = {}

        # 1. importance CSVを探す
        if self.models_dir:
            csv_paths = [
                self.models_dir / f"feature_importance_{target}_v4.csv",
                self.models_dir / f"feature_importance_{target}.csv",
            ]
            for csv_path in csv_paths:
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    if "feature" in df.columns:
                        if "gain" in df.columns:
                            gain_dict = dict(zip(df["feature"], df["gain"]))
                        if "split" in df.columns:
                            split_dict = dict(zip(df["feature"], df["split"].astype(int)))
                        # importance のみの場合
                        if "importance" in df.columns and not gain_dict:
                            gain_dict = dict(zip(df["feature"], df["importance"]))
                        logger.info(f"Loaded importance from: {csv_path}")

                        # mode=pre_race の場合、除外特徴量をフィルタ
                        if mode == "pre_race":
                            gain_dict = {
                                k: v for k, v in gain_dict.items()
                                if k not in PRE_RACE_EXCLUDE_FEATURES
                            }
                            split_dict = {
                                k: v for k, v in split_dict.items()
                                if k not in PRE_RACE_EXCLUDE_FEATURES
                            }
                        return gain_dict, split_dict

        # 2. モデルファイルから直接読み込む
        if self.models_dir:
            model_paths = [
                self.models_dir / f"lgbm_{target}_v4.txt",
                self.models_dir / f"lgbm_{target}.txt",
            ]
            for model_path in model_paths:
                if model_path.exists():
                    try:
                        # model_str フォールバックでロード
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

                        # mode=pre_race の場合、除外特徴量をフィルタ
                        if mode == "pre_race":
                            gain_dict = {
                                k: v for k, v in gain_dict.items()
                                if k not in PRE_RACE_EXCLUDE_FEATURES
                            }
                            split_dict = {
                                k: v for k, v in split_dict.items()
                                if k not in PRE_RACE_EXCLUDE_FEATURES
                            }
                        return gain_dict, split_dict

                    except Exception as e:
                        logger.warning(f"Failed to load model {model_path}: {e}")
                        continue

        raise NotImplementedError(
            "No model file or importance CSV found for v4"
        )
