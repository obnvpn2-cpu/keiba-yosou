"""
RaceFeatureBuilder v5.1 - 本番対応完成版

v5.1（2024-12-07）: 本番運用強化
🔥 改善点:
1. merge 後の行数チェック（データ整合性保証）
2. horse_id の存在・欠損チェック（堅牢性向上）
3. 詳細なログ出力（トラブルシュート改善）
4. パフォーマンス監視（処理時間計測）
5. 防御的コーディングの徹底

v5.0（2024-12-04）: PaceAdjustment v2.0対応 + index修正
🔥 実装済み:
1. build_for_race()がhorse_featuresも返すように変更
2. 返り値を辞書に変更（race_features + horse_features）
3. index/column 衝突問題の修正
4. 既存APIとの互換性維持（get_race_features()追加）

v4.0: 致命的問題完全修正
v3.0: HorseHistoryStore v2.0整合
v2.0: カラム名修正
v1.0: 初版
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
import warnings
import logging
import time

import numpy as np
import pandas as pd

# ロギング設定
logger = logging.getLogger(__name__)

# HorseHistoryStore 側と脚質表記を完全統一
RUNNING_STYLE_NIGE = "逃げ"
RUNNING_STYLE_SENKOU = "先行"
RUNNING_STYLE_SASHI = "差し"
RUNNING_STYLE_OIKOMI = "追込"


@dataclass
class RaceFeatureBuilderConfig:
    """
    RaceFeatureBuilder のハイパーパラメータ
    """
    # 過去走参照数
    lookback_races: int = 5

    # 騎手スタッツの最小サンプル数
    min_jockey_races: int = 5

    # ベイズ補正の prior
    global_late_start_rate: float = 0.10
    global_front_runner_rate: float = 0.18

    late_start_beta: float = 20.0
    front_runner_beta: float = 40.0
    
    # パフォーマンス監視（秒）
    slow_race_threshold: float = 1.0


class RaceFeatureBuilder:
    """
    HorseHistoryStore + race_df/entries_df を統合して、
    ペース予測モデルへ入力するレース単位特徴量を生成する（v5.1）
    
    🔥 v5.1: 本番運用強化
    - 防御的チェックの徹底（行数チェック、欠損チェック）
    - 詳細なログ出力（INFO/WARNING/DEBUG）
    - パフォーマンス監視（処理時間計測）
    
    🔥 v5.0: PaceAdjustment対応 + index修正
    - build_for_race()が辞書を返す
    - race_features（レース単位）とhorse_features（馬ごと）の両方を返す
    - index/column 衝突問題の修正
    
    🔧 Index 設計:
    - 内部的に生成する全ての DataFrame は RangeIndex を使用
    - キーは常に column として持つ（index には持たない）
    - merge/join は常に column ベースで実行
    
    必要なデータ形式:
    
    entries_df (出走馬一覧):
        - horse_id: 馬ID (必須)
        - jockey_id: 騎手ID (推奨)
        - jockey_name: 騎手名 (jockey_idがない場合)
        - ⚠️ index: 任意（内部で reset_index される）
    
    race_row (レース情報):
        - race_id: レースID (必須)
        - track_type: "芝" or "ダート" (デフォルト: "芝")
        - distance: 距離メートル (デフォルト: 1600)
        - field_size: 出走頭数 (デフォルト: entries_dfの行数)
        - track_condition: 馬場状態
        - course: 競馬場名
    """

    def __init__(self, history_store, config: Optional[RaceFeatureBuilderConfig] = None):
        self.history_store = history_store
        self.config = config or RaceFeatureBuilderConfig()

    # ================================================================
    # Public API（v5.1改善版）
    # ================================================================
    def build_for_race(
        self,
        race_row: pd.Series,
        entries_df: pd.DataFrame,
        as_of: datetime,
    ) -> Dict[str, Any]:
        """
        単一レースについて、レース単位特徴量と馬ごと特徴量を構築（v5.1改善版）
        
        🔥 v5.1: 防御的チェック + ログ強化
        🔥 v5.0: 返り値を辞書に変更
        
        Args:
            race_row: レース情報
            entries_df: 出走馬一覧
            as_of: 予測時刻（未来データ禁止）
        
        Returns:
            {
                "race_features": Dict[str, Any],  # ペース予測モデル用
                "horse_features": Dict[str, Dict[str, Any]]  # PaceAdjustment用
            }
            
            horse_features形式:
            {
                "horse_1": {"running_style": "逃げ", ...},
                "horse_2": {"running_style": "差し", ...},
            }
        """
        start_time = time.time()
        
        race_id = race_row.get("race_id")
        if race_id is None:
            raise ValueError("race_row に race_id が必要です。")
        
        logger.info(f"レース {race_id} の特徴量構築を開始")
        logger.debug(f"  出走頭数: {len(entries_df)}")

        if entries_df.empty:
            logger.warning(f"レース {race_id} の entries_df が空です。")
            return {
                "race_features": {},
                "horse_features": {}
            }

        # 馬レベル特徴量を構築
        horse_features_df = self._build_horse_level_features(entries_df, as_of)

        # レース単位特徴量に集約
        race_features = self._aggregate_race_level_features(
            race_row=race_row,
            entries_df=entries_df,
            horse_features=horse_features_df,
        )
        
        # 🔥 v5.0: 馬ごと特徴量を辞書に変換
        horse_features_dict = self._convert_horse_features_to_dict(horse_features_df)
        
        # パフォーマンス監視
        elapsed = time.time() - start_time
        
        if elapsed > self.config.slow_race_threshold:
            logger.warning(
                f"レース {race_id} の特徴量構築に {elapsed:.2f} 秒かかりました。"
            )
        
        logger.info(f"レース {race_id} の特徴量構築完了（{elapsed:.3f} 秒）")
        logger.debug(f"  race_features: {len(race_features)} 項目")
        logger.debug(f"  horse_features: {len(horse_features_dict)} 頭")

        return {
            "race_features": race_features,
            "horse_features": horse_features_dict
        }
    
    def get_race_features(
        self,
        race_row: pd.Series,
        entries_df: pd.DataFrame,
        as_of: datetime,
    ) -> Dict[str, Any]:
        """
        レース単位特徴量のみを取得（後方互換性のため）
        
        🔥 v5.0: build_for_race()["race_features"]のショートカット
        """
        result = self.build_for_race(race_row, entries_df, as_of)
        return result["race_features"]

    # ================================================================
    # 馬レベル特徴量
    # ================================================================
    def _build_horse_level_features(
        self, entries_df: pd.DataFrame, as_of: datetime
    ) -> pd.DataFrame:
        """
        各馬ごとに過去走から脚質・速度・騎手スタッツを計算（v5.1改善版）
        
        🔥 v5.1: 防御的チェック強化
        """
        # 🔥 v5.1: horse_id の存在チェック
        if "horse_id" not in entries_df.columns:
            logger.error("entries_df に horse_id カラムがありません。")
            raise ValueError("entries_df に horse_id カラムがありません。")
        
        # 🔥 v5.1: horse_id の欠損チェック
        n_missing = entries_df["horse_id"].isna().sum()
        if n_missing > 0:
            logger.warning(
                f"entries_df に horse_id が欠損している行が {n_missing} 件あります。"
            )
        
        rows: List[Dict[str, Any]] = []
        skipped_horses = 0

        for idx, row in entries_df.iterrows():
            horse_id = row.get("horse_id")
            
            # 🔥 v5.1: 欠損時のログ追加
            if horse_id is None or pd.isna(horse_id):
                logger.debug(f"entries_df の行 {idx} に horse_id がありません。スキップします。")
                skipped_horses += 1
                continue
            
            # 🔥 v4.0: jockey_idまたはjockey_nameを取得
            jockey = row.get("jockey_id") or row.get("jockey_name")

            # HorseHistoryStore から安全な過去走を取得
            history = self.history_store.get_history(horse_id, as_of)
            if not isinstance(history, pd.DataFrame):
                logger.debug(f"馬 {horse_id} の過去走データがありません。")
                skipped_horses += 1
                continue

            # 直近の過去走だけ抽出
            history_recent = history.tail(self.config.lookback_races)

            # --- 4つの主要 horse-level features ---
            running_style = self._estimate_running_style(history_recent)
            early_speed = self._estimate_early_speed(history_recent)
            late_speed = self._estimate_late_speed(history_recent)
            late_start_rate = self._estimate_late_start_rate(history_recent)
            jockey_aggr = self._estimate_jockey_aggressiveness(history, jockey)

            rows.append(
                {
                    "horse_id": horse_id,
                    "jockey": jockey,
                    "running_style": running_style,
                    "early_speed_index": early_speed,
                    "late_speed_index": late_speed,
                    "late_start_rate": late_start_rate,
                    "jockey_aggressiveness": jockey_aggr,
                }
            )
        
        # 🔥 v5.1: スキップされた馬の報告
        if skipped_horses > 0:
            logger.info(f"  {skipped_horses} 頭をスキップしました（horse_id欠損または過去走なし）")

        # ❗ v5.0修正: index は素直に RangeIndex のままにしておく
        if not rows:
            logger.warning("全ての馬で特徴量構築に失敗しました。空のDataFrameを返します。")
            return self._get_empty_horse_features_df()

        df = pd.DataFrame(rows)

        # 念のため整形しておく
        if "horse_id" not in df.columns:
            raise ValueError("horse_id column が horse-level features に存在しません。")

        df = df.reset_index(drop=True)
        
        logger.debug(f"  horse_features: {len(df)} 頭分の特徴量を構築")
        
        return df
    
    def _get_empty_horse_features_df(self) -> pd.DataFrame:
        """空の horse_features DataFrame を返す（v5.1新機能）"""
        return pd.DataFrame(
            columns=[
                "horse_id",
                "jockey",
                "running_style",
                "early_speed_index",
                "late_speed_index",
                "late_start_rate",
                "jockey_aggressiveness",
            ]
        )
    
    def _convert_horse_features_to_dict(
        self,
        horse_features_df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        馬レベル特徴量DataFrameを辞書に変換（v5.0新機能）
        
        🔥 v5.0: PaceAdjustment用の形式に変換
        
        Args:
            horse_features_df: _build_horse_level_features()の出力
        
        Returns:
            {
                "horse_1": {"running_style": "逃げ", ...},
                "horse_2": {"running_style": "差し", ...},
            }
        """
        result = {}
        
        for _, row in horse_features_df.iterrows():
            horse_id = row.get("horse_id")
            if horse_id is None:
                continue
            
            result[horse_id] = {
                "running_style": row.get("running_style"),
                "early_speed_index": row.get("early_speed_index"),
                "late_speed_index": row.get("late_speed_index"),
                "late_start_rate": row.get("late_start_rate"),
                "jockey_aggressiveness": row.get("jockey_aggressiveness"),
            }
        
        return result

    # ================================================================
    # 脚質推定（HorseHistoryStore の classify_running_style を使用）
    # ================================================================
    def _estimate_running_style(self, history_recent: pd.DataFrame) -> str:
        """
        直近の過去走から脚質を推定
        
        🔥 HorseHistoryStoreの高精度分類を利用
        """
        if history_recent is None or history_recent.empty:
            return RUNNING_STYLE_SASHI

        styles = []
        for _, r in history_recent.iterrows():
            style = self.history_store._classify_running_style(
                corner1_pos=r.get("corner1_pos"),
                field_size=r.get("field_size"),
                course_type=r.get("course_type", "芝"),
                distance=r.get("distance", 1600),
            )
            styles.append(style)

        if not styles:
            return RUNNING_STYLE_SASHI

        # 最頻値
        from collections import Counter
        counter = Counter(styles)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else RUNNING_STYLE_SASHI

    # ================================================================
    # スピード系指標
    # ================================================================
    def _estimate_early_speed(self, history_recent: pd.DataFrame) -> float:
        """
        序盤スピード指標（v4.0修正版）
        
        🔥 v4.0: corner1_pos → corner1_posに修正
        """
        if history_recent is None or history_recent.empty:
            return 0.5

        if "corner1_pos" not in history_recent.columns:
            return 0.5
        if "field_size" not in history_recent.columns:
            return 0.5

        df = history_recent.copy()
        df = df.dropna(subset=["corner1_pos", "field_size"])

        if df.empty:
            return 0.5

        # 0-1正規化（1位=1.0, 最下位=0.0）
        normalized = 1.0 - (df["corner1_pos"] - 1) / (df["field_size"] - 1)
        normalized = normalized.clip(0.0, 1.0)

        return float(normalized.mean())

    def _estimate_late_speed(self, history_recent: pd.DataFrame) -> float:
        """
        終盤スピード指標（v4.0修正版）
        
        🔥 v4.0: final_3f_time対応
        """
        if history_recent is None or history_recent.empty:
            return 0.5

        if "final_3f_time" not in history_recent.columns:
            return 0.5

        df = history_recent.copy()
        df = df.dropna(subset=["final_3f_time"])

        if df.empty:
            return 0.5

        # タイムが速いほど高スコア
        max_time = df["final_3f_time"].max()
        min_time = df["final_3f_time"].min()

        if max_time == min_time:
            return 0.5

        normalized = 1.0 - (df["final_3f_time"] - min_time) / (max_time - min_time)
        return float(np.clip(normalized.mean(), 0.0, 1.0))

    # ================================================================
    # 出遅れ率（HorseHistoryStore の _has_late_start を使用）
    # ================================================================
    def _estimate_late_start_rate(self, history_recent: pd.DataFrame) -> float:
        """
        出遅れ率推定
        
        🔥 HorseHistoryStoreの高精度判定を利用
        """
        if history_recent is None or history_recent.empty:
            return self.config.global_late_start_rate

        if "remarks" not in history_recent.columns:
            return self.config.global_late_start_rate

        # HorseHistoryStore の高精度判定を利用
        flags = self.history_store._has_late_start(history_recent["remarks"])

        n = len(flags)
        k = float(flags.sum())

        alpha = self.config.late_start_beta
        p0 = self.config.global_late_start_rate

        return float((k + alpha * p0) / (n + alpha))

    # ================================================================
    # 騎手 aggressiveness（v4.0修正：カラム名対応）
    # ================================================================
    def _estimate_jockey_aggressiveness(
        self, 
        full_history: pd.DataFrame, 
        jockey: str
    ) -> float:
        """
        騎手の攻撃性を推定（v4.0修正版）
        
        🔥 v4.0: jockey_id/jockey_name対応 + running_style計算
        """
        if jockey is None or pd.isna(jockey):
            return 0.5

        if full_history.empty:
            return 0.5

        # 🔥 v4.0: jockey_idとjockey_nameの両方で検索
        if "jockey_id" in full_history.columns:
            df = full_history[full_history["jockey_id"] == jockey]
        elif "jockey_name" in full_history.columns:
            df = full_history[full_history["jockey_name"] == jockey]
        else:
            logger.debug("full_historyにjockey_idまたはjockey_nameカラムがありません")
            return 0.5

        if len(df) < self.config.min_jockey_races:
            logger.debug(
                f"騎手 {jockey} のサンプル数が少ない（{len(df)}件）。"
                "全体平均に寄せます。"
            )
            return self.config.global_front_runner_rate

        # 🔥 v4.0: 各レースの脚質を計算
        styles = []
        for _, r in df.iterrows():
            style = self.history_store._classify_running_style(
                corner1_pos=r.get("corner1_pos"),
                field_size=r.get("field_size"),
                course_type=r.get("course_type", "芝"),
                distance=r.get("distance", 1600),
            )
            styles.append(style)

        if not styles:
            return self.config.global_front_runner_rate

        n = len(styles)
        k = sum(1 for s in styles if s in [RUNNING_STYLE_NIGE, RUNNING_STYLE_SENKOU])

        alpha = self.config.front_runner_beta
        p0 = self.config.global_front_runner_rate

        return float((k + alpha * p0) / (n + alpha))

    # ================================================================
    # レース単位集約
    # ================================================================
    def _aggregate_race_level_features(
        self,
        race_row: pd.Series,
        entries_df: pd.DataFrame,
        horse_features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        馬レベル特徴量をレース単位に集約（v5.1改善版）
        
        🔥 v5.1: 防御的チェック強化
        """
        # ========================================
        # 🔥 v5.1: 防御的チェック
        # ========================================
        
        # 1. 入力の妥当性チェック
        if entries_df.empty:
            logger.warning("entries_df が空です。デフォルト値を返します。")
            return self._get_default_race_features(race_row)
        
        if "horse_id" not in entries_df.columns:
            raise ValueError("entries_df に horse_id カラムがありません。")
        
        # 2. horse_features の妥当性チェック
        if not horse_features.empty:
            if "horse_id" not in horse_features.columns:
                raise ValueError("horse_features に horse_id カラムがありません。")
            
            # 重複チェック
            n_unique = horse_features["horse_id"].nunique()
            n_total = len(horse_features)
            
            if n_unique != n_total:
                logger.warning(
                    f"horse_features に重複した horse_id があります。"
                    f"（ユニーク数: {n_unique}, 総数: {n_total}）"
                )
        
        # ========================================
        # merge 処理
        # ========================================
        
        # ❗ v5.0修正: 念のため、index 名に依存しないよう両方ともリセット
        entries_df = entries_df.reset_index(drop=True)
        horse_features = horse_features.reset_index(drop=True)
        
        # 🔥 v5.1: 行数記録
        n_entries_before = len(entries_df)

        df = entries_df.merge(
            horse_features,
            on="horse_id",
            how="left",
        )
        
        # 🔥 v5.1: 行数チェック
        n_entries_after = len(df)
        
        if n_entries_after != n_entries_before:
            logger.warning(
                f"merge 後の行数が変化しました。"
                f"（前: {n_entries_before}, 後: {n_entries_after}）"
            )
        
        # ========================================
        # 欠損チェック + 埋め処理
        # ========================================
        
        # 🔥 v5.1: 欠損チェック
        for col in ["running_style", "early_speed_index", "late_speed_index", 
                    "late_start_rate", "jockey_aggressiveness"]:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                logger.debug(
                    f"merge 後の {col} に {n_missing} 件の欠損があります。"
                    "デフォルト値で埋めます。"
                )

        # 欠損埋め
        df["running_style"] = df["running_style"].fillna(RUNNING_STYLE_SASHI)
        df["early_speed_index"] = df["early_speed_index"].fillna(0.5)
        df["late_speed_index"] = df["late_speed_index"].fillna(0.5)
        df["late_start_rate"] = df["late_start_rate"].fillna(self.config.global_late_start_rate)
        df["jockey_aggressiveness"] = df["jockey_aggressiveness"].fillna(0.5)

        # ========================================
        # 集約処理
        # ========================================
        
        # 脚質分布
        rs = df["running_style"]
        num_nige = int((rs == RUNNING_STYLE_NIGE).sum())
        num_senkou = int((rs == RUNNING_STYLE_SENKOU).sum())
        num_sashi = int((rs == RUNNING_STYLE_SASHI).sum())
        num_oikomi = int((rs == RUNNING_STYLE_OIKOMI).sum())

        # 逃げ馬の速度
        nige_df = df[rs == RUNNING_STYLE_NIGE]
        if not nige_df.empty:
            nige_speed_max = float(nige_df["early_speed_index"].max())
            nige_speed_mean = float(nige_df["early_speed_index"].mean())
            nige_speed_std = float(nige_df["early_speed_index"].std(ddof=0))
        else:
            nige_speed_max = 0.0
            nige_speed_mean = 0.0
            nige_speed_std = 0.0

        # 先行圧力
        senkou_df = df[rs == RUNNING_STYLE_SENKOU]
        senkou_pressure = float(senkou_df["early_speed_index"].mean()) if not senkou_df.empty else 0.0

        # 逃げ馬の競り合いリスク
        escape_competition_risk = self._compute_escape_competition_risk(nige_df)

        # 差し・追込末脚
        sashi_df = df[rs == RUNNING_STYLE_SASHI]
        oikomi_df = df[rs == RUNNING_STYLE_OIKOMI]

        sashi_late_speed_mean = float(sashi_df["late_speed_index"].mean()) if not sashi_df.empty else 0.0
        oikomi_late_speed_mean = float(oikomi_df["late_speed_index"].mean()) if not oikomi_df.empty else 0.0

        # 騎手要因
        mean_jockey_aggr = float(df["jockey_aggressiveness"].mean())
        aggressive_jockey_count = int((df["jockey_aggressiveness"] >= 0.6).sum())
        mean_late_start_rate = float(df["late_start_rate"].mean())

        # レース条件
        track_type = race_row.get("track_type", "芝")
        distance = int(race_row.get("distance", 1600))
        field_size = int(race_row.get("field_size", len(entries_df)))

        track_condition = race_row.get("track_condition")
        course = race_row.get("course")
        turn_type = race_row.get("turn_type")
        track_bias = race_row.get("track_bias", 0.0)

        return {
            # 脚質分布
            "field_size": field_size,
            "num_nige": num_nige,
            "num_senkou": num_senkou,
            "num_sashi": num_sashi,
            "num_oikomi": num_oikomi,

            # 逃げ・先行
            "nige_speed_max": nige_speed_max,
            "nige_speed_mean": nige_speed_mean,
            "nige_speed_std": nige_speed_std,
            "senkou_pressure": senkou_pressure,
            "escape_competition_risk": escape_competition_risk,

            # 差し・追込
            "sashi_late_speed_mean": sashi_late_speed_mean,
            "oikomi_late_speed_mean": oikomi_late_speed_mean,

            # 騎手
            "mean_jockey_aggressiveness": mean_jockey_aggr,
            "aggressive_jockey_count": aggressive_jockey_count,
            "mean_late_start_rate": mean_late_start_rate,

            # レース条件
            "track_type": track_type,
            "distance": distance,
            "track_condition": track_condition,
            "course": course,
            "turn_type": turn_type,
            "track_bias": track_bias,
        }
    
    def _get_default_race_features(self, race_row: pd.Series) -> Dict[str, Any]:
        """
        entries_df が空の場合のデフォルト値（v5.1新機能）
        """
        return {
            "field_size": 0,
            "num_nige": 0,
            "num_senkou": 0,
            "num_sashi": 0,
            "num_oikomi": 0,
            "nige_speed_max": 0.0,
            "nige_speed_mean": 0.0,
            "nige_speed_std": 0.0,
            "senkou_pressure": 0.0,
            "escape_competition_risk": 0.0,
            "sashi_late_speed_mean": 0.0,
            "oikomi_late_speed_mean": 0.0,
            "mean_jockey_aggressiveness": 0.5,
            "aggressive_jockey_count": 0,
            "mean_late_start_rate": self.config.global_late_start_rate,
            "track_type": race_row.get("track_type", "芝"),
            "distance": int(race_row.get("distance", 1600)),
            "track_condition": race_row.get("track_condition"),
            "course": race_row.get("course"),
            "turn_type": race_row.get("turn_type"),
            "track_bias": race_row.get("track_bias", 0.0),
        }

    # ================================================================
    # 逃げ馬同士の競り合いリスク（v4.0改善）
    # ================================================================
    def _compute_escape_competition_risk(self, nige_df: pd.DataFrame) -> float:
        """
        逃げ馬同士の競り合いリスク（v4.0改善版）
        
        🔥 v4.0: 閾値改善
        
        Returns:
            0.0: 単独逃げ（リスクなし）
            0.5-0.9: 2頭の逃げ（速度差に応じて）
            1.0: 3頭以上の逃げ（確実に競り合う）
        """
        n = len(nige_df)
        
        if n == 0:
            return 0.0
        if n == 1:
            return 0.0
        if n >= 3:
            return 1.0

        # 2頭の場合は速度差で判定
        speeds = nige_df["early_speed_index"].values
        if len(speeds) != 2:
            return 0.5

        diff = abs(speeds[0] - speeds[1])

        # 🔥 v4.0: 閾値改善
        # 差が0.15以下なら高リスク（能力が近い）
        # 差が0.3以上なら低リスク（能力差が明確）
        if diff <= 0.15:
            risk = 0.9
        elif diff >= 0.3:
            risk = 0.5
        else:
            # 0.15-0.3の間は線形補間
            risk = 0.9 - (diff - 0.15) / 0.15 * 0.4

        return float(np.clip(risk, 0.0, 1.0))


def example_usage():
    """使用例（v5.1）"""
    
    print("=" * 80)
    print("RaceFeatureBuilder v5.1 - 使用例（本番対応版）")
    print("=" * 80)
    
    print("\n✅ v5.1完成 - 本番運用強化")
    print("  - 防御的チェックの徹底（行数チェック、欠損チェック）")
    print("  - 詳細なログ出力（INFO/WARNING/DEBUG）")
    print("  - パフォーマンス監視（処理時間計測）")
    
    print("\n✅ v5.0完成 - PaceAdjustment v2.0対応")
    print("  - build_for_race()が辞書を返す")
    print("  - race_features（レース単位）とhorse_features（馬ごと）の両方を返す")
    print("  - 後方互換性のためget_race_features()も提供")


if __name__ == "__main__":
    example_usage()
