"""
馬場補正モデル（v2.0 - ChatGPT版レビュー対応完全版）

v2.0（2024-12-04）: ChatGPT版レビュー + Claude追加修正
🔥 実運用レベル完成:
1. race_date型変換を明示的に（try-exceptからpd.to_datetimeへ）
2. デフォルト値をconfig化（prior_win_rate）
3. track_statistics異常時に警告追加
4. targetクリップを分位点ベースに変更
5. 全体的なコード品質向上

v1.0: 初版（ChatGPT修正版）
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.special import logit, expit
from typing import Dict, Optional, Tuple
import warnings


class BabaAdjustmentModel:
    """
    馬場補正モデル（log-odds空間での補正）

    アプローチ:
    1. ベース予測との log-odds 差 (delta) を学習
    2. 外れ値を分位点ベースでクリップ🔥
    3. 馬ごとの対象馬場レース数に応じて Shrinkage
    """

    def __init__(
        self,
        alpha: float = 0.5,
        clip_percentile: float = 99.0,
        min_data_for_full_weight: int = 10,
        params: Optional[Dict] = None
    ):
        """
        Args:
            alpha: ラプラススムージング係数
            clip_percentile: targetをこの分位点でクリップ（99.0=99%点）🔥
            min_data_for_full_weight: Shrinkageで完全信頼とするデータ数
            params: LightGBMのパラメータ
        """
        self.alpha = alpha
        self.clip_percentile = clip_percentile
        self.min_data_for_full_weight = min_data_for_full_weight

        if params is None:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 15,
                "max_depth": 5,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "verbose": -1,
            }

        self.params = params
        self.model: Optional[lgb.Booster] = None
        self.feature_names = None
        self.target_lower: Optional[float] = None
        self.target_upper: Optional[float] = None

    def prepare_training_data(
        self,
        calibrated_pred: np.ndarray,
        actual_win: np.ndarray,
        features: pd.DataFrame,
        horse_baba_race_counts: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        学習用のターゲットとサンプルウェイトを準備（v2.0改善版）

        Args:
            calibrated_pred: キャリブレーション済み予測勝率
            actual_win: 実際の勝利（1/0）
            features: 馬場関連の特徴量
            horse_baba_race_counts: 各馬の対象馬場条件でのレース数

        Returns:
            (特徴量, target_clipped, sample_weight)
        """
        # smoothing: smoothed_actual = (win + α) / (1 + 2α)
        smoothed_actual = (actual_win + self.alpha) / (1.0 + 2.0 * self.alpha)

        # logit計算（0,1直撃を避ける）
        calibrated_pred_clipped = np.clip(calibrated_pred, 0.001, 0.999)
        smoothed_actual_clipped = np.clip(smoothed_actual, 0.001, 0.999)

        logit_pred = logit(calibrated_pred_clipped)
        logit_actual = logit(smoothed_actual_clipped)

        # 目的変数 = logit(actual) - logit(pred)
        target = logit_actual - logit_pred

        # 🔥 v2.0: 分位点ベースでクリップ（Claude指摘）
        lower_percentile = (100 - self.clip_percentile) / 2
        upper_percentile = 100 - lower_percentile
        
        self.target_lower = float(np.percentile(target, lower_percentile))
        self.target_upper = float(np.percentile(target, upper_percentile))

        target_clipped = np.clip(target, self.target_lower, self.target_upper)

        # サンプルウェイト（データ量に応じた信頼度）
        if horse_baba_race_counts is not None:
            if len(horse_baba_race_counts) != len(target_clipped):
                raise ValueError(
                    "horse_baba_race_counts の長さが特徴量と一致していません"
                )
            sample_weight = np.minimum(
                1.0,
                horse_baba_race_counts / float(self.min_data_for_full_weight),
            )
        else:
            sample_weight = np.ones(len(target_clipped), dtype=float)

        return features, target_clipped, sample_weight

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
    ):
        """モデルを訓練"""
        self.feature_names = X.columns.tolist()

        train_data = lgb.Dataset(
            X,
            label=y,
            weight=sample_weight,
            free_raw_data=False,
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                weight=sample_weight_val,
                reference=train_data,
                free_raw_data=False,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=500,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
            ],
        )

        print(f"✅ BabaAdjustmentModel 訓練完了 - best_iteration: {self.model.best_iteration}")

    def predict_delta(
        self,
        features: pd.DataFrame,
        horse_baba_race_counts: Optional[np.ndarray] = None,
        apply_shrinkage: bool = True,
    ) -> np.ndarray:
        """馬場補正量（log-odds差）を予測"""
        if self.model is None:
            raise ValueError("モデルが訓練されていません")

        delta = self.model.predict(features, num_iteration=self.model.best_iteration)

        # Shrinkage適用
        if apply_shrinkage and horse_baba_race_counts is not None:
            if len(horse_baba_race_counts) != len(delta):
                raise ValueError(
                    "horse_baba_race_counts の長さが予測値と一致していません"
                )
            weight = np.minimum(
                1.0,
                horse_baba_race_counts / float(self.min_data_for_full_weight),
            )
            delta = weight * delta

        return delta

    def apply_adjustment(
        self,
        calibrated_pred: np.ndarray,
        delta_baba: np.ndarray,
    ) -> np.ndarray:
        """ベース予測に馬場補正を適用して最終勝率に変換"""
        calibrated_pred_clipped = np.clip(calibrated_pred, 0.001, 0.999)
        logit_base = logit(calibrated_pred_clipped)

        logit_final = logit_base + delta_baba
        final_prob = expit(logit_final)

        return final_prob

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: int = 10,
    ) -> pd.DataFrame:
        """特徴量の重要度を取得"""
        if self.model is None:
            raise ValueError("モデルが訓練されていません")

        importance = self.model.feature_importance(importance_type=importance_type)
        df = (
            pd.DataFrame({"feature": self.feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )
        return df


class BabaFeatureExtractor:
    """
    馬場補正用の特徴量を抽出するクラス（v2.0改善版）

    🔥 v2.0での改善:
    - race_date型変換を明示的に
    - デフォルト値をconfig化
    - track_statistics異常時に警告
    """

    def __init__(
        self, 
        track_statistics: Dict, 
        date_column: str = "race_date",
        prior_win_rate: float = 0.1
    ):
        """
        Args:
            track_statistics: 競馬場別の統計
            date_column: レース日付カラム名
            prior_win_rate: データ不足時のデフォルト勝率🔥
        """
        self.track_statistics = track_statistics
        self.date_column = date_column
        self.prior_win_rate = prior_win_rate

    def extract_features(
        self,
        race_data: pd.DataFrame,
        horse_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        馬場補正用の特徴量を抽出（v2.0改善版）

        Args:
            race_data: 予測対象レースの情報
            horse_history: 各馬の過去成績

        Returns:
            特徴量DataFrame
        """
        features = []

        for _, race in race_data.iterrows():
            track = race["track_name"]

            # 競馬場別の標準化特徴量
            normalized_cushion = 0.0
            normalized_moisture = 0.0

            if track in self.track_statistics:
                stats = self.track_statistics[track]

                # 🔥 v2.0: track_statistics異常時に警告
                if (
                    "cushion_value" in race
                    and "avg_cushion" in stats
                    and "std_cushion" in stats
                ):
                    if stats["std_cushion"] in (0, None):
                        warnings.warn(
                            f"競馬場 '{track}' のstd_cushionが0またはNoneです。"
                            "normalized_cushionは0にセットされます。",
                            UserWarning
                        )
                    else:
                        normalized_cushion = (
                            race["cushion_value"] - stats["avg_cushion"]
                        ) / stats["std_cushion"]

                if (
                    "moisture" in race
                    and "avg_moisture" in stats
                    and "std_moisture" in stats
                ):
                    if stats["std_moisture"] in (0, None):
                        warnings.warn(
                            f"競馬場 '{track}' のstd_moistureが0またはNoneです。"
                            "normalized_moistureは0にセットされます。",
                            UserWarning
                        )
                    else:
                        normalized_moisture = (
                            race["moisture"] - stats["avg_moisture"]
                        ) / stats["std_moisture"]

            # 馬ごとの馬場適性的な特徴量
            horse_id = race["horse_id"]

            # 🔥 v2.0: デフォルト値をconfig化（Claude指摘）
            high_speed_win_rate = self.prior_win_rate
            slow_win_rate = self.prior_win_rate
            high_speed_count = 0
            slow_count = 0

            if "horse_id" in horse_history.columns:
                horse_past = horse_history[horse_history["horse_id"] == horse_id]

                # 🔥 v2.0: race_date型変換を明示的に（Claude指摘）
                race_date = None
                if self.date_column in race.index:
                    race_date = race[self.date_column]
                elif self.date_column in race_data.columns:
                    race_date = race[self.date_column]

                if race_date is not None and self.date_column in horse_past.columns:
                    # 明示的に型変換
                    if not pd.api.types.is_datetime64_any_dtype(horse_past[self.date_column]):
                        horse_past = horse_past.copy()
                        horse_past[self.date_column] = pd.to_datetime(
                            horse_past[self.date_column], errors='coerce'
                        )
                    
                    if not pd.api.types.is_datetime64_any_dtype(pd.Series([race_date])):
                        race_date = pd.to_datetime(race_date, errors='coerce')
                    
                    # 時系列リーク防止フィルタ
                    if pd.notna(race_date):
                        horse_past = horse_past[horse_past[self.date_column] < race_date]
                    else:
                        warnings.warn(
                            f"race_dateの変換に失敗しました: {race[self.date_column]}",
                            UserWarning
                        )

                if (
                    len(horse_past) > 0
                    and "baba_index" in horse_past.columns
                    and "finish_position" in horse_past.columns
                ):
                    # 高速馬場での成績
                    high_speed_races = horse_past[horse_past["baba_index"] < -1.5]
                    if len(high_speed_races) > 0:
                        high_speed_win_rate = (
                            high_speed_races["finish_position"] == 1
                        ).mean()
                        high_speed_count = int(len(high_speed_races))

                    # 時計かかる馬場での成績
                    slow_races = horse_past[horse_past["baba_index"] > 1.5]
                    if len(slow_races) > 0:
                        slow_win_rate = (
                            slow_races["finish_position"] == 1
                        ).mean()
                        slow_count = int(len(slow_races))

            features.append(
                {
                    "predicted_baba_index": race.get("predicted_baba_index", 0.0),
                    "normalized_cushion": normalized_cushion,
                    "normalized_moisture": normalized_moisture,
                    "horse_high_speed_win_rate": high_speed_win_rate,
                    "horse_slow_win_rate": slow_win_rate,
                    "horse_high_speed_count": high_speed_count,
                    "horse_slow_count": slow_count,
                    "track_correlation": self.track_statistics.get(track, {}).get(
                        "cushion_correlation", 0.0
                    ),
                }
            )

        return pd.DataFrame(features)


def example_usage():
    """使用例（v2.0）"""

    print("=" * 80)
    print("BabaAdjustmentModel v2.0 - 使用例（実運用完成版）")
    print("=" * 80)

    np.random.seed(42)
    n = 1000

    calibrated_pred = np.random.beta(2, 8, n)
    actual_win = np.random.binomial(1, 0.1, n)

    features = pd.DataFrame(
        {
            "predicted_baba_index": np.random.normal(0, 1.5, n),
            "normalized_cushion": np.random.normal(0, 1, n),
            "normalized_moisture": np.random.normal(0, 1, n),
            "horse_high_speed_win_rate": np.random.uniform(0, 0.3, n),
            "horse_slow_win_rate": np.random.uniform(0, 0.3, n),
            "horse_high_speed_count": np.random.randint(0, 10, n),
            "horse_slow_count": np.random.randint(0, 10, n),
            "track_correlation": np.random.uniform(-0.5, 0.5, n),
        }
    )

    horse_baba_race_counts = np.random.randint(1, 15, n)

    # 🔥 v2.0: clip_percentileパラメータ
    model = BabaAdjustmentModel(clip_percentile=99.0)
    X_train, y_train, w_train = model.prepare_training_data(
        calibrated_pred=calibrated_pred,
        actual_win=actual_win,
        features=features,
        horse_baba_race_counts=horse_baba_race_counts,
    )

    model.train(X_train, y_train, sample_weight=w_train)

    delta_baba = model.predict_delta(
        features=features,
        horse_baba_race_counts=horse_baba_race_counts,
        apply_shrinkage=True,
    )

    final_prob = model.apply_adjustment(calibrated_pred, delta_baba)

    print("\n=== 馬場補正の例 ===")
    print("元の予測確率:", calibrated_pred[800:805])
    print("delta_baba:", delta_baba[800:805])
    print("補正後確率:", final_prob[800:805])

    print("\n=== 特徴量重要度 ===")
    print(model.get_feature_importance())
    
    print("\n" + "=" * 80)
    print("✅ v2.0完成 - ChatGPT版 + Claude追加修正完了")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
