"""
BaseWinModel v1.1 - プロダクション完成版

v1.1（2024-12-04）: 実務レベル完全修正
🔥 修正内容:
1. predict_for_race()を非推奨化（外部ヘルパーに移行推奨）
2. カテゴリカル特徴量の自動判定を改善（pd.api.types使用）
3. best_iterationの取得を改善（-1チェック）
4. early_stoppingを改善（val_dfがない場合は無効化）
5. positive_up_toをconfigから削除
6. NDCGを単着のみに修正
7. 評価関数を改善（例外処理、パフォーマンス、groupby使用）
8. docstringを完全追加
9. 型ヒントを完全追加
10. save/loadを改善（パス管理明確化）
11. feature_colsの自動推定機能を追加
12. EVベース回収率の閾値をconfigに追加
13. verbose設定を追加

v1.0: 初版（ChatGPT版 - 多数の問題あり）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Mapping
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    ndcg_score,
)


@dataclass
class BaseWinModelConfig:
    """
    BaseWinModel の設定（v1.1改善版）
    
    🔥 v1.1: positive_up_to削除、ev_threshold追加、verbose追加
    """

    lgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 100,
            "subsample": 0.7,
            "colsample_bytree": 0.6,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "verbose": -1,
        }
    )
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    
    # 🔥 v1.1: 新規追加
    ev_threshold: float = 0.15  # EVベース回収率の購入閾値
    verbose: int = 0  # 学習時のログレベル


class BaseWinModel:
    """
    単勝勝率のベースモデル（v1.1改善版）
    
    役割:
    - ペース・馬場補正前の素の勝率を予測
    - BaseFeatureBuilderが生成した特徴量を入力
    - LightGBMによる二値分類
    
    入力:
    - BaseFeatureBuilder が生成した特徴量 DataFrame
    
    出力:
    - 勝率（0〜1）
    
    🔥 v1.1: 実務レベル完全修正
    """

    VERSION = "v1.1"

    def __init__(self, config: Optional[BaseWinModelConfig] = None) -> None:
        """
        Args:
            config: ハイパーパラメータ設定
        """
        self.config = config or BaseWinModelConfig()
        self.model: Optional[lgb.Booster] = None
        self.feature_cols: List[str] = []
        self.categorical_features: List[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # 内部: カテゴリカル特徴量検出（v1.1改善版）
    # ------------------------------------------------------------------
    def _detect_categorical_features(self, X: pd.DataFrame) -> List[str]:
        """
        カテゴリカル特徴量を自動判別（v1.1改善版）
        
        🔥 v1.1: pd.api.types使用、判定ロジック改善
        
        検出基準:
        - object / category 型
        - int型かつユニーク数が2〜20（連番系は除外）
        
        Args:
            X: 特徴量DataFrame
        
        Returns:
            カテゴリカル特徴量名リスト
        """
        categorical_cols: List[str] = []

        # 🔥 v1.1: BaseFeatureBuilder v2.0と整合
        exclude_cols = [
            "horse_id",
            "race_id",
            "horse_number",
            "frame",
        ]

        for col in X.columns:
            if col in exclude_cols:
                continue

            # object / category型
            if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_categorical_dtype(X[col]):
                categorical_cols.append(col)
                continue
            
            # 🔥 v1.1: is_integer_dtype()を使用
            if pd.api.types.is_integer_dtype(X[col]):
                n_unique = X[col].nunique()
                # 2値以上20値以下をカテゴリカルとみなす
                if 2 <= n_unique <= 20:
                    categorical_cols.append(col)

        return categorical_cols

    # ------------------------------------------------------------------
    # 学習（v1.1改善版）
    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        *,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "win_flag",
        val_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        特徴量テーブルからLightGBMを学習（v1.1改善版）
        
        🔥 v1.1: feature_cols自動推定、early_stopping改善
        
        Args:
            train_df: 学習データ（特徴量 + ラベル）
            feature_cols: 使用する特徴量カラム名（Noneの場合は自動推定）
            target_col: ラベル列名（デフォルト: "win_flag"）
            val_df: 検証データ（early stopping用、オプション）
        
        Raises:
            ValueError: target_colが存在しない場合
            ValueError: 指定されたfeature_colsが存在しない場合
        """
        if target_col not in train_df.columns:
            raise ValueError(f"target_col '{target_col}' が train_df に存在しません")

        # 🔥 v1.1: feature_cols自動推定
        if feature_cols is None:
            exclude_cols = ["horse_id", "race_id", target_col]
            self.feature_cols = [c for c in train_df.columns if c not in exclude_cols]
            warnings.warn(
                f"feature_colsが指定されていないため、{len(self.feature_cols)}個の特徴量を自動推定しました。"
            )
        else:
            # 存在確認
            missing = set(feature_cols) - set(train_df.columns)
            if missing:
                raise ValueError(f"指定された特徴量が train_df に存在しません: {missing}")
            self.feature_cols = list(feature_cols)

        X_train = train_df[self.feature_cols]
        y_train = train_df[target_col].astype(float)

        # カテゴリカル自動判定
        self.categorical_features = self._detect_categorical_features(X_train)

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=self.categorical_features or None,
            free_raw_data=False,
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        # 🔥 v1.1: val_dfの処理を改善
        if val_df is not None:
            if target_col not in val_df.columns:
                raise ValueError(f"target_col '{target_col}' が val_df に存在しません")
            X_val = val_df[self.feature_cols]
            y_val = val_df[target_col].astype(float)
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
                categorical_feature=self.categorical_features or None,
                free_raw_data=False,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")
        else:
            val_data = None

        # 🔥 v1.1: early_stoppingはval_dfがある場合のみ
        callbacks = []
        if val_df is not None:
            callbacks.append(
                lgb.early_stopping(
                    self.config.early_stopping_rounds,
                    verbose=bool(self.config.verbose),
                )
            )
        
        if self.config.verbose > 0:
            callbacks.append(lgb.log_evaluation(period=100))

        self.model = lgb.train(
            self.config.lgbm_params,
            train_data,
            num_boost_round=self.config.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks or None,
        )

        self._fitted = True

    # ------------------------------------------------------------------
    # 予測（v1.1改善版）
    # ------------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        勝率を予測（v1.1改善版）
        
        🔥 v1.1: best_iteration取得を改善
        
        Args:
            df: BaseFeatureBuilder が生成した特徴量 DataFrame
        
        Returns:
            shape = (N,) の numpy array（0〜1）
        
        Raises:
            ValueError: モデルが未学習の場合
        """
        if not self._fitted or self.model is None:
            raise ValueError(
                "BaseWinModel がまだ学習されていません。fit() を先に呼んでください。"
            )

        X = df[self.feature_cols]
        
        # 🔥 v1.1: best_iteration取得を改善
        num_iter = self.model.best_iteration
        if num_iter < 0:
            num_iter = self.model.current_iteration()
        
        preds = self.model.predict(X, num_iteration=num_iter)
        return preds.astype(float)

    # ------------------------------------------------------------------
    # レース単位の予測（v1.1非推奨化）
    # ------------------------------------------------------------------
    def predict_for_race(
        self,
        entries_df: pd.DataFrame,
        race_row: Mapping[str, Any],
        as_of: Any,
        race_feature_builder: Any,  # RaceFeatureBuilder
        feature_builder: Any,       # BaseFeatureBuilder
    ) -> Dict[str, float]:
        """
        実運用時用の簡易API（v1.1非推奨）
        
        🔥 v1.1: 非推奨化
        
        このメソッドはモデルクラスに推論パイプラインを含めるため、
        責務が混乱します。代わりに外部ヘルパー関数の使用を推奨します。
        
        推奨実装:
        ```python
        def predict_win_probs(
            base_model: BaseWinModel,
            entries_df: pd.DataFrame,
            race_row: Mapping[str, Any],
            as_of: Any,
            race_feature_builder: RaceFeatureBuilder,
            feature_builder: BaseFeatureBuilder,
        ) -> Dict[str, float]:
            # 外部のヘルパー関数として実装
        ```
        
        Args:
            entries_df: 出馬表
            race_row: レース情報
            as_of: 基準日時
            race_feature_builder: RaceFeatureBuilder v5
            feature_builder: BaseFeatureBuilder v2
        
        Returns:
            {horse_id: win_prob}
        """
        warnings.warn(
            "predict_for_race()は非推奨です。外部ヘルパー関数の使用を推奨します。",
            DeprecationWarning,
            stacklevel=2,
        )
        
        if "horse_id" not in entries_df.columns:
            raise ValueError("entries_df に 'horse_id' カラムが必要です。")

        race_feature_output = race_feature_builder.build_for_race(
            race_row=race_row,
            entries_df=entries_df,
            as_of=as_of,
        )

        feat_df = feature_builder.build_features_for_race(
            entries_df=entries_df,
            race_row=race_row,
            as_of=as_of,
            race_feature_output=race_feature_output,
        )

        probs = self.predict_proba(feat_df)
        horse_ids = feat_df["horse_id"].astype(str).tolist()

        return {hid: float(p) for hid, p in zip(horse_ids, probs)}

    # ------------------------------------------------------------------
    # 評価（v1.1改善版）
    # ------------------------------------------------------------------
    def evaluate(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        race_ids: Optional[pd.Series] = None,
        finish_positions: Optional[pd.Series] = None,
        odds: Optional[pd.Series] = None,
        ev_threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        包括的な評価メトリクスを計算（v1.1改善版）
        
        🔥 v1.1: 例外処理改善、パフォーマンス改善、NDCG修正
        
        Args:
            df: 特徴量DataFrame
            y: 真のラベル（0/1）
            race_ids: レースID（レース単位評価用、オプション）
            finish_positions: 着順（NDCG計算用、オプション）
            odds: オッズ（回収率計算用、オプション）
            ev_threshold: EV購入閾値（Noneの場合はconfigから取得）
        
        Returns:
            評価メトリクス辞書:
                - brier_score: Brierスコア
                - log_loss: 対数損失
                - auc: ROC-AUC
                - ndcg: NDCG（レース単位、1着のみ評価）
                - top1_accuracy: レース単位正解率
                - recovery_rate_ev: EVベース回収率
        """
        y = y.astype(int)
        y_pred = self.predict_proba(df)

        metrics: Dict[str, float] = {}

        # 1. Brier Score
        metrics["brier_score"] = float(brier_score_loss(y, y_pred))

        # 2. Log Loss
        metrics["log_loss"] = float(log_loss(y, y_pred))

        # 3. AUC（v1.1改善版）
        try:
            metrics["auc"] = float(roc_auc_score(y, y_pred))
        except ValueError as e:
            warnings.warn(
                f"AUC計算に失敗: {e}（全てのラベルが同じ値の可能性があります）"
            )
            metrics["auc"] = float("nan")

        # レース単位評価が不要な場合はここで終了
        if race_ids is None:
            return metrics

        # 4. NDCG（v1.1修正版: 単着のみ評価）
        if finish_positions is not None:
            ndcg_scores: List[float] = []
            
            # 🔥 v1.1: groupbyでパフォーマンス改善
            for rid, indices in pd.Series(race_ids).groupby(race_ids).groups.items():
                pos = finish_positions.iloc[indices].values
                pred = y_pred[indices]

                if len(pos) <= 1:
                    continue

                # 🔥 v1.1: 単着のみ評価（1着=1, それ以外=0）
                true_rel = (pos == 1).astype(int).reshape(1, -1)
                pred_scores = pred.reshape(1, -1)

                try:
                    ndcg = ndcg_score(true_rel, pred_scores)
                    ndcg_scores.append(float(ndcg))
                except Exception as e:
                    warnings.warn(f"NDCG計算に失敗（レース{rid}）: {e}")
                    continue

            if ndcg_scores:
                metrics["ndcg"] = float(np.mean(ndcg_scores))

        # 5. Top-1 Accuracy（レース単位）
        top1_correct = 0
        total_races = 0
        
        for rid, indices in pd.Series(race_ids).groupby(race_ids).groups.items():
            r_y = y.iloc[indices].values
            r_pred = y_pred[indices]
            
            if len(r_y) <= 1:
                continue
            
            top_idx = int(np.argmax(r_pred))
            if r_y[top_idx] == 1:
                top1_correct += 1
            total_races += 1
        
        if total_races > 0:
            metrics["top1_accuracy"] = float(top1_correct / total_races)

        # 6. EVベース回収率
        if odds is not None:
            threshold = ev_threshold if ev_threshold is not None else self.config.ev_threshold
            metrics["recovery_rate_ev"] = self._calculate_ev_based_recovery(
                y_pred=y_pred,
                y_true=y,
                race_ids=race_ids,
                odds=odds,
                threshold=threshold,
            )

        return metrics

    def _calculate_ev_based_recovery(
        self,
        y_pred: np.ndarray,
        y_true: pd.Series,
        race_ids: pd.Series,
        odds: pd.Series,
        threshold: float,
    ) -> float:
        """
        EV（期待値）ベースの回収率を計算（v1.1改善版）
        
        🔥 v1.1: docstring追加、パフォーマンス改善
        
        計算方法:
        1. 各馬のEV = p * oddsを計算
        2. EV > 1.0 かつ p > thresholdの馬を購入
        3. レース単位で100円を購入馬に均等配分
        4. 総回収額 / 総投資額
        
        Args:
            y_pred: 予測確率
            y_true: 真のラベル（0/1）
            race_ids: レースID
            odds: オッズ
            threshold: 購入する最低確率
        
        Returns:
            回収率（total_return / total_bet）
        """
        total_bet = 0.0
        total_return = 0.0

        for rid, indices in pd.Series(race_ids).groupby(race_ids).groups.items():
            r_pred = y_pred[indices]
            r_true = y_true.iloc[indices].values
            r_odds = odds.iloc[indices].values

            if len(r_pred) <= 1:
                continue

            # EV計算
            ev = r_pred * r_odds
            buy_mask = (ev > 1.0) & (r_pred > threshold)

            if not np.any(buy_mask):
                continue

            # 均等購入
            n_buy = int(np.sum(buy_mask))
            bet_per_horse = 100.0 / n_buy
            total_bet += 100.0

            # 配当計算
            for idx in np.where(buy_mask)[0]:
                if r_true[idx] == 1:
                    total_return += bet_per_horse * r_odds[idx]

        if total_bet == 0:
            return 0.0

        return float(total_return / total_bet)

    # ------------------------------------------------------------------
    # モデルの可視化・永続化（v1.1改善版）
    # ------------------------------------------------------------------
    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: Optional[int] = 30,
    ) -> pd.DataFrame:
        """
        特徴量重要度を取得
        
        Args:
            importance_type: 重要度タイプ（"gain", "split"等）
            top_n: 上位N件を取得（Noneの場合は全件）
        
        Returns:
            特徴量重要度DataFrame
        
        Raises:
            ValueError: モデルが未学習の場合
        """
        if not self._fitted or self.model is None:
            raise ValueError("BaseWinModel がまだ学習されていません。")

        importances = self.model.feature_importance(importance_type=importance_type)
        df = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)
        
        if top_n is not None:
            df = df.head(top_n)
        
        return df

    def save(self, path: str) -> None:
        """
        モデルを保存（v1.1改善版）
        
        🔥 v1.1: パス管理明確化
        
        Args:
            path: モデルファイルのパス（例: "model.txt"）
        
        保存内容:
        - {path}: LightGBMモデル
        - {path}_meta.json: メタ情報（特徴量名等）
        
        Raises:
            ValueError: モデルが未学習の場合
        """
        if not self._fitted or self.model is None:
            raise ValueError("BaseWinModel がまだ学習されていません。")

        # モデル保存
        self.model.save_model(path)

        # メタ情報保存
        meta = {
            "version": self.VERSION,
            "feature_cols": self.feature_cols,
            "categorical_features": self.categorical_features,
            "config": {
                "lgbm_params": self.config.lgbm_params,
                "num_boost_round": self.config.num_boost_round,
                "early_stopping_rounds": self.config.early_stopping_rounds,
                "ev_threshold": self.config.ev_threshold,
                "verbose": self.config.verbose,
            },
        }

        import json
        from pathlib import Path

        # 🔥 v1.1: パス管理明確化
        model_path = Path(path)
        meta_path = model_path.parent / f"{model_path.stem}_meta.json"
        
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """
        モデルを読み込み（v1.1改善版）
        
        Args:
            path: モデルファイルのパス
        
        Raises:
            FileNotFoundError: ファイルが存在しない場合
        """
        from pathlib import Path
        import json
        
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
        
        # モデル読み込み
        self.model = lgb.Booster(model_file=path)
        self._fitted = True

        # メタ情報読み込み
        meta_path = model_path.parent / f"{model_path.stem}_meta.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            
            self.feature_cols = meta.get("feature_cols", [])
            self.categorical_features = meta.get("categorical_features", [])
            
            cfg = meta.get("config", {})
            self.config.lgbm_params = cfg.get("lgbm_params", self.config.lgbm_params)
            self.config.num_boost_round = cfg.get("num_boost_round", self.config.num_boost_round)
            self.config.early_stopping_rounds = cfg.get("early_stopping_rounds", self.config.early_stopping_rounds)
            self.config.ev_threshold = cfg.get("ev_threshold", self.config.ev_threshold)
            self.config.verbose = cfg.get("verbose", self.config.verbose)
        else:
            warnings.warn(f"メタファイルが見つかりません: {meta_path}")


# ----------------------------------------------------------------------
# 補助関数
# ----------------------------------------------------------------------
def create_win_labels(
    finish_positions: pd.Series,
    positive_up_to: int = 1,
) -> pd.Series:
    """
    着順から win_flag を作成
    
    Args:
        finish_positions: 着順（1,2,3,...）
        positive_up_to: 何着まで 1 とするか（1 or 2 推奨）
    
    Returns:
        win_flag（0/1）の Series
    
    Example:
        >>> finish_positions = pd.Series([1, 3, 2, 5])
        >>> create_win_labels(finish_positions, positive_up_to=1)
        0    1
        1    0
        2    0
        3    0
        dtype: int64
    """
    return (finish_positions <= positive_up_to).astype(int)


def example_usage():
    """使用例（v1.1）"""
    
    print("=" * 80)
    print("BaseWinModel v1.1 - 使用例（プロダクション完成版）")
    print("=" * 80)
    
    print("\n✅ v1.1完成 - 実務レベル完全修正")
    print("  - predict_for_race()非推奨化（外部ヘルパー推奨）")
    print("  - カテゴリカル特徴量判定改善")
    print("  - best_iteration取得改善")
    print("  - early_stopping改善（val_dfがない場合は無効化）")
    print("  - positive_up_to削除")
    print("  - NDCG修正（単着のみ評価）")
    print("  - 評価関数改善（例外処理、パフォーマンス）")
    print("  - docstring完全追加")
    print("  - 型ヒント完全追加")
    print("  - save/load改善")
    print("  - feature_cols自動推定")
    print("  - EVベース回収率閾値設定化")


if __name__ == "__main__":
    example_usage()
