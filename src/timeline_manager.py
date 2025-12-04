"""
TimelineManager: データリークを防ぐ時系列データ管理

役割:
1. race_dateより後の情報を参照不可にする
2. ウォークフォワードsplitを返す
3. オッズのタイムライン管理
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class DataSplit:
    """データ分割の情報を保持"""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_indices: np.ndarray
    test_indices: np.ndarray


class TimelineManager:
    """
    時系列データの管理とウォークフォワードCV
    
    重要な原則:
    - race_dateより後のデータは絶対に参照しない
    - オッズは「実際の発売タイミング」と整合させる
    - ウォークフォワード方式で学習・評価
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        date_column: str = 'race_date',
        cutoff_time: str = '前日15:00'
    ):
        """
        Args:
            data: レースデータ（DataFrame）
            date_column: 日付カラム名
            cutoff_time: データ取得基準時点
        """
        self.data = data.copy()
        self.date_column = date_column
        self.cutoff_time = cutoff_time
        
        # 日付でソート
        self.data = self.data.sort_values(date_column).reset_index(drop=True)
        
        # 日付をdatetimeに変換
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            self.data[date_column] = pd.to_datetime(self.data[date_column])
    
    def walk_forward_split(
        self,
        n_splits: int = 5,
        test_size_months: int = 3,
        gap_days: int = 0
    ) -> List[DataSplit]:
        """
        ウォークフォワードCV用のデータ分割
        
        例: n_splits=5, test_size_months=3
        Fold 1: Train [2020-01 ~ 2020-12], Test [2021-01 ~ 2021-03]
        Fold 2: Train [2020-01 ~ 2021-03], Test [2021-04 ~ 2021-06]
        Fold 3: Train [2020-01 ~ 2021-06], Test [2021-07 ~ 2021-09]
        ...
        
        Args:
            n_splits: 分割数
            test_size_months: テスト期間（月）
            gap_days: 学習期間とテスト期間の間のギャップ（日）
        
        Returns:
            DataSplitのリスト
        """
        
        min_date = self.data[self.date_column].min()
        max_date = self.data[self.date_column].max()
        
        # 全期間（月数）
        total_months = (max_date.year - min_date.year) * 12 + \
                      (max_date.month - min_date.month)
        
        # 最小学習期間（月）
        min_train_months = 12  # 最低1年分
        
        # テスト期間の開始月を計算
        test_start_months = []
        for i in range(n_splits):
            test_start_month = min_train_months + i * test_size_months
            if test_start_month + test_size_months <= total_months:
                test_start_months.append(test_start_month)
        
        splits = []
        
        for fold, test_start_month in enumerate(test_start_months):
            # テスト期間の計算
            test_start = min_date + pd.DateOffset(months=test_start_month)
            test_end = test_start + pd.DateOffset(months=test_size_months)
            
            # ギャップを考慮した学習期間の終了
            train_end = test_start - timedelta(days=gap_days)
            
            # 学習期間の開始
            train_start = min_date
            
            # インデックスの抽出
            train_mask = (self.data[self.date_column] >= train_start) & \
                        (self.data[self.date_column] < train_end)
            test_mask = (self.data[self.date_column] >= test_start) & \
                       (self.data[self.date_column] < test_end)
            
            train_indices = self.data[train_mask].index.to_numpy()
            test_indices = self.data[test_mask].index.to_numpy()
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                split = DataSplit(
                    fold=fold + 1,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_indices=train_indices,
                    test_indices=test_indices
                )
                splits.append(split)
        
        return splits
    
    def get_safe_features(
        self,
        race_id: str,
        as_of_date: Optional[datetime] = None,
        include_odds: bool = False
    ) -> Dict:
        """
        タイムライン安全な特徴量のみ取得
        
        Args:
            race_id: レースID
            as_of_date: データ取得時点（Noneの場合はレース前日）
            include_odds: オッズを含めるか（本番予測時のみTrue）
        
        Returns:
            安全な特徴量の辞書
        """
        
        race_data = self.data[self.data['race_id'] == race_id].iloc[0]
        race_date = race_data[self.date_column]
        
        if as_of_date is None:
            # デフォルト: レース前日15:00
            as_of_date = race_date - timedelta(days=1)
        
        # race_dateより後のデータは参照不可
        if as_of_date > race_date:
            raise ValueError(
                f"as_of_date ({as_of_date}) cannot be after race_date ({race_date})"
            )
        
        # 安全な特徴量のみ抽出
        safe_features = {
            # レース情報（事前に確定）
            'race_id': race_id,
            'race_date': race_date,
            'track_name': race_data.get('track_name'),
            'distance': race_data.get('distance'),
            'track_type': race_data.get('track_type'),
            'track_condition': race_data.get('track_condition'),
            
            # 馬場状態（当日朝に取得可能）
            'moisture': race_data.get('moisture'),
            'cushion_value': race_data.get('cushion_value'),
            'baba_index_predicted': race_data.get('baba_index_predicted'),
            
            # 馬情報（事前に確定）
            'horse_id': race_data.get('horse_id'),
            'horse_age': race_data.get('horse_age'),
            'sex': race_data.get('sex'),
            'weight': race_data.get('weight'),
            'weight_change': race_data.get('weight_change'),
            'burden_weight': race_data.get('burden_weight'),
            'gate_number': race_data.get('gate_number'),
            'horse_number': race_data.get('horse_number'),
            
            # 過去成績（as_of_dateまでのデータ）
            'past_3_avg_position': self._get_past_performance(
                race_data.get('horse_id'),
                as_of_date,
                n_races=3
            ),
            
            # 騎手・調教師（事前に確定）
            'jockey_id': race_data.get('jockey_id'),
            'trainer_id': race_data.get('trainer_id'),
            'jockey_win_rate': race_data.get('jockey_win_rate'),
            'trainer_win_rate': race_data.get('trainer_win_rate'),
        }
        
        # オッズ（本番予測時のみ）
        if include_odds:
            safe_features['odds'] = race_data.get('odds')
            safe_features['popularity'] = race_data.get('popularity')
        
        return safe_features
    
    def _get_past_performance(
        self,
        horse_id: str,
        as_of_date: datetime,
        n_races: int = 3
    ) -> float:
        """
        as_of_date時点での過去成績を取得
        
        重要: race_dateがas_of_dateより前のレースのみ使用
        """
        
        past_races = self.data[
            (self.data['horse_id'] == horse_id) &
            (self.data[self.date_column] < as_of_date)
        ].sort_values(self.date_column, ascending=False).head(n_races)
        
        if len(past_races) == 0:
            return np.nan
        
        return past_races['finish_position'].mean()
    
    def validate_no_leakage(
        self,
        feature_df: pd.DataFrame,
        target_df: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        データリークがないか検証
        
        Returns:
            (リークなし, 問題リスト)
        """
        
        issues = []
        
        # 1. race_dateの順序チェック
        if self.date_column in feature_df.columns:
            if not feature_df[self.date_column].is_monotonic_increasing:
                issues.append("race_dateが昇順になっていません")
        
        # 2. 未来情報のチェック（カラム名ベース）
        forbidden_keywords = [
            'finish_position',  # 着順（結果）
            'finish_time',      # タイム（結果）
            'actual_',          # 実際の〜
            'result_',          # 結果の〜
        ]
        
        for col in feature_df.columns:
            for keyword in forbidden_keywords:
                if keyword in col.lower():
                    issues.append(f"未来情報の可能性: {col}")
        
        # 3. target_dfとfeature_dfのインデックス一致チェック
        if not feature_df.index.equals(target_df.index):
            issues.append("feature_dfとtarget_dfのインデックスが一致しません")
        
        return len(issues) == 0, issues


def example_usage():
    """使用例"""
    
    # ダミーデータ作成
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='W')
    data = pd.DataFrame({
        'race_id': [f'race_{i}' for i in range(len(dates))],
        'race_date': dates,
        'track_name': np.random.choice(['東京', '中山', '京都'], len(dates)),
        'distance': np.random.choice([1600, 1800, 2000], len(dates)),
        'horse_id': [f'horse_{i % 100}' for i in range(len(dates))],
        'finish_position': np.random.randint(1, 19, len(dates))
    })
    
    # TimelineManager初期化
    tm = TimelineManager(data, date_column='race_date')
    
    # ウォークフォワードsplit
    splits = tm.walk_forward_split(n_splits=5, test_size_months=3)
    
    print("=== ウォークフォワードCV ===")
    for split in splits:
        print(f"\nFold {split.fold}:")
        print(f"  Train: {split.train_start.date()} ~ {split.train_end.date()} ({len(split.train_indices)} samples)")
        print(f"  Test:  {split.test_start.date()} ~ {split.test_end.date()} ({len(split.test_indices)} samples)")


if __name__ == "__main__":
    example_usage()
