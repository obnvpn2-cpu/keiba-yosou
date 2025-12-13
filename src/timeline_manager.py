"""
TimelineManager: データリークを防ぐ時系列データ管理（v5.1 - 実運用完成版）

v5.1（2024-12-04）: ChatGPT最終レビュー対応
🔥 実運用レベル完成:
1. RESULTパターンを厳密化（誤爆防止）
2. オッズパターンを最終オッズのみに限定
3. 縦持ちテーブルでfeature_nameごとに最新行のみ取得
4. strict_mode=Trueをデフォルトに変更
5. 横持ちの警告追加

v5.0: カラム名パターンマッチング、縦持ち対応、horse_id単位管理
v4.0: 過去成績別テーブル化、タイムライン導入
v3.0: データ取得タイミング管理、race_time考慮
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo
from enum import Enum
import warnings
import re


class DataAvailability(Enum):
    """データの取得可能時点を定義"""
    PRE_RACE = 'pre_race'
    MORNING = 'morning'
    PADDOCK = 'paddock'
    JUST_BEFORE = 'just_before'
    RESULT = 'result'


# 最小限のFEATURE_AVAILABILITY
FEATURE_AVAILABILITY = {
    'race_id': DataAvailability.PRE_RACE,
    'horse_id': DataAvailability.PRE_RACE,
    'race_date': DataAvailability.PRE_RACE,
    'finish_position': DataAvailability.RESULT,
    'finish_time': DataAvailability.RESULT,
    '着順': DataAvailability.RESULT,
    '着差': DataAvailability.RESULT,
}

# 🔥 v5.1: パターンマッチング厳密化（誤爆防止）
COLUMN_PATTERNS = {
    # RESULT（厳密に限定）
    DataAvailability.RESULT: [
        r'^finish_position$', r'^finish_time$', r'^final_3f$', r'^final_3F$',
        r'.*着順.*', r'.*着差.*', r'.*着時間.*',
        r'.*上がり.*3[fF].*', r'.*上がり.*タイム.*',
        r'.*通過順.*', r'.*コーナー.*通過.*', r'.*passing.*order.*',
        r'.*prize.*money.*', r'.*払戻.*', r'.*payout.*',
        r'.*人気結果.*', r'.*final.*popularity.*'
    ],
    
    # JUST_BEFORE（最終オッズのみ）
    DataAvailability.JUST_BEFORE: [
        r'^odds$', r'^odds_win$', r'^odds_place$', r'^odds_show$',
        r'^popularity$', r'^人気$', r'^人気順位$',
        r'.*最終.*オッズ.*', r'.*直前.*オッズ.*'
    ],
    
    # PADDOCK（馬体重のみ）
    DataAvailability.PADDOCK: [
        r'^weight$', r'^horse_weight$', r'^weight_change$',
        r'.*馬体重$', r'.*体重増減.*', r'.*体重変化.*'
    ],
    
    # MORNING（当日朝データ）
    DataAvailability.MORNING: [
        r'.*moisture.*', r'.*cushion.*', r'^track_condition$',
        r'.*馬場状態.*', r'.*weather$', r'.*天候.*', r'.*天気.*',
        r'.*baba.*index.*', r'.*馬場指数.*'
    ],
}


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
    時系列データの管理とウォークフォワードCV（v5.1 - 実運用完成版）
    
    🔥 v5.1での重要な変更（ChatGPT最終レビュー対応）:
    1. RESULTパターン厳密化（time等の汎用単語を除外）
    2. オッズパターン厳密化（最終オッズのみ）
    3. 縦持ちテーブルでfeature_nameごとに最新行のみ取得
    4. strict_mode=Trueをデフォルト（安全第一）
    5. 横持ちDataFrameの場合は警告を表示
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        past_performance_table: Optional[pd.DataFrame] = None,
        time_series_features_table: Optional[pd.DataFrame] = None,
        date_column: str = 'race_date',
        time_column: Optional[str] = None,
        cutoff_time: time = time(15, 0),
        tz: str = 'Asia/Tokyo',
        strict_mode: bool = True,
        auto_infer_levels: bool = True
    ):
        """
        Args:
            data: レースデータ（DataFrame）
            past_performance_table: 過去成績テーブル（推奨）
            time_series_features_table: 縦持ち時点管理テーブル（推奨）
            date_column: 日付カラム名
            time_column: レース時刻カラム名
            cutoff_time: データ取得基準時刻
            tz: タイムゾーン
            strict_mode: 厳格モード（デフォルト: True）🔥
            auto_infer_levels: カラム名からレベル自動推定
        """
        self.data = data.copy()
        self.past_performance_table = past_performance_table
        self.time_series_features_table = time_series_features_table
        self.date_column = date_column
        self.time_column = time_column
        self.cutoff_time = cutoff_time
        self.tz = ZoneInfo(tz)
        self.strict_mode = strict_mode
        self.auto_infer_levels = auto_infer_levels
        
        # 必須カラムチェック
        required_columns = ['race_id', 'horse_id', date_column]
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            raise ValueError(f"必須カラムが不足: {missing}")
        
        # 過去成績テーブル検証
        if past_performance_table is not None:
            required_perf_cols = ['horse_id', 'as_of_date']
            missing_perf = [c for c in required_perf_cols if c not in past_performance_table.columns]
            if missing_perf:
                warnings.warn(f"past_performance_tableに推奨カラムが不足: {missing_perf}")
        
        # 🔥 v5.1: 縦持ち時点管理テーブル検証
        if time_series_features_table is not None:
            required_ts_cols = ['race_id', 'feature_name', 'value', 'timestamp']
            missing_ts = [c for c in required_ts_cols if c not in time_series_features_table.columns]
            if missing_ts:
                raise ValueError(f"time_series_features_tableに必須カラムが不足: {missing_ts}")
            
            if not pd.api.types.is_datetime64_any_dtype(time_series_features_table['timestamp']):
                self.time_series_features_table['timestamp'] = pd.to_datetime(
                    time_series_features_table['timestamp']
                )
        else:
            # 🔥 v5.1: 横持ちテーブルのみの場合は警告
            warnings.warn(
                "time_series_features_tableが指定されていません。\n"
                "横持ちDataFrameのみでは時点管理が不完全です。\n"
                "本番運用では time_series_features_table の使用を強く推奨します。",
                UserWarning
            )
        
        # 日付でソート
        self.data = self.data.sort_values(date_column).reset_index(drop=True)
        
        # 日付をdatetimeに変換
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            self.data[self.date_column] = pd.to_datetime(self.data[date_column])
        
        self._past_performance_cache = {}
        
        # カラムレベルを自動推定してキャッシュ
        self._column_level_cache = {}
        if auto_infer_levels:
            self._build_column_level_cache()
    
    def _build_column_level_cache(self):
        """カラム名からデータレベルを自動推定してキャッシュ"""
        for col in self.data.columns:
            if col in ['index', 'level_0']:
                continue
            
            if col in FEATURE_AVAILABILITY:
                self._column_level_cache[col] = FEATURE_AVAILABILITY[col]
                continue
            
            inferred_level = self._infer_column_level(col)
            self._column_level_cache[col] = inferred_level
    
    def _infer_column_level(self, column: str) -> DataAvailability:
        """
        カラム名からデータレベルを推定（v5.1厳密化版）
        
        Args:
            column: カラム名
        
        Returns:
            推定されたDataAvailability
        """
        # 🔥 v5.1: パターンマッチング（厳密化）
        for level, patterns in COLUMN_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, column, re.IGNORECASE):
                    return level
        
        # デフォルトはPRE_RACE
        return DataAvailability.PRE_RACE
    
    def walk_forward_split(
        self,
        n_splits: int = 5,
        test_size_months: int = 3,
        gap_days: int = 0,
        min_train_months: int = 12
    ) -> List[DataSplit]:
        """ウォークフォワードCV用のデータ分割"""
        
        min_date = self.data[self.date_column].min()
        max_date = self.data[self.date_column].max()
        
        test_start_dates = []
        current = min_date + relativedelta(months=min_train_months)
        
        while current + relativedelta(months=test_size_months) <= max_date:
            test_start_dates.append(current)
            current += relativedelta(months=test_size_months)
            
            if len(test_start_dates) >= n_splits:
                break
        
        if len(test_start_dates) == 0:
            raise ValueError(
                f"データ期間が短すぎます。最小要件: {min_train_months + test_size_months}ヶ月"
            )
        
        splits = []
        
        for fold, test_start in enumerate(test_start_dates):
            test_end = test_start + relativedelta(months=test_size_months)
            train_end = test_start - timedelta(days=gap_days)
            train_start = min_date
            
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
    
    def get_race_datetime(self, race_id: str) -> datetime:
        """レースの開催日時を取得"""
        race_rows = self.data[self.data['race_id'] == race_id]
        if race_rows.empty:
            raise KeyError(f"race_id {race_id} が見つかりません")
        
        race_date = race_rows.iloc[0][self.date_column]
        
        if self.time_column and self.time_column in race_rows.columns:
            race_time = race_rows.iloc[0][self.time_column]
            if pd.notna(race_time):
                if isinstance(race_time, str):
                    race_time = datetime.strptime(race_time, '%H:%M').time()
                race_datetime = datetime.combine(
                    race_date.date(),
                    race_time,
                    tzinfo=self.tz
                )
            else:
                race_datetime = datetime.combine(
                    race_date.date(),
                    time(15, 0),
                    tzinfo=self.tz
                )
        else:
            race_datetime = datetime.combine(
                race_date.date(),
                time(15, 0),
                tzinfo=self.tz
            )
        
        return race_datetime
    
    def get_safe_features(
        self,
        race_id: str,
        horse_id: str,
        as_of_datetime: Optional[datetime] = None,
        max_availability: DataAvailability = DataAvailability.PRE_RACE,
        include_features: Optional[Set[str]] = None
    ) -> Dict:
        """
        horse_id単位で安全な特徴量のみ取得（v5.1実運用版）
        
        Args:
            race_id: レースID
            horse_id: 馬ID
            as_of_datetime: データ取得時点
            max_availability: 取得可能な最大レベル
            include_features: 明示的に含める特徴量セット
        
        Returns:
            安全な特徴量の辞書
        """
        
        if as_of_datetime is None:
            as_of_datetime = self._calculate_as_of_datetime(race_id, max_availability)
        
        if as_of_datetime.tzinfo is None:
            as_of_datetime = as_of_datetime.replace(tzinfo=self.tz)
        
        safe_features = {
            'race_id': race_id,
            'horse_id': horse_id,
            'as_of_datetime': as_of_datetime,
            'max_availability': max_availability.value,
        }
        
        # 縦持ち時点管理テーブルを優先
        if self.time_series_features_table is not None:
            ts_features = self._get_features_from_time_series(
                race_id, horse_id, as_of_datetime, max_availability
            )
            safe_features.update(ts_features)
        else:
            # フォールバック: 横持ちDataFrame
            race_features = self._get_features_from_dataframe(
                race_id, horse_id, as_of_datetime, max_availability, include_features
            )
            safe_features.update(race_features)
        
        # 過去成績を追加
        if self.past_performance_table is not None:
            past_perf = self._get_past_performance_from_table(horse_id, as_of_datetime)
            safe_features.update(past_perf)
        
        return safe_features
    
    def _get_features_from_time_series(
        self,
        race_id: str,
        horse_id: str,
        as_of_datetime: datetime,
        max_availability: DataAvailability
    ) -> Dict:
        """
        縦持ち時点管理テーブルから特徴量を取得（v5.1改善版）
        
        🔥 v5.1: feature_nameごとに最新の1行のみ取得
        
        Args:
            race_id: レースID
            horse_id: 馬ID
            as_of_datetime: 取得時点
            max_availability: 最大レベル
        
        Returns:
            特徴量の辞書
        """
        features = {}
        
        # race_idでフィルタ
        ts_rows = self.time_series_features_table[
            self.time_series_features_table['race_id'] == race_id
        ]
        
        # horse_idカラムがあればフィルタ
        if 'horse_id' in self.time_series_features_table.columns:
            ts_rows = ts_rows[
                (ts_rows['horse_id'] == horse_id) | (ts_rows['horse_id'].isna())
            ]
        
        # as_of_datetime以前のみ
        ts_rows = ts_rows[ts_rows['timestamp'] <= as_of_datetime]
        
        # 🔥 v5.1: feature_nameごとに最新の1行のみ取得
        if not ts_rows.empty:
            ts_rows = ts_rows.sort_values('timestamp', ascending=False)
            ts_rows = ts_rows.drop_duplicates(subset=['feature_name'], keep='first')
        
        for _, row in ts_rows.iterrows():
            feature_name = row['feature_name']
            value = row['value']
            
            # レベルチェック
            if 'availability_level' in row.index:
                feature_level = DataAvailability(row['availability_level'])
            else:
                feature_level = self._column_level_cache.get(
                    feature_name,
                    self._infer_column_level(feature_name)
                )
            
            # RESULTレベルは除外
            if feature_level == DataAvailability.RESULT:
                continue
            
            # max_availabilityチェック
            if not self._is_available(feature_level, max_availability):
                continue
            
            features[feature_name] = value
        
        return features
    
    def _get_features_from_dataframe(
        self,
        race_id: str,
        horse_id: str,
        as_of_datetime: datetime,
        max_availability: DataAvailability,
        include_features: Optional[Set[str]] = None
    ) -> Dict:
        """横持ちDataFrameから特徴量を取得（フォールバック）"""
        
        race_rows = self.data[self.data['race_id'] == race_id]
        if race_rows.empty:
            raise KeyError(f"race_id {race_id} が見つかりません")
        
        horse_row = race_rows[race_rows['horse_id'] == horse_id]
        if horse_row.empty:
            raise KeyError(f"horse_id {horse_id} が race_id {race_id} に存在しません")
        
        race_data = horse_row.iloc[0]
        features = {}
        
        for column in race_data.index:
            if column in ['index', 'level_0', 'race_id', 'horse_id']:
                continue
            
            # レベル取得
            feature_level = self._column_level_cache.get(
                column,
                self._infer_column_level(column)
            )
            
            # RESULTレベルは除外
            if feature_level == DataAvailability.RESULT:
                continue
            
            # max_availabilityチェック
            if not self._is_available(feature_level, max_availability):
                continue
            
            # 未知のレベルは警告（strict_modeのみ）
            if feature_level is None and self.strict_mode:
                raise ValueError(f"列 '{column}' のレベルが不明です")
            
            if include_features is None or column in include_features:
                features[column] = race_data.get(column)
        
        return features
    
    def _get_past_performance_from_table(
        self,
        horse_id: str,
        as_of_datetime: datetime
    ) -> Dict:
        """過去成績テーブルから取得"""
        
        timestamp = int(as_of_datetime.timestamp())
        cache_key = (horse_id, timestamp)
        
        if cache_key in self._past_performance_cache:
            return self._past_performance_cache[cache_key]
        
        perf_rows = self.past_performance_table[
            (self.past_performance_table['horse_id'] == horse_id) &
            (pd.to_datetime(self.past_performance_table['as_of_date']) <= as_of_datetime)
        ].sort_values('as_of_date', ascending=False)
        
        if perf_rows.empty:
            result = {
                'past_3_avg_position': 9.0,
                'past_3_win_rate': 0.1,
                'past_5_avg_position': 9.0,
                'past_5_win_rate': 0.1,
            }
        else:
            latest = perf_rows.iloc[0]
            result = {
                'past_3_avg_position': latest.get('avg_position', 9.0),
                'past_3_win_rate': latest.get('win_rate', 0.1),
                'past_5_avg_position': latest.get('avg_position_5', 9.0),
                'past_5_win_rate': latest.get('win_rate_5', 0.1),
            }
        
        self._past_performance_cache[cache_key] = result
        return result
    
    def _is_available(
        self,
        feature_level: DataAvailability,
        max_level: DataAvailability
    ) -> bool:
        """特徴量が取得可能かチェック"""
        level_order = {
            DataAvailability.PRE_RACE: 0,
            DataAvailability.MORNING: 1,
            DataAvailability.PADDOCK: 2,
            DataAvailability.JUST_BEFORE: 3,
            DataAvailability.RESULT: 4
        }
        
        return level_order[feature_level] <= level_order[max_level]
    
    def _calculate_as_of_datetime(
        self,
        race_id: str,
        max_availability: DataAvailability
    ) -> datetime:
        """max_availabilityに応じた適切なas_of_datetimeを計算"""
        race_datetime = self.get_race_datetime(race_id)
        
        if max_availability == DataAvailability.PRE_RACE:
            return (race_datetime - timedelta(days=1)).replace(
                hour=self.cutoff_time.hour,
                minute=self.cutoff_time.minute,
                second=0,
                microsecond=0
            )
        
        elif max_availability == DataAvailability.MORNING:
            return race_datetime.replace(hour=9, minute=0, second=0, microsecond=0)
        
        elif max_availability == DataAvailability.PADDOCK:
            return race_datetime - timedelta(minutes=30)
        
        elif max_availability == DataAvailability.JUST_BEFORE:
            return race_datetime - timedelta(minutes=5)
        
        else:
            raise ValueError(f"RESULTレベルのデータは取得できません")
    
    def validate_no_leakage(
        self,
        feature_df: pd.DataFrame,
        target_df: pd.DataFrame,
        max_availability: DataAvailability = DataAvailability.PRE_RACE,
        show_samples: bool = True
    ) -> Tuple[bool, List[str]]:
        """データリークがないか検証"""
        
        issues = []
        
        for col in feature_df.columns:
            if col in ['index', 'level_0', 'race_id', 'horse_id']:
                continue
            
            feature_level = self._column_level_cache.get(
                col,
                self._infer_column_level(col)
            )
            
            if feature_level == DataAvailability.RESULT:
                issues.append(
                    f"❌ 致命的エラー: 列 '{col}' はRESULTレベルのデータです（使用禁止）"
                )
                if show_samples:
                    sample = feature_df[col].head(3).tolist()
                    issues.append(f"   サンプル値: {sample}")
            
            if not self._is_available(feature_level, max_availability):
                issues.append(
                    f"❌ データレベル違反: 列 '{col}' は {feature_level.value} レベルですが、"
                    f"max_availability は {max_availability.value} です"
                )
                if show_samples:
                    sample = feature_df[col].head(3).tolist()
                    issues.append(f"   サンプル値: {sample}")
        
        if not feature_df.index.equals(target_df.index):
            issues.append(
                f"❌ feature_dfとtarget_dfのインデックスが一致しません\n"
                f"   feature_df: {len(feature_df)}行, target_df: {len(target_df)}行"
            )
        
        nan_cols = feature_df.columns[feature_df.isna().any()].tolist()
        if nan_cols:
            issues.append(
                f"⚠️  警告: 以下の列にNaNがあります: {nan_cols[:5]}"
                f"{'...' if len(nan_cols) > 5 else ''}"
            )
        
        inf_cols = feature_df.columns[
            np.isinf(feature_df.select_dtypes(include=[np.number])).any()
        ].tolist()
        if inf_cols:
            issues.append(f"⚠️  警告: 以下の列に無限値があります: {inf_cols}")
        
        is_safe = len([i for i in issues if i.startswith('❌')]) == 0
        
        return is_safe, issues
    
    def validate_split_integrity(
        self,
        splits: List[DataSplit]
    ) -> Tuple[bool, List[str]]:
        """ウォークフォワードsplitの整合性を検証"""
        
        issues = []
        
        for i, split in enumerate(splits):
            if split.train_end >= split.test_start:
                issues.append(
                    f"❌ Fold {split.fold}: train_end ({split.train_end}) が "
                    f"test_start ({split.test_start}) 以降です"
                )
            
            if split.test_start >= split.test_end:
                issues.append(
                    f"❌ Fold {split.fold}: test期間が不正です "
                    f"({split.test_start} ~ {split.test_end})"
                )
            
            if len(split.train_indices) < 100:
                issues.append(
                    f"⚠️  警告: Fold {split.fold}の訓練データが少ないです "
                    f"({len(split.train_indices)}サンプル)"
                )
            
            if len(split.test_indices) < 10:
                issues.append(
                    f"⚠️  警告: Fold {split.fold}のテストデータが少ないです "
                    f"({len(split.test_indices)}サンプル)"
                )
            
            if i > 0:
                prev_split = splits[i - 1]
                if split.test_start < prev_split.test_end:
                    issues.append(
                        f"❌ Fold {split.fold}とFold {prev_split.fold}の "
                        f"テスト期間が重複しています"
                    )
        
        is_valid = len([i for i in issues if i.startswith('❌')]) == 0
        
        return is_valid, issues
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self._past_performance_cache.clear()
    
    def register_feature(
        self,
        feature_name: str,
        availability: DataAvailability
    ):
        """特徴量を登録"""
        FEATURE_AVAILABILITY[feature_name] = availability
        self._column_level_cache[feature_name] = availability
        print(f"✅ 特徴量 '{feature_name}' を {availability.value} レベルとして登録しました")
    
    def get_column_level(self, column: str) -> DataAvailability:
        """カラムのデータレベルを取得"""
        return self._column_level_cache.get(
            column,
            self._infer_column_level(column)
        )


def example_usage():
    """使用例（v5.1実運用版）"""
    
    print("=" * 80)
    print("TimelineManager v5.1 - 使用例（実運用完成版）")
    print("=" * 80)
    
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='W')
    
    data = []
    race_counter = 0
    
    for date in dates[:100]:
        for horse_num in range(1, 11):
            data.append({
                'race_id': f'race_{race_counter}',
                'race_date': date,
                'horse_id': f'horse_{np.random.randint(1, 50)}',
                'track_name': np.random.choice(['東京', '中山', '京都']),
                'distance': np.random.choice([1600, 1800, 2000]),
                'track_type': '芝',
                'horse_number': horse_num,
                'horse_age': np.random.randint(3, 8),
                'gate_number': horse_num,
                'jockey_id': f'jockey_{np.random.randint(1, 50)}',
                'trainer_id': f'trainer_{np.random.randint(1, 30)}',
            })
        race_counter += 1
    
    df = pd.DataFrame(data)
    
    past_perf_data = []
    for horse_id in df['horse_id'].unique()[:20]:
        for date in pd.date_range('2020-01-01', '2022-12-31', freq='M'):
            past_perf_data.append({
                'horse_id': horse_id,
                'as_of_date': date,
                'avg_position': np.random.uniform(5, 10),
                'win_rate': np.random.uniform(0.05, 0.2),
            })
    
    past_perf_df = pd.DataFrame(past_perf_data)
    
    print("\n初期化中...")
    tm = TimelineManager(
        df,
        past_performance_table=past_perf_df,
        date_column='race_date',
        cutoff_time=time(15, 0),
        tz='Asia/Tokyo',
        strict_mode=True,
        auto_infer_levels=True
    )
    
    print("✅ TimelineManager v5.1 初期化完了")
    print(f"   カラムレベルキャッシュ: {len(tm._column_level_cache)}個")
    
    print("\n" + "=" * 80)
    print("【1】カラムレベル自動推定（厳密化版）")
    print("=" * 80)
    
    sample_columns = ['track_name', 'distance', 'horse_age']
    for col in sample_columns:
        if col in tm.data.columns:
            level = tm.get_column_level(col)
            print(f"  {col:20s} → {level.value}")
    
    print("\n" + "=" * 80)
    print("【2】安全な特徴量取得")
    print("=" * 80)
    
    test_race_id = df['race_id'].iloc[0]
    test_horse_id = df['horse_id'].iloc[0]
    
    features = tm.get_safe_features(
        race_id=test_race_id,
        horse_id=test_horse_id,
        max_availability=DataAvailability.PRE_RACE
    )
    
    print(f"\n取得した特徴量: {len(features)}個")
    print(f"  - track_name: {features.get('track_name')}")
    print(f"  - distance: {features.get('distance')}")
    print(f"  - 過去3走平均: {features.get('past_3_avg_position', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("✅ v5.1完成 - 実運用レベル到達")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
