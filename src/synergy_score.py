"""
相性スコア（Synergy Score）

目的: 騎手×調教師、騎手×馬などの隠れ相性を発見
方式: SVD + Bayesian Shrinkage
"""

import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from typing import Dict, Tuple, Optional
from collections import defaultdict


class SynergyScorer:
    """
    相性スコア計算
    
    アプローチ:
    1. SVDで隠れ相性を学習
    2. データ量に応じたBayesian Shrinkage
    """
    
    def __init__(
        self,
        n_factors: int = 20,
        n_epochs: int = 50,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        prior_count: int = 10
    ):
        """
        Args:
            n_factors: SVDの因子数
            n_epochs: エポック数
            lr_all: 学習率
            reg_all: 正則化係数
            prior_count: Shrinkageの事前カウント
        """
        
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.prior_count = prior_count
        
        self.algo = None
        self.global_mean = None
        self.combination_counts = defaultdict(int)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        user_col: str = 'jockey_id',
        item_col: str = 'trainer_id',
        rating_col: str = 'finish_position'
    ) -> Dataset:
        """
        SVD用のデータを準備
        
        Args:
            df: データフレーム
            user_col: ユーザー列（例: 騎手ID）
            item_col: アイテム列（例: 調教師ID）
            rating_col: 評価列（例: 着順）
        
        Returns:
            Surpriseのデータセット
        """
        
        # 着順を5段階評価に変換
        # 1位=5, 2位=4, 3位=3, 4-6位=2, 7位以下=1
        def position_to_rating(pos):
            if pos == 1:
                return 5
            elif pos == 2:
                return 4
            elif pos == 3:
                return 3
            elif pos <= 6:
                return 2
            else:
                return 1
        
        df = df.copy()
        df['rating'] = df[rating_col].apply(position_to_rating)
        
        # 組み合わせごとのカウント
        for _, row in df.iterrows():
            key = (row[user_col], row[item_col])
            self.combination_counts[key] += 1
        
        # 全体平均
        self.global_mean = df['rating'].mean()
        
        # Surpriseのデータセット作成
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            df[[user_col, item_col, 'rating']],
            reader
        )
        
        return data
    
    def train(
        self,
        dataset: Dataset
    ):
        """
        SVDモデルを訓練
        
        Args:
            dataset: Surpriseのデータセット
        """
        
        self.algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=42
        )
        
        trainset = dataset.build_full_trainset()
        self.algo.fit(trainset)
        
        print(f"\nSVDモデル訓練完了")
        print(f"  因子数: {self.n_factors}")
        print(f"  エポック数: {self.n_epochs}")
    
    def predict_with_shrinkage(
        self,
        user_id: str,
        item_id: str
    ) -> float:
        """
        Shrinkage付きで相性スコアを予測
        
        Args:
            user_id: ユーザーID（例: 騎手ID）
            item_id: アイテムID（例: 調教師ID）
        
        Returns:
            相性スコア（1~5）
        """
        
        if self.algo is None:
            raise ValueError("モデルが訓練されていません")
        
        # SVDによる予測
        svd_prediction = self.algo.predict(user_id, item_id).est
        
        # データ量
        key = (user_id, item_id)
        n_observations = self.combination_counts.get(key, 0)
        
        # Shrinkage係数
        alpha = self._calculate_shrinkage_factor(n_observations)
        
        # Shrinkage適用
        shrunk_prediction = alpha * self.global_mean + (1 - alpha) * svd_prediction
        
        return shrunk_prediction
    
    def _calculate_shrinkage_factor(
        self,
        n_observations: int
    ) -> float:
        """
        Shrinkage係数を計算
        
        α = prior_count / (prior_count + n_observations)
        
        n=0: α=1（完全に全体平均）
        n=prior_count: α=0.5（半々）
        n→∞: α→0（データのみ）
        """
        
        alpha = self.prior_count / (self.prior_count + n_observations)
        return alpha
    
    def batch_predict(
        self,
        df: pd.DataFrame,
        user_col: str = 'jockey_id',
        item_col: str = 'trainer_id'
    ) -> np.ndarray:
        """
        バッチ予測
        
        Args:
            df: データフレーム
            user_col: ユーザー列
            item_col: アイテム列
        
        Returns:
            相性スコアの配列
        """
        
        scores = []
        
        for _, row in df.iterrows():
            score = self.predict_with_shrinkage(
                row[user_col],
                row[item_col]
            )
            scores.append(score)
        
        return np.array(scores)
    
    def get_top_combinations(
        self,
        user_id: str,
        item_ids: list,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        指定ユーザーの最高相性アイテムを取得
        
        Args:
            user_id: ユーザーID
            item_ids: アイテムIDのリスト
            top_n: 上位N個
        
        Returns:
            上位組み合わせのDataFrame
        """
        
        results = []
        
        for item_id in item_ids:
            score = self.predict_with_shrinkage(user_id, item_id)
            key = (user_id, item_id)
            count = self.combination_counts.get(key, 0)
            
            results.append({
                'user_id': user_id,
                'item_id': item_id,
                'synergy_score': score,
                'n_observations': count
            })
        
        df = pd.DataFrame(results).sort_values(
            'synergy_score',
            ascending=False
        ).head(top_n)
        
        return df


class MultiSynergyManager:
    """
    複数の相性スコアを管理
    
    例:
    - 騎手×調教師
    - 騎手×馬
    - 調教師×馬
    """
    
    def __init__(self):
        self.scorers = {}
    
    def add_scorer(
        self,
        name: str,
        scorer: SynergyScorer
    ):
        """
        相性スコア計算器を追加
        
        Args:
            name: 名前（例: 'jockey_trainer'）
            scorer: SynergyScorerインスタンス
        """
        self.scorers[name] = scorer
    
    def train_all(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ):
        """
        全スコア計算器を訓練
        
        Args:
            data_dict: {name: データフレーム} の辞書
        """
        
        for name, scorer in self.scorers.items():
            if name not in data_dict:
                print(f"警告: {name} のデータがありません")
                continue
            
            print(f"\n=== {name} を訓練中 ===")
            
            df = data_dict[name]
            
            # データセット準備
            if name == 'jockey_trainer':
                dataset = scorer.prepare_data(
                    df,
                    user_col='jockey_id',
                    item_col='trainer_id'
                )
            elif name == 'jockey_horse':
                dataset = scorer.prepare_data(
                    df,
                    user_col='jockey_id',
                    item_col='horse_id'
                )
            elif name == 'trainer_horse':
                dataset = scorer.prepare_data(
                    df,
                    user_col='trainer_id',
                    item_col='horse_id'
                )
            
            # 訓練
            scorer.train(dataset)
    
    def predict_all(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        全相性スコアを予測
        
        Args:
            df: データフレーム（必要な列を含む）
        
        Returns:
            相性スコア列が追加されたDataFrame
        """
        
        result = df.copy()
        
        if 'jockey_trainer' in self.scorers:
            result['synergy_jockey_trainer'] = self.scorers['jockey_trainer'].batch_predict(
                df,
                user_col='jockey_id',
                item_col='trainer_id'
            )
        
        if 'jockey_horse' in self.scorers:
            result['synergy_jockey_horse'] = self.scorers['jockey_horse'].batch_predict(
                df,
                user_col='jockey_id',
                item_col='horse_id'
            )
        
        if 'trainer_horse' in self.scorers:
            result['synergy_trainer_horse'] = self.scorers['trainer_horse'].batch_predict(
                df,
                user_col='trainer_id',
                item_col='horse_id'
            )
        
        return result


def example_usage():
    """使用例"""
    
    # ダミーデータ
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'race_id': [f'race_{i // 18}' for i in range(n)],
        'jockey_id': [f'jockey_{np.random.randint(1, 50)}' for _ in range(n)],
        'trainer_id': [f'trainer_{np.random.randint(1, 30)}' for _ in range(n)],
        'horse_id': [f'horse_{np.random.randint(1, 500)}' for _ in range(n)],
        'finish_position': np.random.randint(1, 19, n)
    })
    
    # 騎手×調教師の相性
    scorer = SynergyScorer(
        n_factors=20,
        n_epochs=50,
        prior_count=10
    )
    
    # データ準備
    dataset = scorer.prepare_data(
        df,
        user_col='jockey_id',
        item_col='trainer_id',
        rating_col='finish_position'
    )
    
    # 訓練
    scorer.train(dataset)
    
    # 予測
    test_jockey = 'jockey_1'
    test_trainer = 'trainer_1'
    
    score = scorer.predict_with_shrinkage(test_jockey, test_trainer)
    print(f"\n=== 相性スコア ===")
    print(f"{test_jockey} × {test_trainer}: {score:.3f}")
    
    # 上位組み合わせ
    all_trainers = [f'trainer_{i}' for i in range(1, 31)]
    top_combos = scorer.get_top_combinations(
        test_jockey,
        all_trainers,
        top_n=5
    )
    
    print(f"\n=== {test_jockey} の最高相性調教師 ===")
    print(top_combos)
    
    # バッチ予測
    test_df = df.head(10)
    scores = scorer.batch_predict(
        test_df,
        user_col='jockey_id',
        item_col='trainer_id'
    )
    
    print("\n=== バッチ予測 ===")
    test_df['synergy_score'] = scores
    print(test_df[['jockey_id', 'trainer_id', 'synergy_score']])


if __name__ == "__main__":
    example_usage()
