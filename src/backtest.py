"""
バックテスト

目的: 予測モデルのパフォーマンスを実戦形式で評価
重要: オッズタイミング別、控除率適用、包括的評価指標
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BacktestConfig:
    """バックテスト設定"""
    odds_timing: str  # '前日', '1時間前', '直前'
    takeout_rate: float  # 控除率（単勝: 0.25）
    strategies: List[str]  # ['all', 'ev>1.0', 'top1']
    bet_amount: float = 100  # 1レースあたりのベット額


class BacktestEngine:
    """
    バックテストエンジン
    
    機能:
    - オッズタイミング別評価
    - 控除率の適用
    - 複数戦略の比較
    - ドローダウン分析
    """
    
    def __init__(
        self,
        config: BacktestConfig
    ):
        """
        Args:
            config: バックテスト設定
        """
        self.config = config
        self.results = []
    
    def run(
        self,
        predictions: pd.DataFrame,
        actual_results: pd.DataFrame,
        odds_data: pd.DataFrame
    ) -> Dict:
        """
        バックテストを実行
        
        Args:
            predictions: 予測データ
                columns: ['race_id', 'horse_id', 'final_prob', ...]
            actual_results: 実際の結果
                columns: ['race_id', 'horse_id', 'finish_position']
            odds_data: オッズデータ
                columns: ['race_id', 'horse_id', 'odds_前日', 'odds_1時間前', 'odds_直前']
        
        Returns:
            バックテスト結果の辞書
        """
        
        # データ結合
        df = predictions.merge(
            actual_results,
            on=['race_id', 'horse_id'],
            how='inner'
        ).merge(
            odds_data,
            on=['race_id', 'horse_id'],
            how='inner'
        )
        
        # オッズ列の選択
        odds_col = f'odds_{self.config.odds_timing}'
        if odds_col not in df.columns:
            raise ValueError(f"{odds_col} がデータに存在しません")
        
        df['odds'] = df[odds_col]
        
        # 期待値計算（控除率考慮）
        df['ev'] = self._calculate_ev(df['final_prob'], df['odds'])
        
        # 各戦略でバックテスト
        strategy_results = {}
        
        for strategy in self.config.strategies:
            result = self._backtest_strategy(df, strategy)
            strategy_results[strategy] = result
        
        # 統合結果
        summary = {
            'config': self.config,
            'n_races': df['race_id'].nunique(),
            'n_bets_total': len(df),
            'strategy_results': strategy_results
        }
        
        return summary
    
    def _calculate_ev(
        self,
        prob: pd.Series,
        odds: pd.Series
    ) -> pd.Series:
        """
        期待値を計算（控除率考慮）
        
        EV = 勝率 × オッズ
        
        ただし、オッズは控除率が適用されているため、
        実際の期待値は EV - 控除率分 下がる
        """
        
        ev = prob * odds
        
        return ev
    
    def _backtest_strategy(
        self,
        df: pd.DataFrame,
        strategy: str
    ) -> Dict:
        """
        特定戦略でバックテスト
        
        Args:
            df: データ
            strategy: 戦略名
        
        Returns:
            戦略結果の辞書
        """
        
        # 戦略別のフィルタリング
        if strategy == 'all':
            # 全馬購入
            bet_df = df.copy()
        
        elif strategy == 'ev>1.0':
            # EV > 1.0 の馬のみ
            bet_df = df[df['ev'] > 1.0].copy()
        
        elif strategy == 'ev>1.2':
            # EV > 1.2 の馬のみ
            bet_df = df[df['ev'] > 1.2].copy()
        
        elif strategy == 'top1':
            # 各レースで予測1位のみ
            bet_df = df.loc[
                df.groupby('race_id')['final_prob'].idxmax()
            ].copy()
        
        elif strategy == 'top3':
            # 各レースで予測上位3頭
            bet_df = df.sort_values(
                ['race_id', 'final_prob'],
                ascending=[True, False]
            ).groupby('race_id').head(3)
        
        else:
            raise ValueError(f"未知の戦略: {strategy}")
        
        # 収支計算
        total_bet = len(bet_df) * self.config.bet_amount
        
        bet_df['win'] = (bet_df['finish_position'] == 1).astype(int)
        bet_df['return'] = bet_df['win'] * bet_df['odds'] * self.config.bet_amount
        
        total_return = bet_df['return'].sum()
        net_profit = total_return - total_bet
        roi = (net_profit / total_bet) if total_bet > 0 else 0
        
        # 的中率
        hit_rate = bet_df['win'].mean() if len(bet_df) > 0 else 0
        
        # 最大ドローダウン
        max_drawdown = self._calculate_max_drawdown(bet_df)
        
        # レース別収支
        race_profits = self._calculate_race_profits(bet_df)
        
        # 詳細統計
        result = {
            'n_bets': len(bet_df),
            'total_bet': total_bet,
            'total_return': total_return,
            'net_profit': net_profit,
            'roi': roi,
            'hit_rate': hit_rate,
            'max_drawdown': max_drawdown,
            'avg_odds': bet_df['odds'].mean() if len(bet_df) > 0 else 0,
            'avg_ev': bet_df['ev'].mean() if len(bet_df) > 0 else 0,
            'race_profits': race_profits
        }
        
        return result
    
    def _calculate_max_drawdown(
        self,
        bet_df: pd.DataFrame
    ) -> float:
        """
        最大ドローダウンを計算
        
        Returns:
            最大ドローダウン（円）
        """
        
        if len(bet_df) == 0:
            return 0
        
        # 時系列順にソート（race_idで）
        bet_df = bet_df.sort_values('race_id')
        
        # 累積損益
        bet_df['profit'] = bet_df['return'] - self.config.bet_amount
        cumulative_profit = bet_df['profit'].cumsum()
        
        # 最大ドローダウン
        peak = cumulative_profit.expanding().max()
        drawdown = peak - cumulative_profit
        max_dd = drawdown.max()
        
        return max_dd
    
    def _calculate_race_profits(
        self,
        bet_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        レース別の収支を計算
        
        Returns:
            レース別収支のDataFrame
        """
        
        race_summary = bet_df.groupby('race_id').agg({
            'odds': 'mean',
            'win': 'sum',
            'return': 'sum'
        }).reset_index()
        
        race_summary['bet'] = bet_df.groupby('race_id').size().values * self.config.bet_amount
        race_summary['profit'] = race_summary['return'] - race_summary['bet']
        
        return race_summary


class BacktestVisualizer:
    """
    バックテスト結果の可視化
    """
    
    @staticmethod
    def print_summary(
        summary: Dict
    ):
        """
        サマリーを出力
        """
        
        config = summary['config']
        
        print("=" * 80)
        print(f"バックテスト結果")
        print("=" * 80)
        print(f"オッズタイミング: {config.odds_timing}")
        print(f"控除率: {config.takeout_rate * 100:.1f}%")
        print(f"レース数: {summary['n_races']:,}")
        print("-" * 80)
        
        for strategy, result in summary['strategy_results'].items():
            print(f"\n【戦略: {strategy}】")
            print(f"  購入数: {result['n_bets']:,} 点")
            print(f"  総投資額: ¥{result['total_bet']:,.0f}")
            print(f"  総回収額: ¥{result['total_return']:,.0f}")
            print(f"  純損益: ¥{result['net_profit']:,.0f}")
            print(f"  回収率: {result['roi'] * 100:.2f}%")
            print(f"  的中率: {result['hit_rate'] * 100:.2f}%")
            print(f"  平均オッズ: {result['avg_odds']:.2f}")
            print(f"  平均EV: {result['avg_ev']:.3f}")
            print(f"  最大ドローダウン: ¥{result['max_drawdown']:,.0f}")
    
    @staticmethod
    def plot_cumulative_profit(
        race_profits: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        累積損益をプロット
        """
        
        import matplotlib.pyplot as plt
        
        cumulative = race_profits['profit'].cumsum()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cumulative.values, linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Race', fontsize=12)
        ax.set_ylabel('Cumulative Profit (¥)', fontsize=12)
        ax.set_title('Cumulative Profit', fontsize=14)
        ax.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def example_usage():
    """使用例"""
    
    # ダミーデータ
    np.random.seed(42)
    
    n_races = 100
    n_horses_per_race = 18
    
    predictions = []
    actual_results = []
    odds_data = []
    
    for race_id in range(n_races):
        for horse_id in range(n_horses_per_race):
            # 予測
            predictions.append({
                'race_id': f'race_{race_id}',
                'horse_id': f'horse_{horse_id}',
                'final_prob': np.random.beta(2, 15)
            })
            
            # 実際の結果
            actual_results.append({
                'race_id': f'race_{race_id}',
                'horse_id': f'horse_{horse_id}',
                'finish_position': horse_id + 1
            })
            
            # オッズ
            odds_data.append({
                'race_id': f'race_{race_id}',
                'horse_id': f'horse_{horse_id}',
                'odds_前日': np.random.uniform(2, 50),
                'odds_1時間前': np.random.uniform(2, 50),
                'odds_直前': np.random.uniform(2, 50)
            })
    
    predictions_df = pd.DataFrame(predictions)
    actual_df = pd.DataFrame(actual_results)
    odds_df = pd.DataFrame(odds_data)
    
    # バックテスト設定
    config = BacktestConfig(
        odds_timing='前日',
        takeout_rate=0.25,
        strategies=['all', 'ev>1.0', 'ev>1.2', 'top1']
    )
    
    # バックテスト実行
    engine = BacktestEngine(config)
    summary = engine.run(predictions_df, actual_df, odds_df)
    
    # 結果出力
    BacktestVisualizer.print_summary(summary)


if __name__ == "__main__":
    example_usage()
