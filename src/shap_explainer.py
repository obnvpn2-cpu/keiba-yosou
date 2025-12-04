"""
SHAP解析 + 説明文生成

目的: 予測の理由を説明可能にする
機能: SHAP values計算、重要特徴抽出、テキスト説明生成
"""

import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    SHAP解析器
    
    機能:
    - SHAP values計算
    - 特徴量の寄与度分析
    - 個別予測の説明
    """
    
    def __init__(
        self,
        model,
        feature_names: List[str]
    ):
        """
        Args:
            model: LightGBMモデル
            feature_names: 特徴量名のリスト
        """
        
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.base_value = None
    
    def initialize(
        self,
        X_background: pd.DataFrame
    ):
        """
        Explainerを初期化
        
        Args:
            X_background: 背景データ（サンプリング推奨）
        """
        
        # TreeExplainerを使用（LightGBM高速）
        self.explainer = shap.TreeExplainer(self.model)
        
        # Base value（全体平均の予測）
        self.base_value = self.explainer.expected_value
        
        print(f"SHAP Explainer初期化完了")
        print(f"  Base value: {self.base_value:.4f}")
    
    def explain(
        self,
        X: pd.DataFrame,
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        SHAP valuesを計算
        
        Args:
            X: 特徴量
            check_additivity: 加法性をチェックするか
        
        Returns:
            SHAP values (n_samples, n_features)
        """
        
        if self.explainer is None:
            raise ValueError("initialize()を先に実行してください")
        
        shap_values = self.explainer.shap_values(
            X,
            check_additivity=check_additivity
        )
        
        return shap_values
    
    def get_top_features(
        self,
        shap_values: np.ndarray,
        top_n: int = 5,
        abs_values: bool = True
    ) -> pd.DataFrame:
        """
        重要特徴量のトップNを取得
        
        Args:
            shap_values: SHAP values (1次元配列を想定)
            top_n: 上位N個
            abs_values: 絶対値でソートするか
        
        Returns:
            重要特徴量のDataFrame
        """
        
        if abs_values:
            sorted_idx = np.argsort(np.abs(shap_values))[::-1]
        else:
            sorted_idx = np.argsort(shap_values)[::-1]
        
        top_features = []
        
        for i in range(min(top_n, len(sorted_idx))):
            idx = sorted_idx[i]
            top_features.append({
                'feature': self.feature_names[idx],
                'shap_value': shap_values[idx],
                'abs_shap_value': abs(shap_values[idx])
            })
        
        return pd.DataFrame(top_features)
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        horse_idx: int = 0,
        top_n: int = 5
    ) -> Dict:
        """
        個別予測を説明
        
        Args:
            X: 特徴量
            horse_idx: 馬のインデックス
            top_n: 表示する特徴量数
        
        Returns:
            説明の辞書
        """
        
        # SHAP values計算
        shap_values = self.explain(X)
        
        # 対象馬のSHAP values
        horse_shap = shap_values[horse_idx]
        
        # トップ特徴量
        top_features = self.get_top_features(horse_shap, top_n=top_n)
        
        # 予測値
        prediction = self.base_value + np.sum(horse_shap)
        
        explanation = {
            'horse_idx': horse_idx,
            'base_value': self.base_value,
            'prediction': prediction,
            'top_features': top_features,
            'shap_values': horse_shap
        }
        
        return explanation


class ExplanationGenerator:
    """
    説明文生成器
    
    機能:
    - SHAP結果をテキストに変換
    - ルールベース + テンプレート
    """
    
    def __init__(self):
        # 特徴量名の日本語マッピング
        self.feature_name_mapping = {
            'jockey_win_rate': '騎手勝率',
            'trainer_win_rate': '調教師勝率',
            'same_distance_win_rate': '同距離勝率',
            'same_track_win_rate': '同競馬場勝率',
            'past_3_avg_position': '過去3走平均着順',
            'horse_age': '馬齢',
            'weight_change': '馬体重増減',
            'gate_number': '枠番',
            'horse_number': '馬番',
            'predicted_baba_index': '予測馬場指数',
            'predicted_pace_deviation': '予測ペース偏差',
            'running_style': '脚質',
            'synergy_jockey_trainer': '騎手×調教師相性',
            # ... 他の特徴量も追加
        }
    
    def generate_text_explanation(
        self,
        explanation: Dict,
        feature_values: Optional[pd.Series] = None
    ) -> str:
        """
        テキスト説明を生成
        
        Args:
            explanation: SHAPExplainerからの説明
            feature_values: 特徴量の実際の値
        
        Returns:
            説明テキスト
        """
        
        base_value = explanation['base_value']
        prediction = explanation['prediction']
        top_features = explanation['top_features']
        
        # ベース確率をパーセントに変換
        base_pct = base_value * 100
        pred_pct = prediction * 100
        
        text = f"予測勝率: {pred_pct:.1f}%\n"
        text += f"（全体平均: {base_pct:.1f}%）\n\n"
        text += "主な要因:\n"
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            shap_value = row['shap_value']
            
            # 日本語名に変換
            feature_jp = self.feature_name_mapping.get(feature, feature)
            
            # 寄与の方向
            if shap_value > 0:
                direction = "プラス"
                symbol = "+"
            else:
                direction = "マイナス"
                symbol = ""
            
            # 影響度（確率への寄与）
            impact_pct = shap_value * 100
            
            text += f"  {symbol}{impact_pct:+.1f}% - {feature_jp}（{direction}）\n"
            
            # 特徴量の値も表示
            if feature_values is not None and feature in feature_values.index:
                value = feature_values[feature]
                text += f"    （値: {value}）\n"
        
        return text
    
    def generate_natural_explanation(
        self,
        explanation: Dict,
        feature_values: Optional[pd.Series] = None
    ) -> str:
        """
        より自然な説明文を生成
        
        Args:
            explanation: SHAPExplainerからの説明
            feature_values: 特徴量の実際の値
        
        Returns:
            自然言語の説明テキスト
        """
        
        top_features = explanation['top_features']
        pred_pct = explanation['prediction'] * 100
        
        # 正の寄与と負の寄与に分ける
        positive = top_features[top_features['shap_value'] > 0]
        negative = top_features[top_features['shap_value'] < 0]
        
        text = f"この馬の勝率は{pred_pct:.1f}%と予測されます。\n\n"
        
        # 正の要因
        if len(positive) > 0:
            text += "有利な要因:\n"
            for _, row in positive.iterrows():
                feature = row['feature']
                feature_jp = self.feature_name_mapping.get(feature, feature)
                impact = abs(row['shap_value']) * 100
                
                text += f"- {feature_jp}が勝率を{impact:.1f}%押し上げています\n"
        
        # 負の要因
        if len(negative) > 0:
            text += "\n不利な要因:\n"
            for _, row in negative.iterrows():
                feature = row['feature']
                feature_jp = self.feature_name_mapping.get(feature, feature)
                impact = abs(row['shap_value']) * 100
                
                text += f"- {feature_jp}が勝率を{impact:.1f}%押し下げています\n"
        
        return text
    
    def generate_for_llm(
        self,
        explanation: Dict,
        feature_values: Optional[pd.Series] = None
    ) -> str:
        """
        LLMに渡すためのプロンプトを生成
        
        Args:
            explanation: SHAPExplainerからの説明
            feature_values: 特徴量の実際の値
        
        Returns:
            LLM用プロンプト
        """
        
        # 構造化されたデータ
        structured_data = {
            'predicted_win_rate': f"{explanation['prediction'] * 100:.1f}%",
            'base_win_rate': f"{explanation['base_value'] * 100:.1f}%",
            'top_factors': []
        }
        
        for _, row in explanation['top_features'].iterrows():
            feature = row['feature']
            feature_jp = self.feature_name_mapping.get(feature, feature)
            impact = row['shap_value'] * 100
            
            factor = {
                'feature': feature_jp,
                'impact': f"{impact:+.1f}%",
                'direction': 'positive' if impact > 0 else 'negative'
            }
            
            # 特徴量の値
            if feature_values is not None and feature in feature_values.index:
                factor['value'] = str(feature_values[feature])
            
            structured_data['top_factors'].append(factor)
        
        # プロンプト作成
        prompt = f"""以下の競馬予想AIの分析結果を、競馬ファンに分かりやすく自然な日本語で説明してください。

予測勝率: {structured_data['predicted_win_rate']}
全体平均: {structured_data['base_win_rate']}

主な要因:
"""
        
        for factor in structured_data['top_factors']:
            prompt += f"- {factor['feature']}: {factor['impact']}\n"
        
        prompt += """
注意事項:
- 断定的な表現は避け、「〜の傾向があります」「〜が予想されます」などの表現を使う
- 競馬用語を適切に使う
- 簡潔に2-3文でまとめる
"""
        
        return prompt


def example_usage():
    """使用例"""
    
    import lightgbm as lgb
    
    # ダミーデータ
    np.random.seed(42)
    n = 1000
    
    X = pd.DataFrame({
        'jockey_win_rate': np.random.beta(2, 8, n),
        'trainer_win_rate': np.random.beta(2, 8, n),
        'same_distance_win_rate': np.random.beta(2, 8, n),
        'past_3_avg_position': np.random.uniform(1, 10, n),
        'horse_age': np.random.randint(3, 8, n),
        'gate_number': np.random.randint(1, 9, n)
    })
    
    y = np.random.binomial(1, 0.1, n)
    
    # モデル訓練
    model = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    model.fit(X, y)
    
    # SHAP Explainer
    explainer = SHAPExplainer(
        model.booster_,
        feature_names=X.columns.tolist()
    )
    explainer.initialize(X[:100])
    
    # 説明
    explanation = explainer.explain_prediction(X, horse_idx=0, top_n=5)
    
    # テキスト生成
    generator = ExplanationGenerator()
    
    print("=== SHAP解析結果 ===")
    print(generator.generate_text_explanation(
        explanation,
        feature_values=X.iloc[0]
    ))
    
    print("\n=== 自然な説明 ===")
    print(generator.generate_natural_explanation(
        explanation,
        feature_values=X.iloc[0]
    ))
    
    print("\n=== LLM用プロンプト ===")
    print(generator.generate_for_llm(
        explanation,
        feature_values=X.iloc[0]
    ))


if __name__ == "__main__":
    example_usage()
