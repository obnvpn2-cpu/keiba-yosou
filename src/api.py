"""
FastAPI - 予測API

エンドポイント:
- POST /predict: レース予測
- GET /explain/{race_id}/{horse_id}: SHAP説明
- GET /backtest: バックテスト結果
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime

# 自作モジュール（実際の実装で使用）
# from base_model import BaseModel
# from calibration import CalibrationPipeline
# from baba_adjustment import BabaAdjustmentModel
# from pace_adjustment import PaceAdjustmentModel
# from probability_integration import ProbabilityIntegrator
# from shap_explainer import SHAPExplainer


app = FastAPI(
    title="KEIBA SCENARIO AI",
    description="競馬予想AIのAPI",
    version="1.0.0"
)

# CORS設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# リクエスト/レスポンスモデル
class RaceRequest(BaseModel):
    race_id: str
    include_shap: bool = False


class HorsePrediction(BaseModel):
    horse_id: str
    horse_name: str
    horse_number: int
    base_pred: float
    calibrated_pred: float
    delta_baba: float
    delta_pace: float
    final_prob: float
    ev: Optional[float] = None
    odds: Optional[float] = None


class RaceResponse(BaseModel):
    race_id: str
    race_date: str
    track_name: str
    distance: int
    predictions: List[HorsePrediction]
    race_sum_prob: float
    top_3_horses: List[str]


class SHAPExplanation(BaseModel):
    horse_id: str
    prediction: float
    base_value: float
    top_features: List[Dict]
    text_explanation: str


# グローバル変数（モデルとデータ）
# 実際の実装では起動時にロード
MODEL_CACHE = {}
DATA_CACHE = {}


@app.on_event("startup")
async def startup_event():
    """
    アプリ起動時にモデルをロード
    """
    print("モデルをロード中...")
    
    # 実際の実装例:
    # MODEL_CACHE['base_model'] = load_base_model()
    # MODEL_CACHE['calibrator'] = load_calibrator()
    # MODEL_CACHE['baba_model'] = load_baba_model()
    # MODEL_CACHE['pace_model'] = load_pace_model()
    # MODEL_CACHE['integrator'] = ProbabilityIntegrator()
    # MODEL_CACHE['shap_explainer'] = load_shap_explainer()
    
    print("モデルロード完了")


@app.get("/")
async def root():
    """
    ヘルスチェック
    """
    return {
        "status": "ok",
        "message": "KEIBA SCENARIO AI is running",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=RaceResponse)
async def predict_race(request: RaceRequest):
    """
    レース予測
    
    Args:
        request: レースリクエスト
    
    Returns:
        予測結果
    """
    
    race_id = request.race_id
    
    # レースデータを取得（実際はDBから）
    # race_data = get_race_data(race_id)
    # if race_data is None:
    #     raise HTTPException(status_code=404, detail="Race not found")
    
    # ダミーデータ（実装例）
    race_data = {
        'race_id': race_id,
        'race_date': '2024-12-01',
        'track_name': '東京',
        'distance': 1600
    }
    
    # 予測実行（実際はモデルを使用）
    # predictions = run_prediction_pipeline(race_data)
    
    # ダミー予測
    n_horses = 18
    predictions = []
    
    for i in range(1, n_horses + 1):
        horse_pred = HorsePrediction(
            horse_id=f"horse_{i}",
            horse_name=f"ダミー馬{i}",
            horse_number=i,
            base_pred=np.random.beta(2, 15),
            calibrated_pred=np.random.beta(2, 15),
            delta_baba=np.random.normal(0, 0.3),
            delta_pace=np.random.normal(0, 0.3),
            final_prob=np.random.beta(2, 15),
            ev=np.random.uniform(0.5, 1.5),
            odds=np.random.uniform(3, 30)
        )
        predictions.append(horse_pred)
    
    # 確率の正規化
    total_prob = sum([p.final_prob for p in predictions])
    for p in predictions:
        p.final_prob = p.final_prob / total_prob
    
    # 上位3頭
    sorted_preds = sorted(predictions, key=lambda x: x.final_prob, reverse=True)
    top_3 = [p.horse_name for p in sorted_preds[:3]]
    
    response = RaceResponse(
        race_id=race_id,
        race_date=race_data['race_date'],
        track_name=race_data['track_name'],
        distance=race_data['distance'],
        predictions=predictions,
        race_sum_prob=sum([p.final_prob for p in predictions]),
        top_3_horses=top_3
    )
    
    return response


@app.get("/explain/{race_id}/{horse_id}", response_model=SHAPExplanation)
async def explain_prediction(race_id: str, horse_id: str):
    """
    SHAP説明を取得
    
    Args:
        race_id: レースID
        horse_id: 馬ID
    
    Returns:
        SHAP説明
    """
    
    # 実際の実装:
    # shap_explainer = MODEL_CACHE['shap_explainer']
    # explanation = shap_explainer.explain_prediction(...)
    
    # ダミー説明
    explanation = SHAPExplanation(
        horse_id=horse_id,
        prediction=0.142,
        base_value=0.055,
        top_features=[
            {'feature': '騎手勝率', 'shap_value': 0.032, 'impact': '+3.2%'},
            {'feature': '同距離勝率', 'shap_value': 0.028, 'impact': '+2.8%'},
            {'feature': '馬体重', 'shap_value': 0.015, 'impact': '+1.5%'},
            {'feature': '枠番', 'shap_value': -0.021, 'impact': '-2.1%'},
            {'feature': 'レース間隔', 'shap_value': -0.018, 'impact': '-1.8%'}
        ],
        text_explanation="""予測勝率: 14.2%

主な要因:
  +3.2% - 騎手勝率（プラス）
  +2.8% - 同距離勝率（プラス）
  +1.5% - 馬体重（プラス）
  -2.1% - 枠番（マイナス）
  -1.8% - レース間隔（マイナス）"""
    )
    
    return explanation


@app.get("/backtest")
async def get_backtest_results(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    strategy: str = "ev>1.0"
):
    """
    バックテスト結果を取得
    
    Args:
        start_date: 開始日
        end_date: 終了日
        strategy: 戦略
    
    Returns:
        バックテスト結果
    """
    
    # 実際の実装:
    # backtest_results = load_backtest_results(start_date, end_date, strategy)
    
    # ダミー結果
    results = {
        'period': {
            'start_date': start_date or '2023-01-01',
            'end_date': end_date or '2024-12-31'
        },
        'strategy': strategy,
        'metrics': {
            'n_races': 500,
            'n_bets': 1250,
            'total_bet': 125000,
            'total_return': 142500,
            'net_profit': 17500,
            'roi': 0.14,
            'hit_rate': 0.184,
            'avg_odds': 8.5,
            'max_drawdown': 32000
        }
    }
    
    return results


@app.get("/races/today")
async def get_today_races():
    """
    今日のレース一覧を取得
    """
    
    # 実際の実装:
    # races = get_races_by_date(datetime.now().date())
    
    # ダミーデータ
    races = [
        {
            'race_id': 'race_001',
            'race_number': '10R',
            'track_name': '東京',
            'race_name': 'サンプルS',
            'distance': 1600,
            'track_type': '芝',
            'time': '15:30'
        },
        {
            'race_id': 'race_002',
            'race_number': '11R',
            'track_name': '中山',
            'race_name': 'テストH',
            'distance': 1800,
            'track_type': '芝',
            'time': '16:00'
        }
    ]
    
    return {'races': races}


# 開発用設定
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
