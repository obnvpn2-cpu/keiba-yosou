import React, { useState, useEffect, useCallback } from "react";
import type { ScenarioUiContext, SortType } from "./types";
import RaceHeader from "./components/RaceHeader";
import ScenarioControlBar from "./components/ScenarioControlBar";
import ScenarioSummary from "./components/ScenarioSummary";
import HorseList from "./components/HorseList";

/**
 * 利用可能なシナリオJSONファイルのリスト
 * public/ フォルダに配置されている想定
 */
const AVAILABLE_SCENARIOS = [
  { id: "scenario_slow_inner", label: "スロー × 内伸び", file: "scenario_slow_inner.json" },
  { id: "scenario_middle_flat", label: "ミドル × フラット", file: "scenario_middle_flat.json" },
  { id: "scenario_high_outer", label: "ハイ × 外伸び", file: "scenario_high_outer.json" },
];

/**
 * JSONファイルをfetchする関数
 */
async function fetchScenarioData(filename: string): Promise<ScenarioUiContext> {
  const response = await fetch(`/${filename}`);
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  const data = await response.json();
  return data as ScenarioUiContext;
}

/**
 * メインアプリケーションコンポーネント
 */
function ScenarioApp(): React.ReactElement {
  // 状態管理
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ScenarioUiContext | null>(null);
  const [selectedScenario, setSelectedScenario] = useState<string>(AVAILABLE_SCENARIOS[0].id);
  const [sortBy, setSortBy] = useState<SortType>("adj_win");

  /**
   * シナリオデータを読み込む
   */
  const loadScenario = useCallback(async (scenarioId: string) => {
    const scenario = AVAILABLE_SCENARIOS.find((s) => s.id === scenarioId);
    
    if (!scenario) {
      setError(`シナリオが見つかりません: ${scenarioId}`);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await fetchScenarioData(scenario.file);
      setData(result);
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "データの読み込みに失敗しました";
      setError(message);
      setData(null);
      console.error("Fetch error:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  // 初回読み込み
  useEffect(() => {
    loadScenario(selectedScenario);
  }, [selectedScenario, loadScenario]);

  /**
   * シナリオ切り替えハンドラ
   */
  const handleScenarioChange = useCallback((newScenarioId: string) => {
    setSelectedScenario(newScenarioId);
  }, []);

  /**
   * ソート切り替えハンドラ
   */
  const handleSortChange = useCallback((newSortBy: SortType) => {
    setSortBy(newSortBy);
  }, []);

  // ローディング表示
  if (loading) {
    return (
      <div className="app-container">
        <div className="loading-container">
          <div className="loading-spinner" />
          <p className="loading-text">シナリオデータを読み込み中...</p>
        </div>
      </div>
    );
  }

  // エラー表示
  if (error) {
    return (
      <div className="app-container">
        <div className="error-container">
          <h2 className="error-title">エラーが発生しました</h2>
          <p className="error-message">{error}</p>
          <button
            className="retry-button"
            onClick={() => loadScenario(selectedScenario)}
          >
            再読み込み
          </button>
        </div>
        <ScenarioControlBar
          scenarios={AVAILABLE_SCENARIOS}
          selectedScenario={selectedScenario}
          onScenarioChange={handleScenarioChange}
          sortBy={sortBy}
          onSortChange={handleSortChange}
        />
      </div>
    );
  }

  // データがない場合
  if (!data) {
    return (
      <div className="app-container">
        <div className="empty-container">
          <p>データがありません</p>
        </div>
        <ScenarioControlBar
          scenarios={AVAILABLE_SCENARIOS}
          selectedScenario={selectedScenario}
          onScenarioChange={handleScenarioChange}
          sortBy={sortBy}
          onSortChange={handleSortChange}
        />
      </div>
    );
  }

  // 正常表示
  return (
    <div className="app-container">
      <RaceHeader race={data.race} scenario={data.scenario} />
      
      <ScenarioControlBar
        scenarios={AVAILABLE_SCENARIOS}
        selectedScenario={selectedScenario}
        onScenarioChange={handleScenarioChange}
        sortBy={sortBy}
        onSortChange={handleSortChange}
      />
      
      <main className="main-layout">
        <section className="horses-section">
          <HorseList horses={data.horses} sortBy={sortBy} />
        </section>
        
        <aside className="summary-section">
          <ScenarioSummary summary={data.summary} scenario={data.scenario} />
        </aside>
      </main>
    </div>
  );
}

export default ScenarioApp;
