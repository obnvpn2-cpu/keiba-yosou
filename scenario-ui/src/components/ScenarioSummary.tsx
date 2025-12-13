import React from "react";
import type { SummaryData, ScenarioData } from "../types";

interface ScenarioSummaryProps {
  summary?: SummaryData;
  scenario: ScenarioData;
}

/**
 * ペースラベル
 */
function getPaceDescription(pace: string): string {
  const descriptions: Record<string, string> = {
    S: "スローペース想定 - 逃げ・先行馬有利",
    M: "ミドルペース想定 - 展開フラット",
    H: "ハイペース想定 - 差し・追込馬有利",
  };
  return descriptions[pace] || `ペース: ${pace}`;
}

/**
 * シナリオサマリーコンポーネント
 */
function ScenarioSummary({
  summary,
  scenario,
}: ScenarioSummaryProps): React.ReactElement {
  return (
    <div className="scenario-summary">
      <h2 className="summary-title">シナリオ分析</h2>

      {/* ペース分析 */}
      <section className="summary-section">
        <h3 className="summary-section-title">展開予想</h3>
        <p className="pace-description">{getPaceDescription(scenario.pace)}</p>
        
        <div className="scenario-stats">
          {scenario.front_runner_count !== undefined && (
            <div className="stat-item">
              <span className="stat-label">逃げ想定</span>
              <span className="stat-value">{scenario.front_runner_count}頭</span>
            </div>
          )}
          {scenario.stalker_count !== undefined && (
            <div className="stat-item">
              <span className="stat-label">先行想定</span>
              <span className="stat-value">{scenario.stalker_count}頭</span>
            </div>
          )}
          {scenario.closer_count !== undefined && (
            <div className="stat-item">
              <span className="stat-label">差し想定</span>
              <span className="stat-value">{scenario.closer_count}頭</span>
            </div>
          )}
          {scenario.cushion_value !== undefined && (
            <div className="stat-item">
              <span className="stat-label">クッション値</span>
              <span className="stat-value">{scenario.cushion_value}</span>
            </div>
          )}
        </div>
      </section>

      {/* 上位馬 */}
      {summary?.top_by_adj_win && summary.top_by_adj_win.length > 0 && (
        <section className="summary-section">
          <h3 className="summary-section-title">注目馬（補正後上位）</h3>
          <ul className="summary-horse-list">
            {summary.top_by_adj_win.slice(0, 5).map((horse) => (
              <li key={horse.horse_id} className="summary-horse-item">
                <span className="summary-horse-name">{horse.name}</span>
                {horse.adj_win !== undefined && (
                  <span className="summary-horse-value">
                    {(horse.adj_win * 100).toFixed(1)}%
                  </span>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* プラス補正馬 */}
      {summary?.boosted_horses && summary.boosted_horses.length > 0 && (
        <section className="summary-section">
          <h3 className="summary-section-title positive-title">
            ↑ シナリオ恩恵馬
          </h3>
          <ul className="summary-horse-list">
            {summary.boosted_horses.slice(0, 5).map((horse) => (
              <li key={horse.horse_id} className="summary-horse-item boosted">
                <span className="summary-horse-name">{horse.name}</span>
                {horse.delta !== undefined && (
                  <span className="summary-horse-delta positive">
                    +{(horse.delta * 100).toFixed(1)}%
                  </span>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* マイナス補正馬 */}
      {summary?.dropped_horses && summary.dropped_horses.length > 0 && (
        <section className="summary-section">
          <h3 className="summary-section-title negative-title">
            ↓ シナリオ不利馬
          </h3>
          <ul className="summary-horse-list">
            {summary.dropped_horses.slice(0, 5).map((horse) => (
              <li key={horse.horse_id} className="summary-horse-item dropped">
                <span className="summary-horse-name">{horse.name}</span>
                {horse.delta !== undefined && (
                  <span className="summary-horse-delta negative">
                    {(horse.delta * 100).toFixed(1)}%
                  </span>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* 堅軸候補 */}
      {summary?.solid_placers && summary.solid_placers.length > 0 && (
        <section className="summary-section">
          <h3 className="summary-section-title">堅軸候補（複勝安定）</h3>
          <ul className="summary-horse-list">
            {summary.solid_placers.slice(0, 3).map((horse) => (
              <li key={horse.horse_id} className="summary-horse-item solid">
                <span className="summary-horse-name">{horse.name}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* 追加メモ */}
      {summary?.notes && summary.notes.length > 0 && (
        <section className="summary-section">
          <h3 className="summary-section-title">メモ</h3>
          <ul className="summary-notes">
            {summary.notes.map((note, index) => (
              <li key={index} className="summary-note">
                {note}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

export default ScenarioSummary;
