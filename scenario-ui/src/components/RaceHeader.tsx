import React from "react";
import type { RaceData, ScenarioData } from "../types";

interface RaceHeaderProps {
  race: RaceData;
  scenario: ScenarioData;
}

/**
 * ペースラベルを日本語に変換
 */
function getPaceLabel(pace: string): string {
  const labels: Record<string, string> = {
    S: "スロー",
    M: "ミドル",
    H: "ハイ",
  };
  return labels[pace] || pace;
}

/**
 * バイアスラベルを日本語に変換
 */
function getBiasLabel(bias: string | undefined): string {
  if (!bias) return "不明";
  const labels: Record<string, string> = {
    内: "内伸び",
    外: "外伸び",
    フラット: "フラット",
  };
  return labels[bias] || bias;
}

/**
 * レースヘッダーコンポーネント
 */
function RaceHeader({ race, scenario }: RaceHeaderProps): React.ReactElement {
  const raceName = race.race_name || "レース名未設定";
  const raceInfo = [
    race.course,
    race.surface === "turf" ? "芝" : race.surface === "dirt" ? "ダート" : race.surface,
    race.distance ? `${race.distance}m` : null,
    race.race_class,
  ]
    .filter(Boolean)
    .join(" / ");

  return (
    <header className="race-header">
      <div className="race-header-main">
        <h1 className="race-name">{raceName}</h1>
        {race.race_date && <span className="race-date">{race.race_date}</span>}
      </div>
      
      {raceInfo && <p className="race-info">{raceInfo}</p>}
      
      <div className="scenario-badges">
        <span className={`badge badge-pace pace-${scenario.pace}`}>
          {getPaceLabel(scenario.pace)}ペース
        </span>
        
        {scenario.bias && (
          <span className="badge badge-bias">
            {getBiasLabel(scenario.bias)}
          </span>
        )}
        
        {scenario.track_condition && (
          <span className="badge badge-condition">
            {scenario.track_condition}
          </span>
        )}
      </div>
      
      {scenario.notes && (
        <p className="scenario-notes">{scenario.notes}</p>
      )}
    </header>
  );
}

export default RaceHeader;
