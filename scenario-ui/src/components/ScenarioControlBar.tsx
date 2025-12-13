import React from "react";
import type { SortType } from "../types";

interface ScenarioOption {
  id: string;
  label: string;
  file: string;
}

interface ScenarioControlBarProps {
  scenarios: ScenarioOption[];
  selectedScenario: string;
  onScenarioChange: (scenarioId: string) => void;
  sortBy: SortType;
  onSortChange: (sortBy: SortType) => void;
}

/**
 * ソートオプション
 */
const SORT_OPTIONS: { value: SortType; label: string }[] = [
  { value: "adj_win", label: "補正後勝率" },
  { value: "delta_win", label: "変動幅" },
  { value: "frame_no", label: "枠番" },
  { value: "odds", label: "オッズ" },
];

/**
 * シナリオ選択・ソート切り替えバー
 */
function ScenarioControlBar({
  scenarios,
  selectedScenario,
  onScenarioChange,
  sortBy,
  onSortChange,
}: ScenarioControlBarProps): React.ReactElement {
  const handleScenarioSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onScenarioChange(e.target.value);
  };

  const handleSortSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onSortChange(e.target.value as SortType);
  };

  return (
    <div className="control-bar">
      <div className="control-group">
        <label htmlFor="scenario-select" className="control-label">
          シナリオ:
        </label>
        <select
          id="scenario-select"
          className="control-select"
          value={selectedScenario}
          onChange={handleScenarioSelect}
        >
          {scenarios.map((scenario) => (
            <option key={scenario.id} value={scenario.id}>
              {scenario.label}
            </option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label htmlFor="sort-select" className="control-label">
          並び替え:
        </label>
        <select
          id="sort-select"
          className="control-select"
          value={sortBy}
          onChange={handleSortSelect}
        >
          {SORT_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

export default ScenarioControlBar;
