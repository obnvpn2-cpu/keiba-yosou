/**
 * 競馬シナリオ補正UI - 型定義
 */

// 勝率/複勝率の補正情報
export interface ProbabilityData {
  base: number;
  adj: number;
  delta: number;
  delta_pct?: number;
}

// 馬情報
export interface HorseData {
  horse_id: string;
  name: string;
  frame_no?: number;
  horse_no?: number;
  run_style?: string;
  win: ProbabilityData;
  in3: ProbabilityData;
  odds?: number;
  popularity?: number;
  reasons?: string[];
}

// レース情報
export interface RaceData {
  race_id: string;
  race_name?: string;
  course?: string;
  surface?: string;
  distance?: number;
  race_date?: string;
  race_class?: string;
}

// シナリオ情報
export interface ScenarioData {
  scenario_id?: string;
  pace: string; // "S" | "M" | "H"
  track_condition?: string;
  bias?: string;
  notes?: string;
  cushion_value?: number;
  front_runner_count?: number;
  stalker_count?: number;
  closer_count?: number;
  weak_horse_count?: number;
}

// サマリー内の馬情報（簡易版）
export interface SummaryHorse {
  horse_id: string;
  name: string;
  adj_win?: number;
  delta?: number;
}

// サマリー情報
export interface SummaryData {
  top_by_adj_win?: SummaryHorse[];
  boosted_horses?: SummaryHorse[];
  dropped_horses?: SummaryHorse[];
  solid_placers?: SummaryHorse[];
  total_horses?: number;
  notes?: string[];
}

// UIコンテキスト全体（JSONの構造）
export interface ScenarioUiContext {
  race: RaceData;
  scenario: ScenarioData;
  horses: HorseData[];
  summary?: SummaryData;
}

// ソートタイプ
export type SortType = "adj_win" | "delta_win" | "frame_no" | "odds";

// アプリケーション状態
export interface AppState {
  loading: boolean;
  error: string | null;
  data: ScenarioUiContext | null;
  selectedScenario: string;
  sortBy: SortType;
}
