"use client";

import React, { useState, useEffect, useCallback } from "react";
import type { Horse, ScenarioCondition, RaceInfo, ScenarioData } from "@/app/lib/types";
import ScenarioControl from "@/app/components/ScenarioControl";
import HorseTable from "@/app/components/HorseTable";
import HorseDetailModal from "@/app/components/HorseDetailModal";
import ExplainSummary from "@/app/components/ExplainSummary";

/**
 * デフォルトのシナリオ条件
 */
const DEFAULT_CONDITION: ScenarioCondition = {
  weather: "晴",
  trackCondition: "良",
  bias: "フラット",
  pace: "M",
  moisture: 10,
  cushion: 9.0,
};

/**
 * シナリオ条件からJSONファイル名を決定
 */
function getScenarioFileName(condition: ScenarioCondition): string {
  // 今はシンプルにペース×バイアスで切り替え
  const paceMap: Record<string, string> = {
    S: "slow",
    M: "middle",
    H: "high",
  };
  const biasMap: Record<string, string> = {
    内: "inner",
    フラット: "flat",
    外: "outer",
  };

  const pacePart = paceMap[condition.pace] || "middle";
  const biasPart = biasMap[condition.bias] || "flat";

  return `scenario_${pacePart}_${biasPart}.json`;
}

/**
 * シナリオデータをfetch
 * API化前の暫定実装（public JSONを読む）
 */
async function fetchScenarioData(
  condition: ScenarioCondition
): Promise<ScenarioData> {
  const filename = getScenarioFileName(condition);

  try {
    const response = await fetch(`/${filename}`);

    if (!response.ok) {
      // ファイルがなければデフォルトを使う
      console.warn(`${filename} not found, falling back to default`);
      const fallback = await fetch("/scenario_default.json");
      if (!fallback.ok) {
        throw new Error("デフォルトシナリオの読み込みに失敗しました");
      }
      return fallback.json();
    }

    return response.json();
  } catch (error) {
    console.error("Fetch error:", error);
    throw error;
  }
}

/**
 * シナリオページコンポーネント
 */
export default function ScenarioPage() {
  // 状態
  const [condition, setCondition] = useState<ScenarioCondition>(DEFAULT_CONDITION);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [raceInfo, setRaceInfo] = useState<RaceInfo | null>(null);
  const [horses, setHorses] = useState<Horse[]>([]);
  const [selectedHorse, setSelectedHorse] = useState<Horse | null>(null);

  /**
   * データ読み込み
   */
  const loadData = useCallback(async (cond: ScenarioCondition) => {
    setLoading(true);
    setError(null);

    try {
      const data = await fetchScenarioData(cond);
      setRaceInfo(data.race);
      setHorses(data.horses);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "データの読み込みに失敗しました";
      setError(message);
      setHorses([]);
    } finally {
      setLoading(false);
    }
  }, []);

  // 初回読み込み
  useEffect(() => {
    loadData(condition);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * 再計算ボタン押下時
   */
  const handleRecalculate = useCallback(() => {
    loadData(condition);
  }, [condition, loadData]);

  /**
   * 馬選択時（モーダル表示）
   */
  const handleSelectHorse = useCallback((horse: Horse) => {
    setSelectedHorse(horse);
  }, []);

  /**
   * モーダル閉じる
   */
  const handleCloseModal = useCallback(() => {
    setSelectedHorse(null);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <header className="bg-gradient-to-r from-blue-700 to-blue-800 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <svg
                className="w-8 h-8"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
              <div>
                <h1 className="text-xl font-bold">競馬シナリオ分析</h1>
                <p className="text-sm text-blue-200">Keiba Scenario Analyzer</p>
              </div>
            </div>

            {/* レース情報 */}
            {raceInfo && (
              <div className="text-right">
                <div className="text-lg font-bold">{raceInfo.race_name}</div>
                <div className="text-sm text-blue-200">
                  {raceInfo.course} {raceInfo.surface === "turf" ? "芝" : "ダート"}{" "}
                  {raceInfo.distance}m
                  {raceInfo.race_class && ` / ${raceInfo.race_class}`}
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* メインコンテンツ */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* 条件設定 */}
        <ScenarioControl
          condition={condition}
          onChange={setCondition}
          onRecalculate={handleRecalculate}
          loading={loading}
        />

        {/* エラー表示 */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
            <svg
              className="w-6 h-6 text-red-600 flex-shrink-0"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div>
              <p className="font-medium text-red-800">エラーが発生しました</p>
              <p className="text-sm text-red-600">{error}</p>
            </div>
            <button
              onClick={handleRecalculate}
              className="ml-auto px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium transition-colors"
            >
              再試行
            </button>
          </div>
        )}

        {/* ローディング表示 */}
        {loading && (
          <div className="section-card flex items-center justify-center py-20">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-4 text-gray-600">シナリオを計算中...</p>
            </div>
          </div>
        )}

        {/* テーブル */}
        {!loading && !error && (
          <HorseTable horses={horses} onSelectHorse={handleSelectHorse} />
        )}

        {/* Explain（特徴量重要度）セクション */}
        {!loading && !error && (
          <ExplainSummary target="target_win" topN={10} />
        )}
      </main>

      {/* 詳細モーダル */}
      <HorseDetailModal horse={selectedHorse} onClose={handleCloseModal} />

      {/* フッター */}
      <footer className="bg-gray-800 text-gray-400 py-4 mt-8">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm">
          <p>競馬シナリオ分析UI © 2024</p>
          <p className="text-xs mt-1">
            ※ このUIは予測支援ツールです。投資判断は自己責任で行ってください。
          </p>
        </div>
      </footer>
    </div>
  );
}
