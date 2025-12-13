"use client";

import React, { useState, useMemo } from "react";
import type { Horse, SortKey, SortDirection, SortState } from "@/app/lib/types";
import { RUN_STYLE_COLORS, FRAME_COLORS } from "@/app/lib/types";

interface HorseTableProps {
  horses: Horse[];
  onSelectHorse: (horse: Horse) => void;
}

// テーブルカラム定義
const COLUMNS: { key: SortKey; label: string; width?: string }[] = [
  { key: "horse_no", label: "馬番", width: "w-16" },
  { key: "name", label: "馬名", width: "w-32" },
  { key: "run_style", label: "脚質", width: "w-20" },
  { key: "base_win", label: "ベース勝率", width: "w-24" },
  { key: "baba_delta", label: "馬場補正", width: "w-24" },
  { key: "pace_delta", label: "ペース補正", width: "w-24" },
  { key: "synergy_score", label: "相性", width: "w-20" },
  { key: "adj_win", label: "最終勝率", width: "w-24" },
  { key: "delta_win", label: "補正差分", width: "w-24" },
];

/**
 * 数値を%表示に変換
 */
function formatPercent(value: number): string {
  return (value * 100).toFixed(1) + "%";
}

/**
 * 差分を符号付きpt表示に変換
 */
function formatDelta(value: number): string {
  const pt = (value * 100).toFixed(1);
  if (value > 0) return `+${pt}pt`;
  if (value < 0) return `${pt}pt`;
  return "±0pt";
}

/**
 * 差分値に応じたCSSクラス
 */
function getDeltaClass(value: number): string {
  if (value > 0.02) return "num-positive font-bold";
  if (value > 0) return "num-positive";
  if (value < -0.02) return "num-negative font-bold";
  if (value < 0) return "num-negative";
  return "num-neutral";
}

/**
 * 相性スコアに応じたCSSクラス
 */
function getSynergyClass(value: number): string {
  if (value >= 80) return "text-green-600 font-bold";
  if (value >= 60) return "text-green-500";
  if (value <= 30) return "text-red-600";
  if (value <= 50) return "text-orange-500";
  return "text-gray-600";
}

/**
 * 馬一覧テーブルコンポーネント
 */
export default function HorseTable({ horses, onSelectHorse }: HorseTableProps) {
  // ソート状態
  const [sortState, setSortState] = useState<SortState>({
    key: "adj_win",
    direction: "desc",
  });

  // ソート処理
  const sortedHorses = useMemo(() => {
    const sorted = [...horses];
    sorted.sort((a, b) => {
      let aVal: number | string;
      let bVal: number | string;

      switch (sortState.key) {
        case "horse_no":
          aVal = a.horse_no;
          bVal = b.horse_no;
          break;
        case "name":
          aVal = a.name;
          bVal = b.name;
          break;
        case "run_style":
          aVal = a.run_style;
          bVal = b.run_style;
          break;
        case "base_win":
          aVal = a.base_win;
          bVal = b.base_win;
          break;
        case "baba_delta":
          aVal = a.baba_delta;
          bVal = b.baba_delta;
          break;
        case "pace_delta":
          aVal = a.pace_delta;
          bVal = b.pace_delta;
          break;
        case "synergy_score":
          aVal = a.synergy_score;
          bVal = b.synergy_score;
          break;
        case "adj_win":
          aVal = a.adj_win;
          bVal = b.adj_win;
          break;
        case "delta_win":
          aVal = a.delta_win;
          bVal = b.delta_win;
          break;
        default:
          return 0;
      }

      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortState.direction === "asc"
          ? aVal.localeCompare(bVal, "ja")
          : bVal.localeCompare(aVal, "ja");
      }

      const numA = Number(aVal);
      const numB = Number(bVal);
      return sortState.direction === "asc" ? numA - numB : numB - numA;
    });

    return sorted;
  }, [horses, sortState]);

  // ヘッダークリックでソート切替
  const handleSort = (key: SortKey) => {
    setSortState((prev) => {
      if (prev.key === key) {
        return {
          key,
          direction: prev.direction === "asc" ? "desc" : "asc",
        };
      }
      return { key, direction: "desc" };
    });
  };

  // ソートアイコン
  const SortIcon = ({ columnKey }: { columnKey: SortKey }) => {
    if (sortState.key !== columnKey) {
      return (
        <svg
          className="w-4 h-4 text-gray-400 ml-1"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"
          />
        </svg>
      );
    }

    return sortState.direction === "asc" ? (
      <svg
        className="w-4 h-4 text-blue-600 ml-1"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M5 15l7-7 7 7"
        />
      </svg>
    ) : (
      <svg
        className="w-4 h-4 text-blue-600 ml-1"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M19 9l-7 7-7-7"
        />
      </svg>
    );
  };

  if (horses.length === 0) {
    return (
      <div className="section-card text-center py-12">
        <p className="text-gray-500">馬データがありません</p>
      </div>
    );
  }

  return (
    <div className="section-card overflow-hidden">
      <h2 className="section-title flex items-center justify-between">
        <span className="flex items-center gap-2">
          <svg
            className="w-5 h-5 text-blue-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
            />
          </svg>
          出走馬一覧
        </span>
        <span className="text-sm font-normal text-gray-500">
          {horses.length}頭
        </span>
      </h2>

      <div className="overflow-x-auto -mx-4">
        <table className="w-full min-w-[900px]">
          <thead>
            <tr className="border-b border-gray-200">
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className={`table-header ${col.width || ""}`}
                >
                  <div className="flex items-center">
                    {col.label}
                    <SortIcon columnKey={col.key} />
                  </div>
                </th>
              ))}
              <th className="table-header w-48">解説</th>
            </tr>
          </thead>
          <tbody>
            {sortedHorses.map((horse, index) => {
              const frameNo = horse.frame_no ?? Math.ceil(horse.horse_no / 2);
              const frameColorClass =
                FRAME_COLORS[frameNo] || FRAME_COLORS[8];
              const runStyleClass =
                RUN_STYLE_COLORS[horse.run_style] || "bg-gray-100 text-gray-800";

              return (
                <tr
                  key={horse.horse_id}
                  onClick={() => onSelectHorse(horse)}
                  className={`table-row ${
                    index % 2 === 0 ? "bg-white" : "bg-gray-50"
                  }`}
                >
                  {/* 馬番 */}
                  <td className="table-cell">
                    <span
                      className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold ${frameColorClass}`}
                    >
                      {horse.horse_no}
                    </span>
                  </td>

                  {/* 馬名 */}
                  <td className="table-cell font-medium text-gray-900">
                    {horse.name}
                  </td>

                  {/* 脚質 */}
                  <td className="table-cell">
                    <span className={`badge ${runStyleClass}`}>
                      {horse.run_style}
                    </span>
                  </td>

                  {/* ベース勝率 */}
                  <td className="table-cell text-gray-500">
                    {formatPercent(horse.base_win)}
                  </td>

                  {/* 馬場補正 */}
                  <td className={`table-cell ${getDeltaClass(horse.baba_delta)}`}>
                    {formatDelta(horse.baba_delta)}
                  </td>

                  {/* ペース補正 */}
                  <td className={`table-cell ${getDeltaClass(horse.pace_delta)}`}>
                    {formatDelta(horse.pace_delta)}
                  </td>

                  {/* 相性スコア */}
                  <td
                    className={`table-cell ${getSynergyClass(horse.synergy_score)}`}
                  >
                    {horse.synergy_score}
                  </td>

                  {/* 最終勝率 */}
                  <td className="table-cell font-bold text-blue-700">
                    {formatPercent(horse.adj_win)}
                  </td>

                  {/* 補正差分 */}
                  <td className={`table-cell ${getDeltaClass(horse.delta_win)}`}>
                    {formatDelta(horse.delta_win)}
                  </td>

                  {/* 解説（省略表示） */}
                  <td className="table-cell">
                    <p className="text-xs text-gray-600 line-clamp-2 max-w-[200px]">
                      {horse.comment || "-"}
                    </p>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="mt-4 text-sm text-gray-500 text-center">
        ※ 行をクリックすると詳細を表示します
      </div>
    </div>
  );
}
