"use client";

import React from "react";
import type { ScenarioCondition } from "@/app/lib/types";
import {
  WEATHER_OPTIONS,
  TRACK_CONDITION_OPTIONS,
  BIAS_OPTIONS,
  PACE_OPTIONS,
} from "@/app/lib/types";

interface ScenarioControlProps {
  condition: ScenarioCondition;
  onChange: (condition: ScenarioCondition) => void;
  onRecalculate: () => void;
  loading: boolean;
}

/**
 * シナリオ条件設定コンポーネント
 */
export default function ScenarioControl({
  condition,
  onChange,
  onRecalculate,
  loading,
}: ScenarioControlProps) {
  // 条件更新ヘルパー
  const updateCondition = <K extends keyof ScenarioCondition>(
    key: K,
    value: ScenarioCondition[K]
  ) => {
    onChange({ ...condition, [key]: value });
  };

  return (
    <div className="section-card mb-6">
      <h2 className="section-title flex items-center gap-2">
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
            d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
          />
        </svg>
        シナリオ条件設定
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* 天候 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            天候
          </label>
          <div className="radio-group">
            {WEATHER_OPTIONS.map((weather) => (
              <button
                key={weather}
                type="button"
                onClick={() => updateCondition("weather", weather)}
                className={`radio-option ${
                  condition.weather === weather ? "radio-option-selected" : ""
                }`}
              >
                {weather === "晴" && "☀️ "}
                {weather === "曇" && "☁️ "}
                {weather === "雨" && "🌧️ "}
                {weather === "雪" && "❄️ "}
                {weather}
              </button>
            ))}
          </div>
        </div>

        {/* 馬場状態 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            馬場状態
          </label>
          <div className="radio-group">
            {TRACK_CONDITION_OPTIONS.map((tc) => (
              <button
                key={tc}
                type="button"
                onClick={() => updateCondition("trackCondition", tc)}
                className={`radio-option ${
                  condition.trackCondition === tc ? "radio-option-selected" : ""
                }`}
              >
                {tc}
              </button>
            ))}
          </div>
        </div>

        {/* バイアス */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            トラックバイアス
          </label>
          <div className="radio-group">
            {BIAS_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() =>
                  updateCondition("bias", opt.value as ScenarioCondition["bias"])
                }
                className={`radio-option ${
                  condition.bias === opt.value ? "radio-option-selected" : ""
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* ペース */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            想定ペース
          </label>
          <div className="radio-group">
            {PACE_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() =>
                  updateCondition("pace", opt.value as ScenarioCondition["pace"])
                }
                className={`radio-option ${
                  condition.pace === opt.value ? "radio-option-selected" : ""
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* 含水率 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            含水率 (%)
          </label>
          <div className="slider-container">
            <input
              type="range"
              min={0}
              max={30}
              step={1}
              value={condition.moisture}
              onChange={(e) =>
                updateCondition("moisture", Number(e.target.value))
              }
              className="slider-input flex-1"
            />
            <span className="slider-value">{condition.moisture}%</span>
          </div>
        </div>

        {/* クッション値 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            クッション値
          </label>
          <div className="slider-container">
            <input
              type="range"
              min={5}
              max={12}
              step={0.1}
              value={condition.cushion}
              onChange={(e) =>
                updateCondition("cushion", Number(e.target.value))
              }
              className="slider-input flex-1"
            />
            <span className="slider-value">{condition.cushion.toFixed(1)}</span>
          </div>
        </div>
      </div>

      {/* 再計算ボタン */}
      <div className="mt-6 flex justify-end">
        <button
          type="button"
          onClick={onRecalculate}
          disabled={loading}
          className={`
            flex items-center gap-2 px-6 py-2.5 rounded-lg font-medium
            transition-all shadow-sm
            ${
              loading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-700 active:bg-blue-800"
            }
            text-white
          `}
        >
          {loading ? (
            <>
              <svg
                className="animate-spin h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              計算中...
            </>
          ) : (
            <>
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              シナリオ再計算
            </>
          )}
        </button>
      </div>
    </div>
  );
}
