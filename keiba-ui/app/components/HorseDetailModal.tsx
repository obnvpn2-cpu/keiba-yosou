"use client";

import React, { useEffect } from "react";
import type { Horse } from "@/app/lib/types";
import { RUN_STYLE_COLORS, FRAME_COLORS } from "@/app/lib/types";

interface HorseDetailModalProps {
  horse: Horse | null;
  onClose: () => void;
}

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
 * 馬詳細モーダルコンポーネント
 */
export default function HorseDetailModal({
  horse,
  onClose,
}: HorseDetailModalProps) {
  // ESCキーでモーダルを閉じる
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    if (horse) {
      document.addEventListener("keydown", handleEsc);
      document.body.style.overflow = "hidden";
    }

    return () => {
      document.removeEventListener("keydown", handleEsc);
      document.body.style.overflow = "auto";
    };
  }, [horse, onClose]);

  if (!horse) return null;

  const frameNo = horse.frame_no ?? Math.ceil(horse.horse_no / 2);
  const frameColorClass = FRAME_COLORS[frameNo] || FRAME_COLORS[8];
  const runStyleClass =
    RUN_STYLE_COLORS[horse.run_style] || "bg-gray-100 text-gray-800";

  // 勝率推移の計算
  const afterBaba = horse.base_win + horse.baba_delta;
  const afterPace = afterBaba + horse.pace_delta;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ヘッダー */}
        <div className="sticky top-0 bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <span
              className={`inline-flex items-center justify-center w-12 h-12 rounded-full text-lg font-bold ${frameColorClass}`}
            >
              {horse.horse_no}
            </span>
            <div>
              <h2 className="text-xl font-bold">{horse.name}</h2>
              <span className={`badge ${runStyleClass} mt-1`}>
                {horse.run_style}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/20 rounded-full transition-colors"
            aria-label="閉じる"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* コンテンツ */}
        <div className="p-6 space-y-6">
          {/* 勝率推移 */}
          <section>
            <h3 className="text-lg font-bold text-gray-800 mb-3 flex items-center gap-2">
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
                  d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                />
              </svg>
              勝率推移
            </h3>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-between gap-2 text-sm">
                <div className="text-center">
                  <div className="text-gray-500 mb-1">ベース</div>
                  <div className="text-xl font-bold text-gray-700">
                    {formatPercent(horse.base_win)}
                  </div>
                </div>
                <svg
                  className="w-6 h-6 text-gray-400 flex-shrink-0"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5l7 7-7 7"
                  />
                </svg>
                <div className="text-center">
                  <div className="text-gray-500 mb-1">+馬場補正</div>
                  <div className="text-xl font-bold text-gray-700">
                    {formatPercent(afterBaba)}
                  </div>
                  <div
                    className={`text-xs ${
                      horse.baba_delta > 0
                        ? "text-green-600"
                        : horse.baba_delta < 0
                        ? "text-red-600"
                        : "text-gray-500"
                    }`}
                  >
                    ({formatDelta(horse.baba_delta)})
                  </div>
                </div>
                <svg
                  className="w-6 h-6 text-gray-400 flex-shrink-0"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5l7 7-7 7"
                  />
                </svg>
                <div className="text-center">
                  <div className="text-gray-500 mb-1">+ペース補正</div>
                  <div className="text-xl font-bold text-gray-700">
                    {formatPercent(afterPace)}
                  </div>
                  <div
                    className={`text-xs ${
                      horse.pace_delta > 0
                        ? "text-green-600"
                        : horse.pace_delta < 0
                        ? "text-red-600"
                        : "text-gray-500"
                    }`}
                  >
                    ({formatDelta(horse.pace_delta)})
                  </div>
                </div>
                <svg
                  className="w-6 h-6 text-gray-400 flex-shrink-0"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5l7 7-7 7"
                  />
                </svg>
                <div className="text-center">
                  <div className="text-gray-500 mb-1">最終</div>
                  <div className="text-2xl font-bold text-blue-700">
                    {formatPercent(horse.adj_win)}
                  </div>
                  <div
                    className={`text-xs font-medium ${
                      horse.delta_win > 0
                        ? "text-green-600"
                        : horse.delta_win < 0
                        ? "text-red-600"
                        : "text-gray-500"
                    }`}
                  >
                    ({formatDelta(horse.delta_win)})
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* 補正詳細 */}
          <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* 馬場補正 */}
            <div className="bg-amber-50 rounded-lg p-4 border border-amber-200">
              <h4 className="text-sm font-medium text-amber-800 mb-2">
                馬場補正
              </h4>
              <div
                className={`text-2xl font-bold ${
                  horse.baba_delta > 0
                    ? "text-green-600"
                    : horse.baba_delta < 0
                    ? "text-red-600"
                    : "text-gray-600"
                }`}
              >
                {formatDelta(horse.baba_delta)}
              </div>
            </div>

            {/* ペース補正 */}
            <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
              <h4 className="text-sm font-medium text-blue-800 mb-2">
                ペース補正
              </h4>
              <div
                className={`text-2xl font-bold ${
                  horse.pace_delta > 0
                    ? "text-green-600"
                    : horse.pace_delta < 0
                    ? "text-red-600"
                    : "text-gray-600"
                }`}
              >
                {formatDelta(horse.pace_delta)}
              </div>
            </div>

            {/* 相性スコア */}
            <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
              <h4 className="text-sm font-medium text-purple-800 mb-2">
                相性スコア
              </h4>
              <div
                className={`text-2xl font-bold ${
                  horse.synergy_score >= 70
                    ? "text-green-600"
                    : horse.synergy_score <= 40
                    ? "text-red-600"
                    : "text-gray-600"
                }`}
              >
                {horse.synergy_score}
              </div>
            </div>
          </section>

          {/* SHAP上位特徴 */}
          {horse.shap_top && horse.shap_top.length > 0 && (
            <section>
              <h3 className="text-lg font-bold text-gray-800 mb-3 flex items-center gap-2">
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
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
                SHAP上位特徴（予測への貢献度）
              </h3>
              <div className="space-y-2">
                {horse.shap_top.map((shap, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between bg-gray-50 rounded-lg px-4 py-2"
                  >
                    <span className="text-sm text-gray-700">{shap.name}</span>
                    <span
                      className={`text-sm font-medium ${
                        shap.value > 0 ? "text-green-600" : "text-red-600"
                      }`}
                    >
                      {shap.value > 0 ? "+" : ""}
                      {shap.value.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 解説コメント */}
          {horse.comment && (
            <section>
              <h3 className="text-lg font-bold text-gray-800 mb-3 flex items-center gap-2">
                <svg
                  className="w-5 h-5 text-green-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                好材料
              </h3>
              <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                <p className="text-sm text-gray-700 whitespace-pre-wrap">
                  {horse.comment}
                </p>
              </div>
            </section>
          )}

          {/* 懸念点 */}
          {horse.concerns && (
            <section>
              <h3 className="text-lg font-bold text-gray-800 mb-3 flex items-center gap-2">
                <svg
                  className="w-5 h-5 text-red-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
                懸念点
              </h3>
              <div className="bg-red-50 rounded-lg p-4 border border-red-200">
                <p className="text-sm text-gray-700 whitespace-pre-wrap">
                  {horse.concerns}
                </p>
              </div>
            </section>
          )}

          {/* 理由タグ */}
          {horse.reasons && horse.reasons.length > 0 && (
            <section>
              <h3 className="text-lg font-bold text-gray-800 mb-3 flex items-center gap-2">
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
                    d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
                  />
                </svg>
                補正理由タグ
              </h3>
              <div className="flex flex-wrap gap-2">
                {horse.reasons.map((reason, index) => {
                  const isPositive =
                    reason.includes("有利") ||
                    reason.includes("逃げ") ||
                    reason.includes("先行");
                  const isNegative =
                    reason.includes("不利") || reason.includes("信頼度低");

                  return (
                    <span
                      key={index}
                      className={`badge ${
                        isPositive
                          ? "badge-positive"
                          : isNegative
                          ? "badge-negative"
                          : "badge-neutral"
                      }`}
                    >
                      {reason}
                    </span>
                  );
                })}
              </div>
            </section>
          )}
        </div>

        {/* フッター */}
        <div className="sticky bottom-0 bg-gray-50 px-6 py-4 border-t border-gray-200 flex justify-end">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-lg font-medium transition-colors"
          >
            閉じる
          </button>
        </div>
      </div>
    </div>
  );
}
