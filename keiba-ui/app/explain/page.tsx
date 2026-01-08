"use client";

import React, { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";

/**
 * Feature explanation data structure
 */
interface FeatureExplanation {
  feature_name: string;
  display_name: string;
  origin: "v4_native" | "v3_bridged";
  safety_label: "safe" | "warn" | "unsafe" | "unknown";
  safety_notes: string;
  importance_gain: number;
  importance_split: number;
  contribution: number;
}

/**
 * Explain JSON structure
 */
interface ExplainData {
  schema_version: string;
  generated_at: string;
  model_version: string;
  target: string;
  n_features: number;
  n_bridged: number;
  n_native: number;
  features: FeatureExplanation[];
  metadata?: Record<string, unknown>;
}

/**
 * Default explain JSON path
 */
const DEFAULT_EXPLAIN_PATH = "/api/explain";

/**
 * Fetch explain data from API or file
 */
async function fetchExplainData(path: string): Promise<ExplainData> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load explain data: ${response.status} ${response.statusText}`);
  }
  const data = await response.json();

  // Validate required fields
  if (!data.features || !Array.isArray(data.features)) {
    throw new Error("Invalid explain data: missing features array");
  }
  if (!data.target) {
    throw new Error("Invalid explain data: missing target field");
  }

  return data;
}

/**
 * Safety label badge component
 */
function SafetyBadge({ label }: { label: string }) {
  const colors: Record<string, string> = {
    safe: "bg-green-100 text-green-800",
    warn: "bg-yellow-100 text-yellow-800",
    unsafe: "bg-red-100 text-red-800",
    unknown: "bg-gray-100 text-gray-800",
  };
  return (
    <span className={`px-2 py-1 rounded text-xs font-medium ${colors[label] || colors.unknown}`}>
      {label}
    </span>
  );
}

/**
 * Origin badge component
 */
function OriginBadge({ origin }: { origin: string }) {
  const isNative = origin === "v4_native";
  return (
    <span
      className={`px-2 py-1 rounded text-xs font-medium ${
        isNative ? "bg-blue-100 text-blue-800" : "bg-purple-100 text-purple-800"
      }`}
    >
      {isNative ? "v4" : "v3"}
    </span>
  );
}

/**
 * Explain Viewer Page
 */
export default function ExplainPage() {
  const searchParams = useSearchParams();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ExplainData | null>(null);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Use query param if provided, otherwise default
        const path = searchParams.get("path") || DEFAULT_EXPLAIN_PATH;
        const explainData = await fetchExplainData(path);
        setData(explainData);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to load explain data";
        setError(message);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [searchParams]);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-green-700 to-green-800 text-white shadow-lg">
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
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <div>
                <h1 className="text-xl font-bold">Feature Explainer</h1>
                <p className="text-sm text-green-200">特徴量説明ビューア</p>
              </div>
            </div>
            <Link
              href="/scenario"
              className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded-lg text-sm font-medium transition-colors"
            >
              シナリオ分析へ
            </Link>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* Error */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center gap-3">
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
                <p className="font-medium text-red-800">Explain JSONの読み込みに失敗しました</p>
                <p className="text-sm text-red-600">{error}</p>
                <p className="text-xs text-red-500 mt-1">
                  生成コマンド: python -m src.features_v4.explain_runner --model-dir models --target target_win
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="bg-white rounded-lg shadow p-8 flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600 mx-auto"></div>
              <p className="mt-4 text-gray-600">読み込み中...</p>
            </div>
          </div>
        )}

        {/* Data Display */}
        {!loading && !error && data && (
          <>
            {/* Summary */}
            <div className="bg-white rounded-lg shadow p-6 mb-6">
              <h2 className="text-lg font-bold text-gray-800 mb-4">概要</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-500">ターゲット</p>
                  <p className="text-lg font-bold text-gray-800">{data.target}</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-500">特徴量数</p>
                  <p className="text-lg font-bold text-gray-800">{data.n_features}</p>
                </div>
                <div className="bg-blue-50 rounded-lg p-4">
                  <p className="text-sm text-blue-600">v4 Native</p>
                  <p className="text-lg font-bold text-blue-800">{data.n_native}</p>
                </div>
                <div className="bg-purple-50 rounded-lg p-4">
                  <p className="text-sm text-purple-600">v3 Bridged</p>
                  <p className="text-lg font-bold text-purple-800">{data.n_bridged}</p>
                </div>
              </div>
              <div className="mt-4 text-xs text-gray-400">
                生成日時: {data.generated_at} | バージョン: {data.model_version} | スキーマ: {data.schema_version}
              </div>
            </div>

            {/* Features Table */}
            <div className="bg-white rounded-lg shadow overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-bold text-gray-800">特徴量一覧</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        #
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        表示名
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        特徴量名
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Origin
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Safety
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Gain
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Split
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {data.features.map((feature, index) => (
                      <tr key={feature.feature_name} className="hover:bg-gray-50">
                        <td className="px-4 py-3 text-sm text-gray-500">{index + 1}</td>
                        <td className="px-4 py-3 text-sm font-medium text-gray-900">
                          {feature.display_name}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-500 font-mono text-xs">
                          {feature.feature_name}
                        </td>
                        <td className="px-4 py-3">
                          <OriginBadge origin={feature.origin} />
                        </td>
                        <td className="px-4 py-3">
                          <SafetyBadge label={feature.safety_label} />
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900 text-right font-mono">
                          {feature.importance_gain.toFixed(2)}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900 text-right font-mono">
                          {feature.importance_split}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-400 py-4 mt-8">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm">
          <p>Feature Explainer © 2024</p>
        </div>
      </footer>
    </div>
  );
}
