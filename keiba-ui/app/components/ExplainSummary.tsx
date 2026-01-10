"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";

/**
 * Feature explanation item
 */
interface FeatureItem {
  feature_name: string;
  display_name: string;
  origin: "v4_native" | "v3_bridged";
  safety_label: "safe" | "warn" | "unsafe" | "unknown";
  importance_gain: number;
  importance_split: number;
  desc?: string;
}

/**
 * Explain API response
 */
interface ExplainResponse {
  target: string;
  n_features: number;
  features: FeatureItem[];
  error?: string;
  message?: string;
}

interface ExplainSummaryProps {
  target?: string;
  topN?: number;
}

/**
 * Origin badge
 */
function OriginBadge({ origin }: { origin: string }) {
  const isNative = origin === "v4_native";
  return (
    <span
      className={`px-1.5 py-0.5 rounded text-xs font-medium ${
        isNative ? "bg-blue-100 text-blue-700" : "bg-purple-100 text-purple-700"
      }`}
    >
      {isNative ? "v4" : "v3"}
    </span>
  );
}

/**
 * Safety badge
 */
function SafetyBadge({ label }: { label: string }) {
  const colors: Record<string, string> = {
    safe: "bg-green-100 text-green-700",
    warn: "bg-yellow-100 text-yellow-700",
    unsafe: "bg-red-100 text-red-700",
    unknown: "bg-gray-100 text-gray-700",
  };
  return (
    <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${colors[label] || colors.unknown}`}>
      {label}
    </span>
  );
}

/**
 * Explain Summary Component
 * Shows top N features with importance
 */
export default function ExplainSummary({ target = "target_win", topN = 10 }: ExplainSummaryProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [features, setFeatures] = useState<FeatureItem[]>([]);
  const [totalFeatures, setTotalFeatures] = useState(0);

  useEffect(() => {
    const fetchExplain = async () => {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch(`/api/explain?target=${target}`);
        const data: ExplainResponse = await res.json();

        if (!res.ok || data.error) {
          throw new Error(data.message || data.error || "Failed to load explain data");
        }

        setFeatures(data.features?.slice(0, topN) || []);
        setTotalFeatures(data.n_features || 0);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Explainデータの読み込みに失敗";
        setError(msg);
      } finally {
        setLoading(false);
      }
    };

    fetchExplain();
  }, [target, topN]);

  return (
    <div className="bg-white rounded-lg shadow mt-6">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
        <h3 className="text-base font-bold text-gray-800">
          Explain（特徴量重要度）
        </h3>
        <Link
          href={`/explain?target=${target}`}
          className="text-sm text-blue-600 hover:text-blue-800 hover:underline"
        >
          詳細を見る →
        </Link>
      </div>

      {/* Content */}
      <div className="p-4">
        {/* Loading */}
        {loading && (
          <div className="text-center py-4">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-2 text-sm text-gray-500">Loading...</p>
          </div>
        )}

        {/* Error */}
        {!loading && error && (
          <div className="bg-yellow-50 border border-yellow-200 rounded p-3">
            <p className="text-sm text-yellow-800">Explainデータがありません</p>
            <p className="text-xs text-yellow-600 mt-1">
              生成: python -m src.features_v4.explain_runner --model-dir models --target {target}
            </p>
          </div>
        )}

        {/* Features Table */}
        {!loading && !error && features.length > 0 && (
          <>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-500 uppercase">
                  <th className="pb-2">#</th>
                  <th className="pb-2">特徴量</th>
                  <th className="pb-2">Origin</th>
                  <th className="pb-2">Safety</th>
                  <th className="pb-2 text-right">Gain</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {features.map((f, i) => (
                  <tr key={f.feature_name} className="hover:bg-gray-50">
                    <td className="py-2 text-gray-400">{i + 1}</td>
                    <td className="py-2">
                      <div className="font-medium text-gray-800">{f.display_name}</div>
                      {f.desc && <div className="text-xs text-gray-400">{f.desc}</div>}
                    </td>
                    <td className="py-2">
                      <OriginBadge origin={f.origin} />
                    </td>
                    <td className="py-2">
                      <SafetyBadge label={f.safety_label} />
                    </td>
                    <td className="py-2 text-right font-mono text-gray-600">
                      {f.importance_gain.toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 text-xs text-gray-400 text-right">
              {topN} / {totalFeatures} 件を表示
            </p>
          </>
        )}

        {/* No features */}
        {!loading && !error && features.length === 0 && (
          <p className="text-sm text-gray-500 text-center py-4">
            特徴量データがありません
          </p>
        )}
      </div>
    </div>
  );
}
