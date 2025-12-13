// // src/App.tsx
// import { useEffect, useState } from "react";
// import type { ScenarioUiContext, HorseRow } from "./types";

// type SortKey = "adj_win" | "base_win" | "delta";

// function formatProb(p: number | null | undefined): string {
//   if (p == null || Number.isNaN(p)) return "-";
//   return `${(p * 100).toFixed(1)}%`;
// }

// function formatDelta(p: number): string {
//   const sign = p > 0 ? "+" : "";
//   return `${sign}${(p * 100).toFixed(1)}%`;
// }

// function App() {
//   const [data, setData] = useState<ScenarioUiContext | null>(null);
//   const [sortKey, setSortKey] = useState<SortKey>("adj_win");
//   const [showOnlyBoosted, setShowOnlyBoosted] = useState(false);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);

//   // 初回マウント時に JSON を読み込む
//   useEffect(() => {
//     const load = async () => {
//       try {
//         setLoading(true);
//         setError(null);

//         const res = await fetch("/ui_slow_inner_202306050811.json");
//         if (!res.ok) {
//           throw new Error(`HTTP ${res.status}`);
//         }

//         const json = (await res.json()) as ScenarioUiContext;
//         setData(json);
//       } catch (e: any) {
//         console.error(e);
//         setError(String(e?.message ?? e));
//       } finally {
//         setLoading(false);
//       }
//     };

//     load();
//   }, []);

//   // ローディング・エラー・データ無しのハンドリング
//   if (loading) {
//     return (
//       <div className="min-h-screen flex items-center justify-center bg-slate-100">
//         <p className="text-slate-600 text-sm">JSON 読み込み中...</p>
//       </div>
//     );
//   }

//   if (error) {
//     return (
//       <div className="min-h-screen flex items-center justify-center bg-slate-100">
//         <div className="bg-white shadow rounded p-4 max-w-md">
//           <h1 className="font-semibold mb-2 text-rose-600 text-sm">
//             読み込みエラー
//           </h1>
//           <p className="text-xs text-slate-700 break-all">
//             /ui_slow_inner_202306050811.json の読み込みに失敗しました。
//             <br />
//             {error}
//           </p>
//         </div>
//       </div>
//     );
//   }

//   if (!data) {
//     return (
//       <div className="min-h-screen flex items-center justify-center bg-slate-100">
//         <p className="text-slate-600 text-sm">
//           データがありません（JSON が空か、パースに失敗しました）。
//         </p>
//       </div>
//     );
//   }

//   const horsesSorted: HorseRow[] = [...data.horses].sort((a, b) => {
//     if (sortKey === "adj_win") {
//       return b.win.adj - a.win.adj;
//     }
//     if (sortKey === "base_win") {
//       return b.win.base - a.win.base;
//     }
//     // delta
//     return b.win.delta - a.win.delta;
//   });

//   const horsesFiltered = showOnlyBoosted
//     ? horsesSorted.filter((h) => h.win.delta > 0)
//     : horsesSorted;

//   const summary = data.summary;

//   return (
//     <div className="min-h-screen bg-slate-100 text-slate-900">
//       <header className="bg-slate-900 text-white">
//         <div className="mx-auto max-w-5xl px-4 py-4 flex flex-col gap-2 sm:flex-row sm:items-baseline sm:justify-between">
//           <div>
//             <h1 className="text-xl font-semibold">
//               {data.race.race_name || "レース名不明"}
//             </h1>
//             <p className="text-sm text-slate-300">
//               {data.race.race_date || "日付不明"} / {data.race.course ?? "-"}{" "}
//               {data.race.surface ?? "-"} {data.race.distance ?? "-"}m /{" "}
//               {data.race.race_class ?? "-"}
//             </p>
//           </div>
//           <div className="text-sm text-slate-200">
//             <div>
//               シナリオ:{" "}
//               <span className="font-mono bg-slate-800 px-2 py-1 rounded">
//                 {data.scenario.scenario_id}
//               </span>
//             </div>
//             <div>
//               ペース: <b>{data.scenario.pace}</b> / 馬場:{" "}
//               <b>{data.scenario.track_condition}</b> / バイアス:{" "}
//               <b>{data.scenario.bias}</b>
//             </div>
//           </div>
//         </div>
//       </header>

//       <main className="mx-auto max-w-5xl px-4 py-6 flex flex-col gap-6">
//         {/* シナリオ概要 */}
//         <section className="bg-white rounded-lg shadow p-4">
//           <h2 className="text-sm font-semibold text-slate-700 mb-2">
//             シナリオ概要
//           </h2>
//           <p className="text-sm whitespace-pre-wrap">
//             {data.scenario.notes || "（補足説明なし）"}
//           </p>
//           <div className="mt-3 text-xs text-slate-500 flex gap-4 flex-wrap">
//             {data.scenario.cushion_value != null && (
//               <span>クッション値: {data.scenario.cushion_value}</span>
//             )}
//             <span>
//               逃げ: {data.scenario.front_runner_count ?? 0} /
//               先行〜中団: {data.scenario.stalker_count ?? 0} /
//               差し・追込: {data.scenario.closer_count ?? 0}
//             </span>
//           </div>
//         </section>

//         {/* 注目馬サマリー */}
//         <section className="grid gap-4 md:grid-cols-3">
//           <div className="bg-white rounded-lg shadow p-4">
//             <h3 className="text-sm font-semibold text-slate-700 mb-2">
//               調整後勝率トップ
//             </h3>
//             <ul className="space-y-1 text-sm">
//               {summary?.top_by_adj_win.slice(0, 5).map((h) => (
//                 <li key={h.id} className="flex justify-between gap-2">
//                   <span>
//                     {h.name}{" "}
//                     <span className="text-xs text-slate-500">
//                       （{h.run_style}）
//                     </span>
//                   </span>
//                   <span className="font-mono">
//                     {formatProb(h.adj_win)}{" "}
//                     <span className="text-xs text-slate-500">
//                       ({formatProb(h.base_win)})
//                     </span>
//                   </span>
//                 </li>
//               )) || <li>データなし</li>}
//             </ul>
//           </div>

//           <div className="bg-white rounded-lg shadow p-4">
//             <h3 className="text-sm font-semibold text-slate-700 mb-2">
//               上げた馬（プラス補正）
//             </h3>
//             <ul className="space-y-1 text-sm">
//               {summary?.boosted_horses.slice(0, 5).map((h) => (
//                 <li key={h.id} className="flex justify-between gap-2">
//                   <span>{h.name}</span>
//                   <span className="font-mono text-emerald-600">
//                     {formatDelta(h.win_delta)}
//                   </span>
//                 </li>
//               )) || <li>データなし</li>}
//             </ul>
//           </div>

//           <div className="bg-white rounded-lg shadow p-4">
//             <h3 className="text-sm font-semibold text-slate-700 mb-2">
//               評価を落とした馬
//             </h3>
//             <ul className="space-y-1 text-sm">
//               {summary?.dropped_horses.slice(0, 5).map((h) => (
//                 <li key={h.id} className="flex justify-between gap-2">
//                   <span>{h.name}</span>
//                   <span className="font-mono text-rose-600">
//                     {formatDelta(h.win_delta)}
//                   </span>
//                 </li>
//               )) || <li>データなし</li>}
//             </ul>
//           </div>
//         </section>

//         {/* テーブルコントロール */}
//         <section className="bg-white rounded-lg shadow p-4">
//           <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between mb-3">
//             <h2 className="text-sm font-semibold text-slate-700">
//               馬ごとの詳細（調整前後の勝率）
//             </h2>
//             <div className="flex flex-wrap gap-2 text-xs">
//               <div className="flex items-center gap-1">
//                 <span>ソート:</span>
//                 <button
//                   className={
//                     "px-2 py-1 rounded border text-xs " +
//                     (sortKey === "adj_win"
//                       ? "bg-slate-900 text-white"
//                       : "bg-white")
//                   }
//                   onClick={() => setSortKey("adj_win")}
//                 >
//                   調整後勝率
//                 </button>
//                 <button
//                   className={
//                     "px-2 py-1 rounded border text-xs " +
//                     (sortKey === "base_win"
//                       ? "bg-slate-900 text-white"
//                       : "bg-white")
//                   }
//                   onClick={() => setSortKey("base_win")}
//                 >
//                   ベース勝率
//                 </button>
//                 <button
//                   className={
//                     "px-2 py-1 rounded border text-xs " +
//                     (sortKey === "delta"
//                       ? "bg-slate-900 text-white"
//                       : "bg-white")
//                   }
//                   onClick={() => setSortKey("delta")}
//                 >
//                   補正幅
//                 </button>
//               </div>
//               <label className="flex items-center gap-1">
//                 <input
//                   type="checkbox"
//                   checked={showOnlyBoosted}
//                   onChange={(e) => setShowOnlyBoosted(e.target.checked)}
//                 />
//                 プラス補正のみ表示
//               </label>
//             </div>
//           </div>

//           {/* テーブル本体 */}
//           <div className="overflow-x-auto">
//             <table className="min-w-full text-xs border-collapse">
//               <thead>
//                 <tr className="bg-slate-100">
//                   <th className="border px-2 py-1 text-left">枠</th>
//                   <th className="border px-2 py-1 text-left">馬名</th>
//                   <th className="border px-2 py-1 text-left">脚質</th>
//                   <th className="border px-2 py-1 text-right">ベース勝率</th>
//                   <th className="border px-2 py-1 text-right">調整後勝率</th>
//                   <th className="border px-2 py-1 text-right">補正幅</th>
//                   <th className="border px-2 py-1 text-right">複勝(前→後)</th>
//                   <th className="border px-2 py-1 text-left">主な理由</th>
//                 </tr>
//               </thead>
//               <tbody>
//                 {horsesFiltered.map((h) => {
//                   const delta = h.win.delta;
//                   const deltaClass =
//                     delta > 0
//                       ? "text-emerald-600"
//                       : delta < 0
//                       ? "text-rose-600"
//                       : "text-slate-600";

//                   return (
//                     <tr key={h.id} className="hover:bg-slate-50">
//                       <td className="border px-2 py-1 text-center">
//                         {h.frame_no ?? "-"}
//                       </td>
//                       <td className="border px-2 py-1">{h.name}</td>
//                       <td className="border px-2 py-1">{h.run_style}</td>
//                       <td className="border px-2 py-1 text-right font-mono">
//                         {formatProb(h.win.base)}
//                       </td>
//                       <td className="border px-2 py-1 text-right font-mono">
//                         {formatProb(h.win.adj)}
//                       </td>
       
// ::contentReference[oaicite:0]{index=0}


// src/App.tsx
function App() {
  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#111827",
        color: "white",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >
      <div>
        <h1 style={{ fontSize: "24px", marginBottom: "8px" }}>
          Scenario UI Debug
        </h1>
        <p>この画面が見えていれば、React は動いています。</p>
      </div>
    </div>
  );
}

export default App;
