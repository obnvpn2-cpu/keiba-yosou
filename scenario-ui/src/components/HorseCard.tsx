import React from "react";
import type { HorseData } from "../types";

interface HorseCardProps {
  horse: HorseData;
  rank?: number;
}

/**
 * 確率を%表示に変換（小数点1桁）
 */
function formatPercent(value: number): string {
  return (value * 100).toFixed(1) + "%";
}

/**
 * 差分を符号付きで表示
 */
function formatDelta(value: number): string {
  const percent = (value * 100).toFixed(1);
  if (value > 0) return `+${percent}%`;
  if (value < 0) return `${percent}%`;
  return "±0%";
}

/**
 * 差分に応じたCSSクラスを返す
 */
function getDeltaClass(delta: number): string {
  if (delta > 0.02) return "delta-positive-strong";
  if (delta > 0) return "delta-positive";
  if (delta < -0.02) return "delta-negative-strong";
  if (delta < 0) return "delta-negative";
  return "delta-neutral";
}

/**
 * 枠番に応じた色クラスを返す
 */
function getFrameColorClass(frameNo: number | undefined): string {
  if (!frameNo) return "frame-unknown";
  
  const colors: Record<number, string> = {
    1: "frame-white",
    2: "frame-black",
    3: "frame-red",
    4: "frame-blue",
    5: "frame-yellow",
    6: "frame-green",
    7: "frame-orange",
    8: "frame-pink",
  };
  
  return colors[frameNo] || "frame-unknown";
}

/**
 * 馬カードコンポーネント
 */
function HorseCard({ horse, rank }: HorseCardProps): React.ReactElement {
  const deltaClass = getDeltaClass(horse.win.delta);
  const frameColorClass = getFrameColorClass(horse.frame_no);

  return (
    <div className={`horse-card ${deltaClass}`}>
      {/* ヘッダー部分 */}
      <div className="horse-card-header">
        {rank && <span className="horse-rank">#{rank}</span>}
        
        <span className={`horse-frame ${frameColorClass}`}>
          {horse.frame_no ?? "-"}枠
          {horse.horse_no && <span className="horse-number">{horse.horse_no}番</span>}
        </span>
        
        <h3 className="horse-name">{horse.name}</h3>
        
        {horse.run_style && (
          <span className="horse-run-style">{horse.run_style}</span>
        )}
      </div>

      {/* 勝率表示部分 */}
      <div className="horse-card-body">
        <div className="probability-section">
          <div className="probability-row">
            <span className="probability-label">勝率</span>
            <span className="probability-base">{formatPercent(horse.win.base)}</span>
            <span className="probability-arrow">→</span>
            <span className="probability-adj">{formatPercent(horse.win.adj)}</span>
            <span className={`probability-delta ${deltaClass}`}>
              {formatDelta(horse.win.delta)}
            </span>
          </div>
          
          <div className="probability-row">
            <span className="probability-label">複勝</span>
            <span className="probability-base">{formatPercent(horse.in3.base)}</span>
            <span className="probability-arrow">→</span>
            <span className="probability-adj">{formatPercent(horse.in3.adj)}</span>
            <span className={`probability-delta ${getDeltaClass(horse.in3.delta)}`}>
              {formatDelta(horse.in3.delta)}
            </span>
          </div>
        </div>

        {/* オッズ・人気表示 */}
        {(horse.odds || horse.popularity) && (
          <div className="odds-section">
            {horse.odds && (
              <span className="horse-odds">{horse.odds.toFixed(1)}倍</span>
            )}
            {horse.popularity && (
              <span className="horse-popularity">{horse.popularity}番人気</span>
            )}
          </div>
        )}
      </div>

      {/* 理由タグ */}
      {horse.reasons && horse.reasons.length > 0 && (
        <div className="horse-card-footer">
          <div className="reason-tags">
            {horse.reasons.map((reason, index) => (
              <span key={index} className={`reason-tag ${getReasonTagClass(reason)}`}>
                {reason}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * 理由タグのCSSクラスを返す
 */
function getReasonTagClass(reason: string): string {
  // ポジティブな理由
  if (
    reason.includes("有利") ||
    reason.includes("逃げ") ||
    reason.includes("先行") ||
    reason === "逃げ指定"
  ) {
    return "tag-positive";
  }
  
  // ネガティブな理由
  if (
    reason.includes("不利") ||
    reason.includes("信頼度低") ||
    reason === "MARKED_WEAK"
  ) {
    return "tag-negative";
  }
  
  // 中立
  return "tag-neutral";
}

export default HorseCard;
