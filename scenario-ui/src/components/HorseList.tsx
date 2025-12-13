import React, { useMemo } from "react";
import type { HorseData, SortType } from "../types";
import HorseCard from "./HorseCard";

interface HorseListProps {
  horses: HorseData[];
  sortBy: SortType;
}

/**
 * 馬データをソートする
 */
function sortHorses(horses: HorseData[], sortBy: SortType): HorseData[] {
  const sorted = [...horses];

  switch (sortBy) {
    case "adj_win":
      // 補正後勝率の降順
      sorted.sort((a, b) => b.win.adj - a.win.adj);
      break;

    case "delta_win":
      // 変動幅の降順（絶対値ではなく、プラスが上）
      sorted.sort((a, b) => b.win.delta - a.win.delta);
      break;

    case "frame_no":
      // 枠番の昇順
      sorted.sort((a, b) => {
        const aFrame = a.frame_no ?? 999;
        const bFrame = b.frame_no ?? 999;
        return aFrame - bFrame;
      });
      break;

    case "odds":
      // オッズの昇順（人気順）
      sorted.sort((a, b) => {
        const aOdds = a.odds ?? 999;
        const bOdds = b.odds ?? 999;
        return aOdds - bOdds;
      });
      break;

    default:
      break;
  }

  return sorted;
}

/**
 * 馬リストコンポーネント
 */
function HorseList({ horses, sortBy }: HorseListProps): React.ReactElement {
  // ソート済みの馬リストをメモ化
  const sortedHorses = useMemo(() => {
    return sortHorses(horses, sortBy);
  }, [horses, sortBy]);

  if (sortedHorses.length === 0) {
    return (
      <div className="horse-list-empty">
        <p>馬データがありません</p>
      </div>
    );
  }

  return (
    <div className="horse-list">
      <h2 className="horse-list-title">
        出走馬一覧
        <span className="horse-count">（{sortedHorses.length}頭）</span>
      </h2>
      
      <div className="horse-cards">
        {sortedHorses.map((horse, index) => (
          <HorseCard
            key={horse.horse_id}
            horse={horse}
            rank={sortBy === "adj_win" ? index + 1 : undefined}
          />
        ))}
      </div>
    </div>
  );
}

export default HorseList;
