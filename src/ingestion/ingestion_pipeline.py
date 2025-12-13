# -*- coding: utf-8 -*-
"""
Ingestion Pipeline（完全版）

改善点:
- 複数レース取得機能
- エラーハンドリングの強化
- 一括保存機能
- プログレスバー表示
- ログ出力の詳細化
"""

import os
import logging
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from .fetcher import NetkeibaFetcher
from .parser_race_result import RaceResultParser

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    netkeiba データ取得パイプライン（完全版）
    
    Features:
        - 単一レース取得
        - 複数レース一括取得
        - エラーハンドリング（skip_on_error）
        - プログレスバー表示
        - 一括保存機能
        - Context manager 対応
    
    Example:
        >>> with IngestionPipeline() as pipeline:
        ...     df = pipeline.scrape_race_result("202301010101")
        ...     pipeline.save_race_csv(df, "202301010101")
        
        # 複数レース取得
        >>> with IngestionPipeline() as pipeline:
        ...     race_ids = ["202301010101", "202301010201", "202301010301"]
        ...     results = pipeline.scrape_multiple_races(race_ids)
        ...     pipeline.save_multiple_as_csv(results)
    """

    def __init__(
        self,
        output_dir: str = "./data",
        sleep_min: float = 2.0,
        sleep_max: float = 3.5,
        max_retry: int = 3
    ):
        """
        Args:
            output_dir: 出力ディレクトリ
            sleep_min: リクエスト間隔の最小値（秒）
            sleep_max: リクエスト間隔の最大値（秒）
            max_retry: リトライ回数
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.fetcher = NetkeibaFetcher(
            sleep_min=sleep_min,
            sleep_max=sleep_max,
            max_retry=max_retry
        )
        self.race_parser = RaceResultParser()

        logger.info(f"IngestionPipeline initialized: output_dir={output_dir}")

    def scrape_race_result(self, race_id: str) -> pd.DataFrame:
        """
        単一レースを取得
        
        Args:
            race_id: レースID（12桁の数字）
        
        Returns:
            レース結果の DataFrame
        
        Raises:
            RuntimeError: 取得に失敗した場合
        
        Example:
            >>> df = pipeline.scrape_race_result("202301010101")
        """
        url = f"https://db.netkeiba.com/race/{race_id}/"
        logger.info(f"Scraping race {race_id} (url={url})")

        try:
            # HTML を取得
            soup = self.fetcher.fetch_soup(url)

            # パース
            df = self.race_parser.parse(soup, race_id)

            logger.info(
                f"Successfully scraped race {race_id}: "
                f"{len(df)} rows, {len(df.columns)} columns"
            )

            return df

        except Exception as e:
            logger.error(f"Failed to scrape race {race_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to scrape race {race_id}: {e}")

    def scrape_multiple_races(
        self,
        race_ids: List[str],
        show_progress: bool = True,
        skip_on_error: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        複数レースを一括取得
        
        Args:
            race_ids: レースID のリスト
            show_progress: プログレスバーを表示するか
            skip_on_error: エラー時にスキップするか（False の場合は中断）
        
        Returns:
            race_id -> DataFrame の辞書
        
        Example:
            >>> race_ids = ["202301010101", "202301010201", "202301010301"]
            >>> results = pipeline.scrape_multiple_races(race_ids)
            >>> print(f"成功: {len(results)}/{len(race_ids)} レース")
        """
        results = {}
        failed = []

        logger.info(f"Scraping {len(race_ids)} races")

        # プログレスバー
        iterator = tqdm(race_ids, desc="Scraping races") if show_progress else race_ids

        for race_id in iterator:
            try:
                df = self.scrape_race_result(race_id)
                results[race_id] = df

            except Exception as e:
                logger.error(f"Failed to scrape {race_id}: {e}")
                failed.append(race_id)

                if not skip_on_error:
                    logger.error("Stopping due to error (skip_on_error=False)")
                    break

        logger.info(
            f"Scraping completed: {len(results)}/{len(race_ids)} races succeeded, "
            f"{len(failed)} failed"
        )

        if failed:
            logger.warning(f"Failed races: {failed}")

        return results

    def save_race_csv(self, df: pd.DataFrame, race_id: str):
        """
        単一レース結果を CSV として保存
        
        Args:
            df: 保存する DataFrame
            race_id: レースID
        
        Example:
            >>> pipeline.save_race_csv(df, "202301010101")
        """
        path = os.path.join(self.output_dir, f"race_{race_id}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved: {path} ({len(df)} rows)")

    def save_multiple_as_csv(
        self,
        results: Dict[str, pd.DataFrame],
        output_dir: Optional[str] = None
    ):
        """
        複数レース結果を CSV として一括保存
        
        Args:
            results: race_id -> DataFrame の辞書
            output_dir: 保存先ディレクトリ（省略時は self.output_dir）
        
        Example:
            >>> results = pipeline.scrape_multiple_races(race_ids)
            >>> pipeline.save_multiple_as_csv(results)
        """
        if output_dir is None:
            output_dir = self.output_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(results)} CSVs to {output_dir}")

        for race_id, df in results.items():
            csv_path = output_path / f"race_{race_id}.csv"
            df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
            logger.debug(f"Saved: {csv_path} ({len(df)} rows)")

        logger.info(f"Saved {len(results)} CSVs")

    def close(self):
        """リソースをクローズ"""
        logger.info("Closing session")
        self.fetcher.close()

    def __enter__(self):
        """Context manager: with 文で使用可能"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager: 自動クローズ"""
        self.close()
        return False

    def __del__(self):
        """デストラクタ: 念のためクローズ"""
        try:
            self.close()
        except:
            pass


# ------------------------------------------------------------------
# デモ実行（python -m src.ingestion.ingestion_pipeline）
# ------------------------------------------------------------------
def main():
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("ingestion.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger.info("=" * 80)
    logger.info("Ingestion Pipeline - デモ")
    logger.info("=" * 80)

    with IngestionPipeline() as pipeline:
        # 単一レース取得
        logger.info("\n【単一レース取得】")
        race_id = "202301010101"

        try:
            df = pipeline.scrape_race_result(race_id)

            print(f"\n取得成功:")
            print(f"  race_id: {race_id}")
            print(f"  行数: {len(df)}")
            print(f"  列数: {len(df.columns)}")
            print(f"\n列名: {df.columns.tolist()}")
            print(f"\n先頭5行:")
            print(df.head())

            # CSV 保存
            pipeline.save_race_csv(df, race_id)

        except Exception as e:
            logger.error(f"Failed to scrape {race_id}: {e}")

        # 複数レース取得
        logger.info("\n【複数レース取得】")
        race_ids = [
            "202301010101",
            "202301010201",
            "202301010301",
        ]

        try:
            results = pipeline.scrape_multiple_races(race_ids)

            print(f"\n取得結果:")
            print(f"  成功: {len(results)}/{len(race_ids)} レース")

            # 一括保存
            pipeline.save_multiple_as_csv(results)

        except Exception as e:
            logger.error(f"Failed to scrape multiple races: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("完了")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
