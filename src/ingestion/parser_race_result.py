# -*- coding: utf-8 -*-
"""
RaceResultParser - Zenn「Results.scrape()」方式 + 完全な列名正規化・メタ情報取得

方針:
- HTML からのテーブル取得は pandas.read_html に丸投げ（Zenn方式）
- 列名の完全な正規化（全角/半角スペース、特殊文字除去）
- 馬ID・騎手ID などのメタ情報を抽出
- レース情報（コース、距離、馬場状態）を抽出
- データ検証の強化
"""

import re
import logging
import unicodedata
from typing import List, Dict, Optional

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class RaceResultParser:
    """
    netkeiba レース結果ページのパーサ（Zenn方式 + 拡張）
    
    Features:
        - pandas.read_html() によるシンプルなテーブル取得
        - 列名の完全な正規化（全角/半角スペース、特殊文字除去）
        - 馬ID・騎手ID の抽出（Zenn方式）
        - レース情報の抽出（コース、距離、馬場状態）
        - データ検証の強化
    
    Example:
        >>> parser = RaceResultParser()
        >>> soup = fetcher.fetch_soup("https://db.netkeiba.com/race/202301010101/")
        >>> df = parser.parse(soup, "202301010101")
    """

    # 最低限ほしいカラム（Zenn 本と同じイメージ）
    REQUIRED_COLUMNS: List[str] = ["着順", "馬名", "騎手"]

    def parse(self, soup: BeautifulSoup, race_id: str) -> pd.DataFrame:
        """
        レース結果ページをパースして DataFrame を返す

        Parameters
        ----------
        soup : BeautifulSoup
            NetkeibaFetcher で取得した BeautifulSoup オブジェクト
        race_id : str
            レースID（例: "202301010101"）

        Returns
        -------
        df : pandas.DataFrame
            レース結果のテーブル。
            必須列: 着順, 馬名, 騎手
            追加列: horse_id, jockey_id, race_id
        """
        logger.info(f"Parsing race result page for race_id={race_id}")

        # 1. pandas.read_html でテーブル取得（Zenn方式）
        df = self._parse_table(soup, race_id)

        # 2. 列名の完全な正規化
        df = self._normalize_columns(df)

        # 3. 必須列の検証
        self._validate_required_columns(df, race_id)

        # 4. 馬ID・騎手ID の抽出（Zenn方式）
        horse_ids = self._extract_horse_ids(soup)
        jockey_ids = self._extract_jockey_ids(soup)

        # 5. 馬ID・騎手ID を DataFrame に追加
        df = self._add_ids_to_dataframe(df, horse_ids, jockey_ids, race_id)

        # 6. レース情報を取得（オプショナル、失敗しても続行）
        race_info = self._extract_race_info(soup, race_id)
        
        # レース情報を DataFrame に追加
        for key, value in race_info.items():
            if key != "race_id":  # race_id は既に追加済み
                df[key] = value

        # 7. データ検証
        self._validate_dataframe(df, race_id)

        # 8. 数値列の型変換（オプショナル）
        df = self._convert_numeric_columns(df)

        logger.info(
            f"Successfully parsed race {race_id}: {len(df)} rows, {len(df.columns)} columns"
        )

        return df

    def _parse_table(self, soup: BeautifulSoup, race_id: str) -> pd.DataFrame:
        """
        pandas.read_html でテーブルを取得（Zenn方式）
        
        Args:
            soup: BeautifulSoup オブジェクト
            race_id: レースID
        
        Returns:
            パースされた DataFrame
        """
        html_str = str(soup)

        try:
            tables = pd.read_html(html_str)
        except ValueError as e:
            logger.error(f"Race {race_id}: pd.read_html でテーブルを取得できませんでした: {e}")
            raise ValueError(f"pd.read_html でテーブルを取得できませんでした: {e}")

        if not tables:
            logger.error(f"Race {race_id}: HTML 内にテーブルが見つかりませんでした")
            raise ValueError("HTML 内にテーブルが見つかりませんでした")

        # Zenn 本と同様、「一番最初のテーブル = 全着順テーブル」とみなす
        df = tables[0]

        if df.empty:
            logger.error(f"Race {race_id}: レース結果テーブルが空です")
            raise ValueError("レース結果テーブルが空です（スクレイピング失敗）")

        logger.debug(f"Race {race_id}: Parsed table with {len(df)} rows, {len(df.columns)} columns")

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        列名を完全に正規化（Zenn方式 + 拡張）
        
        正規化内容:
        - Unicode 正規化（全角→半角）
        - 全てのスペース（半角・全角・タブ・改行）を除去
        - 特殊文字（括弧など）を除去
        
        Args:
            df: 元の DataFrame
        
        Returns:
            正規化された DataFrame
        """
        original_columns = df.columns.tolist()

        # 1. Unicode 正規化（全角→半角）
        df.columns = df.columns.map(lambda x: unicodedata.normalize('NFKC', str(x)))

        # 2. 全てのスペース・タブ・改行を除去（Zenn方式 + 拡張）
        df.columns = df.columns.str.replace(' ', '')    # 半角スペース
        df.columns = df.columns.str.replace('　', '')   # 全角スペース
        df.columns = df.columns.str.replace('\u3000', '')  # 全角スペース（Unicode）
        df.columns = df.columns.str.replace('\t', '')   # タブ
        df.columns = df.columns.str.replace('\n', '')   # 改行
        df.columns = df.columns.str.replace('\r', '')   # キャリッジリターン

        # 3. 特殊文字を除去
        df.columns = df.columns.str.replace('(', '').str.replace(')', '')
        df.columns = df.columns.str.replace('（', '').str.replace('）', '')
        df.columns = df.columns.str.replace('[', '').str.replace(']', '')
        df.columns = df.columns.str.replace('【', '').str.replace('】', '')

        logger.debug(f"Normalized columns: {original_columns} -> {df.columns.tolist()}")

        return df

    def _validate_required_columns(self, df: pd.DataFrame, race_id: str):
        """
        必須列の存在をチェック
        
        Args:
            df: チェックする DataFrame
            race_id: レースID
        
        Raises:
            ValueError: 必須列が不足している場合
        """
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            logger.error(
                f"Race {race_id}: 必須列が不足しています: {missing} / "
                f"取得カラム: {list(df.columns)}"
            )
            raise ValueError(f"必須列が不足しています: {missing} / 取得カラム: {list(df.columns)}")

    def _extract_horse_ids(self, soup: BeautifulSoup) -> List[str]:
        """
        馬IDを抽出（Zenn方式）
        
        Zenn 本の実装:
        ```python
        horse_id_list = []
        for a in soup.find_all("a", attrs={"href": re.compile("^/horse")}):
            horse_id = re.findall(r"\d+", a["href"])[0]
            horse_id_list.append(horse_id)
        ```
        
        Args:
            soup: BeautifulSoup オブジェクト
        
        Returns:
            馬ID のリスト（10桁の数字）
        """
        horse_id_list = []

        # /horse/XXXXXXXXXX/ のパターンを探す
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/horse/" in href:
                # 10桁の数字を抽出
                match = re.search(r'/horse/(\d{10})/', href)
                if match:
                    horse_id = match.group(1)
                    horse_id_list.append(horse_id)

        logger.debug(f"Extracted {len(horse_id_list)} horse IDs")

        return horse_id_list

    def _extract_jockey_ids(self, soup: BeautifulSoup) -> List[str]:
        """
        騎手IDを抽出
        
        Args:
            soup: BeautifulSoup オブジェクト
        
        Returns:
            騎手ID のリスト（5桁の数字）
        """
        jockey_id_list = []

        # /jockey/XXXXX/ のパターンを探す
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/jockey/" in href:
                # 5桁の数字を抽出
                match = re.search(r'/jockey/(\d{5})/', href)
                if match:
                    jockey_id = match.group(1)
                    jockey_id_list.append(jockey_id)

        logger.debug(f"Extracted {len(jockey_id_list)} jockey IDs")

        return jockey_id_list

    def _add_ids_to_dataframe(
        self,
        df: pd.DataFrame,
        horse_ids: List[str],
        jockey_ids: List[str],
        race_id: str
    ) -> pd.DataFrame:
        """
        馬ID・騎手ID を DataFrame に追加
        
        Args:
            df: 元の DataFrame
            horse_ids: 馬ID のリスト
            jockey_ids: 騎手ID のリスト
            race_id: レースID
        
        Returns:
            ID が追加された DataFrame
        """
        # 行数チェック
        if len(horse_ids) != len(df):
            logger.warning(
                f"Race {race_id}: 馬ID の数({len(horse_ids)})と "
                f"テーブルの行数({len(df)})が一致しません"
            )
            # 足りない場合は None で埋める
            while len(horse_ids) < len(df):
                horse_ids.append(None)
            # 多い場合は切り捨て
            horse_ids = horse_ids[:len(df)]

        if len(jockey_ids) != len(df):
            logger.warning(
                f"Race {race_id}: 騎手ID の数({len(jockey_ids)})と "
                f"テーブルの行数({len(df)})が一致しません"
            )
            while len(jockey_ids) < len(df):
                jockey_ids.append(None)
            jockey_ids = jockey_ids[:len(df)]

        # DataFrame に追加
        df["horse_id"] = horse_ids
        df["jockey_id"] = jockey_ids
        df["race_id"] = race_id

        return df

    def _extract_race_info(self, soup: BeautifulSoup, race_id: str) -> Dict[str, any]:
        """
        レース情報を抽出（コース、距離、馬場状態など）
        
        Args:
            soup: BeautifulSoup オブジェクト
            race_id: レースID
        
        Returns:
            レース情報の辞書
        """
        info = {"race_id": race_id}

        try:
            # レース名
            race_name_tag = soup.select_one("h1")
            if race_name_tag:
                info["race_name"] = race_name_tag.text.strip()

            # レース情報エリア
            diary_data = soup.select_one("p.smalltxt")
            if diary_data:
                text = diary_data.text

                # 芝/ダート
                if "芝" in text:
                    info["course_type"] = "芝"
                elif "ダ" in text:
                    info["course_type"] = "ダート"

                # 距離
                match = re.search(r'(\d{3,4})m', text)
                if match:
                    info["distance"] = int(match.group(1))

                # 馬場状態
                match = re.search(r'馬場[:：\s]*([良稍重不]{1})', text)
                if match:
                    info["track_condition"] = match.group(1)

            logger.debug(f"Race {race_id}: Extracted race info: {info}")

        except Exception as e:
            logger.warning(f"Race {race_id}: レース情報の抽出に失敗: {e}")

        return info

    def _validate_dataframe(self, df: pd.DataFrame, race_id: str):
        """
        データの妥当性を検証
        
        Args:
            df: 検証する DataFrame
            race_id: レースID
        """
        # 行数チェック（最低3頭）
        if len(df) < 3:
            logger.warning(f"Race {race_id}: 出走頭数が少ない ({len(df)} 頭)")

        # 着順の妥当性チェック
        if "着順" in df.columns:
            # 数値に変換できるかチェック（「中止」「除外」などがある場合の対応）
            valid_chakujun = pd.to_numeric(df["着順"], errors='coerce')
            num_valid = valid_chakujun.notna().sum()

            if num_valid == 0:
                logger.error(f"Race {race_id}: 着順データが全て無効です")
                raise ValueError(f"Race {race_id}: 着順データが全て無効です")

            if num_valid < len(df) * 0.8:
                logger.warning(
                    f"Race {race_id}: 着順データの欠損が多い "
                    f"({len(df) - num_valid}/{len(df)} 頭が無効)"
                )

        logger.debug(f"Race {race_id}: Validation passed")

    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数値列を適切な型に変換
        
        Args:
            df: 元の DataFrame
        
        Returns:
            型変換された DataFrame
        """
        # 着順（「中止」「除外」は NaN に）
        if "着順" in df.columns:
            df["着順_数値"] = pd.to_numeric(df["着順"], errors='coerce')

        # オッズ
        if "単勝" in df.columns:
            df["単勝_数値"] = pd.to_numeric(df["単勝"], errors='coerce')

        # タイム（"1:23.4" → 83.4 秒に変換）
        if "タイム" in df.columns:
            def time_to_seconds(time_str):
                if pd.isna(time_str) or time_str == "":
                    return None
                try:
                    parts = str(time_str).split(":")
                    if len(parts) == 2:
                        minutes = int(parts[0])
                        seconds = float(parts[1])
                        return minutes * 60 + seconds
                except:
                    return None

            df["タイム秒"] = df["タイム"].apply(time_to_seconds)

        return df
