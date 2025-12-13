# -*- coding: utf-8 -*-
"""
parser_horse_past.py

netkeiba 馬の成績ページ（過去走）パーサ
- URL 例: https://db.netkeiba.com/horse/result/2021101429/

改善点:
- StringIO を使って FutureWarning を回避
- race_id 抽出の堅牢性を向上（レース名カラムの位置を動的に特定）
- カラム名正規化の強化
- エラーハンドリングの改善
- 型変換の安全性向上
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Optional, Tuple
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class HorsePastRunsParser:
    """
    馬の過去走成績テーブルを DataFrame に変換するパーサ。

    主な仕様:
    - BeautifulSoup で取得した `soup` から、該当テーブル要素 (<table>〜) を特定
    - pandas.read_html でテーブルを読み込み、日本語カラム名を正規化
    - 1 行 = 1 レースの成績として、特徴量を整形
    - horse_id を列として付与
    
    改善点:
    - FutureWarning 対応（StringIO 使用）
    - race_id 抽出の堅牢性向上
    - エラーハンドリング強化
    """

    # 正規化後の必須カラム
    REQUIRED_COLUMNS = ["日付", "レース名", "着順", "距離", "馬体重"]

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def parse(self, soup: BeautifulSoup, horse_id: str) -> pd.DataFrame:
        """
        馬の過去走成績テーブルをパースし、1 レコード 1 レースの DataFrame を返す。

        Parameters
        ----------
        soup : BeautifulSoup
            requests + BeautifulSoup で取得した HTML ツリー
        horse_id : str
            対象馬の horse_id

        Returns
        -------
        pd.DataFrame
            過去走成績テーブル。
            主なカラム:
              - horse_id, race_id, race_date, place, weather, race_number, race_name
              - num_head, waku, umaban, odds, popularity, finish, jockey, weight_carried
              - surface, distance, track_condition, time, time_seconds, time_diff
              - pace_front_3f, pace_back_3f, passing, last_3f
              - horse_weight, horse_weight_diff, winner_name, prize
        """
        logger.info("Parsing horse past runs: horse_id=%s", horse_id)
        
        table = self._extract_past_runs_table(soup)
        if table is None:
            raise ValueError(f"horse_id={horse_id}: 過去走成績テーブルが見つかりませんでした")

        # read_html で DataFrame 化（FutureWarning 対応）
        html_str = str(table)
        logger.debug("Parsing horse past runs table via pandas.read_html")
        
        # StringIO を使って FutureWarning を回避
        tables = pd.read_html(StringIO(html_str))
        
        if not tables:
            raise ValueError(f"horse_id={horse_id}: pandas.read_html でテーブルを取得できませんでした")

        df = tables[0]
        if df.empty:
            raise ValueError(f"horse_id={horse_id}: 過去走成績テーブルが空です")

        # カラム名を正規化
        df = self._normalize_columns(df)

        # 必須カラムチェック
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"horse_id={horse_id}: 必須列が不足しています: {missing} / "
                f"取得カラム: {list(df.columns)}"
            )

        # race_id を HTML から抽出（レース名列の a href）
        race_ids = self._extract_race_ids_from_table(table, df)
        if race_ids and len(race_ids) != len(df):
            logger.warning(
                "horse_id=%s: race_id の数(%d)とテーブル行数(%d)が一致しません。race_id を None で埋めます",
                horse_id,
                len(race_ids),
                len(df),
            )
            race_ids = [None] * len(df)

        # 最終的な DataFrame を構築
        out = self._build_output_dataframe(df, horse_id, race_ids)

        logger.info(
            "Successfully parsed horse past runs: horse_id=%s, rows=%d, cols=%d",
            horse_id,
            len(out),
            out.shape[1],
        )
        return out

    # ------------------------------------------------------------------
    # DataFrame 構築
    # ------------------------------------------------------------------
    def _build_output_dataframe(
        self,
        df: pd.DataFrame,
        horse_id: str,
        race_ids: Optional[List[str]],
    ) -> pd.DataFrame:
        """最終的な出力 DataFrame を構築"""
        out = pd.DataFrame()
        # horse_id は全行に同じ値を設定（リストで渡す）
        out["horse_id"] = [horse_id] * len(df)
        out["race_id"] = race_ids if race_ids is not None else [None] * len(df)

        out["race_date"] = self._safe_str_series(df.get("日付"))
        out["place"] = self._safe_str_series(df.get("開催"))
        out["weather"] = self._safe_str_series(df.get("天気"))
        out["race_number"] = self._to_int_series(df.get("R"))
        out["race_name"] = self._safe_str_series(df.get("レース名"))

        out["num_head"] = self._to_int_series(df.get("頭数"))
        out["waku"] = self._to_int_series(df.get("枠番"))
        out["umaban"] = self._to_int_series(df.get("馬番"))

        out["odds"] = self._to_float_series(df.get("オッズ"))
        out["popularity"] = self._to_int_series(df.get("人気"))
        out["finish"] = self._to_int_series(df.get("着順"))

        out["jockey"] = self._safe_str_series(df.get("騎手"))
        out["weight_carried"] = self._to_float_series(df.get("斤量"))

        # 距離 (例: 芝1200, ダ1600, 障芝2890 など) を surface / distance に分割
        surface, distance = self._split_surface_distance(df.get("距離"))
        out["surface"] = surface
        out["distance"] = distance

        out["track_condition"] = self._safe_str_series(df.get("馬場"))

        out["time"] = self._safe_str_series(df.get("タイム"))
        out["time_seconds"] = self._to_time_seconds(out["time"])

        out["time_diff"] = self._safe_str_series(df.get("着差"))

        # ペース (例: "34.0-33.2") -> 前半3F / 後半3F
        pace_front, pace_back = self._split_pace(df.get("ペース"))
        out["pace_front_3f"] = pace_front
        out["pace_back_3f"] = pace_back

        out["passing"] = self._safe_str_series(df.get("通過"))
        out["last_3f"] = self._to_float_series(df.get("上り"))

        # 馬体重 "450(+2)" -> weight, diff
        hw, hw_diff = self._split_horse_weight(df.get("馬体重"))
        out["horse_weight"] = hw
        out["horse_weight_diff"] = hw_diff

        # 勝ち馬(2着馬) -> 勝ち馬名だけを取り出す
        winner_col_name = next((c for c in df.columns if "勝ち馬" in c), None)
        if winner_col_name:
            out["winner_name"] = self._safe_str_series(df[winner_col_name])
        else:
            out["winner_name"] = None

        out["prize"] = self._to_float_series(df.get("賞金"))

        # 余計なインデックスをリセット
        out.reset_index(drop=True, inplace=True)

        return out

    # ------------------------------------------------------------------
    # table 抽出
    # ------------------------------------------------------------------
    def _extract_past_runs_table(self, soup: BeautifulSoup):
        """
        馬の成績テーブル (<table>) 要素を返す。

        - thead 内に「日付」「レース名」「着順」を含む table を探す。
        - 複数のパターンに対応
        """
        tables = soup.find_all("table")
        
        for table in tables:
            thead = table.find("thead")
            if not thead:
                continue
            
            # thead 内の全テキストを取得（改行・空白除去）
            headers = []
            for th in thead.find_all("th"):
                text = th.get_text(strip=True)
                # <br> タグなども処理
                text = text.replace("\n", "").replace("\r", "")
                headers.append(text)
            
            header_text = "".join(headers)
            
            # 必須キーワードをチェック
            required_keywords = ["日付", "レース", "着順"]
            if all(keyword in header_text for keyword in required_keywords):
                logger.debug("Found past runs table with headers: %s", headers)
                return table
        
        logger.warning("Past runs table not found in HTML")
        return None

    # ------------------------------------------------------------------
    # race_id 抽出
    # ------------------------------------------------------------------
    def _extract_race_ids_from_table(
        self,
        table,
        df: pd.DataFrame,
    ) -> Optional[List[str]]:
        """
        テーブルの「レース名」列に含まれるリンクから race_id を抽出する。
        
        改善点:
        - レース名カラムの位置を動的に特定
        - ハードコードを回避

        race URL の例:
          - /race/202301010101/
          - https://db.netkeiba.com/race/202301010101/
        """
        tbody = table.find("tbody")
        if not tbody:
            logger.debug("tbody not found in table")
            return None

        # レース名カラムのインデックスを特定
        race_name_idx = None
        if "レース名" in df.columns:
            race_name_idx = df.columns.tolist().index("レース名")
        else:
            logger.warning("レース名カラムが見つかりません")
            return None

        race_ids: List[str] = []
        for tr in tbody.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) <= race_name_idx:
                race_ids.append(None)
                continue
            
            race_td = tds[race_name_idx]
            a = race_td.find("a")
            if not a:
                race_ids.append(None)
                continue
            
            href = a.get("href", "")
            # race_id を抽出（12桁の数字）
            m = re.search(r"/race/(\d{12})", href)
            if not m:
                # 旧形式も試す（10桁）
                m = re.search(r"/race/(\d{10})", href)
            
            race_ids.append(m.group(1) if m else None)

        if not race_ids:
            logger.debug("No race_ids extracted")
            return None
        
        logger.debug("Extracted %d race_ids", len(race_ids))
        return race_ids

    # ------------------------------------------------------------------
    # column 正規化
    # ------------------------------------------------------------------
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        日本語カラム名のゆらぎ（改行・全角半角・空白など）を吸収する。
        """
        new_cols = []
        for col in df.columns:
            new_cols.append(self._normalize_column_name(str(col)))
        df = df.copy()
        df.columns = new_cols
        logger.debug("Normalized columns: %s", list(df.columns))
        return df

    @staticmethod
    def _normalize_column_name(col: str) -> str:
        """
        カラム名の正規化:
        - 全角 -> 半角 (NFKC)
        - 改行・タブ・スペースの削除
        - 全角スペース削除
        - 括弧削除
        """
        # Unicode 正規化
        col = unicodedata.normalize("NFKC", col)
        # 改行・タブ除去
        col = col.replace("\n", "").replace("\r", "").replace("\t", "")
        # スペース除去（半角・全角）
        col = col.replace(" ", "").replace("\u3000", "")
        # 括弧除去
        col = col.replace("(", "").replace(")", "")
        col = col.replace("（", "").replace("）", "")
        # その他の記号
        col = col.replace("[", "").replace("]", "")
        col = col.replace("【", "").replace("】", "")
        return col

    # ------------------------------------------------------------------
    # helper: Series 変換系
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_str_series(s: Optional[pd.Series]) -> Optional[pd.Series]:
        """安全に文字列化（None や NaN は None のまま）"""
        if s is None:
            return None
        
        def _to_str(x):
            if x is None or pd.isna(x):
                return None
            s_val = str(x).strip()
            if s_val == "" or s_val.lower() in ("nan", "none"):
                return None
            return s_val
        
        return s.map(_to_str)

    @staticmethod
    def _to_int_series(s: Optional[pd.Series]) -> Optional[pd.Series]:
        """安全に int に変換"""
        if s is None:
            return None
        # 'Int64' ではなく通常の int に変換（SQLite 互換性のため）
        result = pd.to_numeric(s, errors="coerce")
        # NaN は None に変換
        return result.where(result.notna(), None).astype(object)

    @staticmethod
    def _to_float_series(s: Optional[pd.Series]) -> Optional[pd.Series]:
        """安全に float に変換"""
        if s is None:
            return None
        result = pd.to_numeric(s, errors="coerce")
        # NaN は None に変換
        return result.where(result.notna(), None)

    @staticmethod
    def _to_time_seconds(time_series: Optional[pd.Series]) -> Optional[pd.Series]:
        """
        "1:09.5" 形式のタイムを秒(float)に変換する。
        """
        if time_series is None:
            return None

        def _one(x: object) -> Optional[float]:
            if x is None or pd.isna(x):
                return None
            s = str(x).strip()
            if not s or s in ("0", "-", "nan", "NaN", "None"):
                return None
            try:
                if ":" in s:
                    parts = s.split(":")
                    if len(parts) != 2:
                        return None
                    m, sec = parts
                    return float(m) * 60.0 + float(sec)
                return float(s)
            except Exception as e:
                logger.debug("Failed to convert time to seconds: %s, error=%s", s, e)
                return None

        return time_series.map(_one)

    # ------------------------------------------------------------------
    # helper: 距離, ペース, 馬体重
    # ------------------------------------------------------------------
    @staticmethod
    def _split_surface_distance(s: Optional[pd.Series]) -> Tuple[pd.Series, pd.Series]:
        """
        距離カラム (例: "芝1200", "ダ1600", "障芝2890") を
        surface (芝/ダ/障など) と distance(int) に分解する。
        """
        if s is None:
            return pd.Series([None]), pd.Series([None])
        
        surfaces: List[Optional[str]] = []
        distances: List[Optional[int]] = []

        for v in s:
            if v is None or pd.isna(v):
                surfaces.append(None)
                distances.append(None)
                continue
            
            text = str(v).strip()
            if not text:
                surfaces.append(None)
                distances.append(None)
                continue
            
            # 例: "芝1200", "ダ1600", "障芝2890", "芝外1600"
            m = re.match(r"([^\d]+)?(\d+)", text)
            if not m:
                surfaces.append(None)
                distances.append(None)
                continue
            
            surface_text = m.group(1)
            if surface_text:
                # "芝外" -> "芝" のような処理
                surface_text = surface_text.replace("外", "").replace("内", "")
            surfaces.append(surface_text)
            
            try:
                distances.append(int(m.group(2)))
            except Exception:
                distances.append(None)

        return pd.Series(surfaces), pd.Series(distances)

    @staticmethod
    def _split_pace(pace_series: Optional[pd.Series]) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """
        ペースカラム "34.0-33.2" を前半3F/後半3Fに分割する。
        """
        if pace_series is None:
            return None, None

        front_list: List[Optional[float]] = []
        back_list: List[Optional[float]] = []

        for v in pace_series:
            if v is None or pd.isna(v):
                front_list.append(None)
                back_list.append(None)
                continue
            
            s = str(v).strip()
            if not s or "-" not in s:
                front_list.append(None)
                back_list.append(None)
                continue
            
            parts = s.split("-")
            if len(parts) != 2:
                front_list.append(None)
                back_list.append(None)
                continue
            
            front, back = parts
            try:
                front_list.append(float(front.strip()))
            except Exception:
                front_list.append(None)
            try:
                back_list.append(float(back.strip()))
            except Exception:
                back_list.append(None)

        return pd.Series(front_list), pd.Series(back_list)

    @staticmethod
    def _split_horse_weight(hw_series: Optional[pd.Series]) -> Tuple[pd.Series, pd.Series]:
        """
        馬体重カラム "450(+2)" などを (450, +2) に分割する。
        """
        if hw_series is None:
            return pd.Series([None]), pd.Series([None])
        
        weights: List[Optional[int]] = []
        diffs: List[Optional[int]] = []

        for v in hw_series:
            if v is None or pd.isna(v):
                weights.append(None)
                diffs.append(None)
                continue
            
            s = str(v).strip()
            if not s:
                weights.append(None)
                diffs.append(None)
                continue

            # 例: "450(+2)", "450(-4)", "450(0)"
            m = re.match(r"(\d+)\(([-+]?\d+)\)", s)
            if m:
                try:
                    weights.append(int(m.group(1)))
                except Exception:
                    weights.append(None)
                try:
                    diffs.append(int(m.group(2)))
                except Exception:
                    diffs.append(None)
                continue

            # 括弧が無い場合は重さだけ
            m2 = re.match(r"(\d+)", s)
            if m2:
                try:
                    weights.append(int(m2.group(1)))
                except Exception:
                    weights.append(None)
                diffs.append(None)
                continue

            weights.append(None)
            diffs.append(None)

        return pd.Series(weights), pd.Series(diffs)