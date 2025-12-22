# -*- coding: utf-8 -*-
# src/ingestion/parser_trainer.py
"""
調教師情報ページ（https://db.netkeiba.com/trainer/{trainer_id}/）用パーサー。

取得するカラム:
- trainer_id   : 調教師ID (5桁)
- trainer_name : 調教師名
- name_kana    : ふりがな
- birth_date   : 生年月日 (YYYY-MM-DD)
- affiliation  : 所属 (美浦/栗東)

成績情報:
- career_wins   : 通算勝利数
- career_in3    : 通算3着内数
- career_starts : 通算出走数
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class TrainerRecord:
    """調教師情報"""
    trainer_id: str
    trainer_name: Optional[str]
    name_kana: Optional[str]
    birth_date: Optional[str]
    affiliation: Optional[str]
    career_wins: Optional[int]
    career_in3: Optional[int]
    career_starts: Optional[int]


class TrainerParser:
    """
    netkeiba 調教師ページ用パーサー

    Example:
        >>> parser = TrainerParser()
        >>> soup = fetcher.fetch_soup("https://db.netkeiba.com/trainer/01234/")
        >>> record = parser.parse("01234", soup)
    """

    def parse(self, trainer_id: str, soup: BeautifulSoup) -> TrainerRecord:
        """
        調教師ページ HTML をパースして TrainerRecord を返す。

        Args:
            trainer_id: URL から分かっている調教師 ID
            soup      : BeautifulSoup オブジェクト

        Returns:
            TrainerRecord

        Raises:
            ValueError: 調教師名が取得できない場合
        """
        logger.debug("Parsing trainer page: trainer_id=%s", trainer_id)

        trainer_name = self._extract_trainer_name(soup)
        name_kana = self._extract_name_kana(soup)
        birth_date = self._extract_birth_date(soup)
        affiliation = self._extract_affiliation(soup)
        career_wins, career_in3, career_starts = self._extract_career_stats(soup)

        if not trainer_name:
            logger.error("trainer_id=%s: 調教師名を取得できませんでした", trainer_id)
            raise ValueError(f"trainer_id={trainer_id}: 調教師名を取得できませんでした")

        record = TrainerRecord(
            trainer_id=trainer_id,
            trainer_name=trainer_name,
            name_kana=name_kana,
            birth_date=birth_date,
            affiliation=affiliation,
            career_wins=career_wins,
            career_in3=career_in3,
            career_starts=career_starts,
        )

        logger.info(
            "Parsed trainer: trainer_id=%s, name=%s, affiliation=%s",
            trainer_id, trainer_name, affiliation
        )
        return record

    # ------------------------------------------------------------------
    # 抽出メソッド
    # ------------------------------------------------------------------
    def _extract_trainer_name(self, soup: BeautifulSoup) -> Optional[str]:
        """調教師名を <h1> 付近から取得する。"""
        candidates = [
            soup.select_one("div.db_head h1"),
            soup.select_one("div.db_main_box h1"),
            soup.find("title"),
            soup.find("h1"),
        ]

        for el in candidates:
            if el:
                text = el.get_text(strip=True)
                if text:
                    if el.name == "title":
                        text = re.sub(r"\s*[|｜]\s*netkeiba\.com.*", "", text)
                        text = re.sub(r"\s*の.*$", "", text)
                    if text:
                        return text
        return None

    def _extract_name_kana(self, soup: BeautifulSoup) -> Optional[str]:
        """ふりがなを抽出する。"""
        for el in soup.find_all(["span", "p", "div"]):
            text = el.get_text(strip=True)
            if text and re.match(r"^[ぁ-んァ-ヴー\s]+$", text):
                return text

        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue

            label = th.get_text(strip=True)
            if "ふりがな" in label or "フリガナ" in label or "よみ" in label:
                return td.get_text(strip=True) or None
        return None

    def _extract_birth_date(self, soup: BeautifulSoup) -> Optional[str]:
        """生年月日を抽出する。"""
        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue

            label = th.get_text(strip=True)
            if "生年月日" not in label:
                continue

            value = td.get_text(strip=True)
            if not value:
                return None

            match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", value)
            if match:
                year, month, day = match.groups()
                return f"{year}-{int(month):02d}-{int(day):02d}"

            return value
        return None

    def _extract_affiliation(self, soup: BeautifulSoup) -> Optional[str]:
        """所属を抽出する。"""
        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue

            label = th.get_text(strip=True)
            if "所属" not in label:
                continue

            value = td.get_text(strip=True)
            if not value:
                return None

            if "美浦" in value:
                return "美浦"
            elif "栗東" in value:
                return "栗東"
            return value
        return None

    def _extract_career_stats(self, soup: BeautifulSoup) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """通算成績を抽出する。"""
        for table in soup.find_all("table"):
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            if "1着" in headers or "勝利" in headers:
                for tr in table.find_all("tr"):
                    tds = tr.find_all("td")
                    if not tds:
                        continue

                    first_cell = tds[0].get_text(strip=True) if tds else ""
                    if any(kw in first_cell for kw in ["通算", "合計", "Total", "計"]):
                        return self._parse_career_row(tds)

        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue

            label = th.get_text(strip=True)
            if "通算成績" in label:
                value = td.get_text(strip=True)
                return self._parse_career_text(value)

        return None, None, None

    def _parse_career_row(self, tds) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """成績行をパースする。"""
        try:
            values = []
            for td in tds:
                text = td.get_text(strip=True).replace(",", "")
                match = re.search(r"(\d+)", text)
                if match:
                    values.append(int(match.group(1)))

            if len(values) >= 4:
                starts = values[0]
                wins = values[1]
                in3 = values[1] + values[2] + values[3]
                return wins, in3, starts
        except (ValueError, IndexError):
            pass
        return None, None, None

    def _parse_career_text(self, value: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """成績テキストをパースする。"""
        match = re.search(r"(\d+)戦(\d+)勝", value)
        if match:
            starts = int(match.group(1))
            wins = int(match.group(2))
            return wins, None, starts

        match = re.search(r"\[?(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)", value)
        if match:
            starts = int(match.group(1))
            wins = int(match.group(2))
            in3 = int(match.group(2)) + int(match.group(3)) + int(match.group(4))
            return wins, in3, starts

        return None, None, None


# ------------------------------------------------------------------
# テスト用
# ------------------------------------------------------------------
if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    print("=" * 80)
    print("TrainerParser テスト")
    print("=" * 80)

    test_html = """
    <html>
    <body>
        <div class="db_head">
            <h1>国枝栄</h1>
        </div>
        <table>
            <tr><th>生年月日</th><td>1955年4月3日</td></tr>
            <tr><th>所属</th><td>美浦</td></tr>
        </table>
    </body>
    </html>
    """

    soup = BeautifulSoup(test_html, "html.parser")
    parser = TrainerParser()

    try:
        record = parser.parse("01012", soup)
        print(f"\nパース成功:")
        print(f"  trainer_id: {record.trainer_id}")
        print(f"  trainer_name: {record.trainer_name}")
        print(f"  birth_date: {record.birth_date}")
        print(f"  affiliation: {record.affiliation}")
    except Exception as e:
        print(f"\nパース失敗: {e}")

    print("\n" + "=" * 80)
