# -*- coding: utf-8 -*-
# src/ingestion/parser_horse_extended.py
"""
馬の詳細情報ページ（https://db.netkeiba.com/horse/{horse_id}/）用パーサー。

HorseBasicParser を拡張し、血統情報を含む完全な馬データを抽出する。

取得するカラム:
- horse_id
- horse_name
- sex         : "牡" / "牝" / "セン"
- birth_date  : 生年月日 (YYYY-MM-DD or YYYY)
- coat_color  : 毛色
- breeder     : 生産者名
- breeder_region : 生産地域
- owner       : 馬主名
- owner_id    : 馬主ID

血統情報:
- sire_id     : 父のhorse_id
- sire_name   : 父の名前
- dam_id      : 母のhorse_id
- dam_name    : 母の名前
- broodmare_sire_id   : 母父のhorse_id
- broodmare_sire_name : 母父の名前
- sire_sire_name      : 父父の名前
- sire_dam_name       : 父母の名前
- dam_dam_name        : 母母の名前

成績情報:
- total_prize  : 総獲得賞金
- total_starts : 通算出走数
- total_wins   : 通算勝利数
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging
import re

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


@dataclass
class HorseExtendedRecord:
    """馬の詳細情報（血統込み）"""
    horse_id: str
    horse_name: Optional[str]
    sex: Optional[str]
    birth_date: Optional[str]
    coat_color: Optional[str]
    breeder: Optional[str]
    breeder_region: Optional[str]
    owner: Optional[str]
    owner_id: Optional[str]

    # Pedigree (血統)
    sire_id: Optional[str]
    sire_name: Optional[str]
    dam_id: Optional[str]
    dam_name: Optional[str]
    broodmare_sire_id: Optional[str]
    broodmare_sire_name: Optional[str]
    sire_sire_name: Optional[str]
    sire_dam_name: Optional[str]
    dam_dam_name: Optional[str]

    # Stats
    total_prize: Optional[int]
    total_starts: Optional[int]
    total_wins: Optional[int]


class HorseExtendedParser:
    """
    netkeiba 馬ページ用パーサー（拡張版）

    血統情報を含む完全な馬データを抽出する。

    Example:
        >>> parser = HorseExtendedParser()
        >>> soup = fetcher.fetch_soup("https://db.netkeiba.com/horse/2020104385/")
        >>> record = parser.parse("2020104385", soup)
    """

    # 性別抽出用の正規表現パターン
    SEX_PATTERNS = {
        "牡": re.compile(r"^牡"),
        "牝": re.compile(r"^牝"),
        "セン": re.compile(r"^セン"),
    }

    # 馬IDを抽出する正規表現
    HORSE_ID_PATTERN = re.compile(r"/horse/(\d+)")
    OWNER_ID_PATTERN = re.compile(r"/owner/([^/]+)")

    def parse(self, horse_id: str, soup: BeautifulSoup) -> HorseExtendedRecord:
        """
        馬ページ HTML をパースして HorseExtendedRecord を返す。

        Args:
            horse_id: URL から分かっている馬 ID（10桁想定）
            soup    : BeautifulSoup オブジェクト

        Returns:
            HorseExtendedRecord

        Raises:
            ValueError: 馬名が取得できない場合（致命的エラー）
        """
        logger.debug("Parsing horse page (extended): horse_id=%s", horse_id)

        # 基本情報
        horse_name = self._extract_horse_name(soup)
        sex = self._extract_sex(soup)
        birth_date = self._extract_birth_date(soup)
        coat_color = self._extract_coat_color(soup)
        breeder = self._extract_breeder(soup)
        breeder_region = self._extract_breeder_region(soup)
        owner, owner_id = self._extract_owner(soup)

        # 血統情報
        pedigree = self._extract_pedigree(soup)

        # 成績情報
        total_prize = self._extract_total_prize(soup)
        total_starts, total_wins = self._extract_career_stats(soup)

        # 馬名は必須
        if not horse_name:
            logger.error("horse_id=%s: 馬名を取得できませんでした（致命的エラー）", horse_id)
            raise ValueError(f"horse_id={horse_id}: 馬名を取得できませんでした")

        record = HorseExtendedRecord(
            horse_id=horse_id,
            horse_name=horse_name,
            sex=sex,
            birth_date=birth_date,
            coat_color=coat_color,
            breeder=breeder,
            breeder_region=breeder_region,
            owner=owner,
            owner_id=owner_id,
            sire_id=pedigree.get("sire_id"),
            sire_name=pedigree.get("sire_name"),
            dam_id=pedigree.get("dam_id"),
            dam_name=pedigree.get("dam_name"),
            broodmare_sire_id=pedigree.get("broodmare_sire_id"),
            broodmare_sire_name=pedigree.get("broodmare_sire_name"),
            sire_sire_name=pedigree.get("sire_sire_name"),
            sire_dam_name=pedigree.get("sire_dam_name"),
            dam_dam_name=pedigree.get("dam_dam_name"),
            total_prize=total_prize,
            total_starts=total_starts,
            total_wins=total_wins,
        )

        logger.info(
            "Parsed horse extended: horse_id=%s, name=%s, sire=%s, dam=%s",
            horse_id, horse_name, pedigree.get("sire_name"), pedigree.get("dam_name")
        )
        return record

    # ------------------------------------------------------------------
    # 基本情報抽出
    # ------------------------------------------------------------------
    def _extract_horse_name(self, soup: BeautifulSoup) -> Optional[str]:
        """馬名を <h1> 付近から取得する。"""
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
                    if text:
                        return text
        return None

    def _extract_sex(self, soup: BeautifulSoup) -> Optional[str]:
        """性齢(牡3 など) から性別だけを抜き出す。"""
        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue

            label = th.get_text(strip=True)
            if not any(keyword in label for keyword in ["性齢", "性別", "性"]):
                continue

            value = td.get_text(strip=True)
            if not value:
                return None

            for sex_name, pattern in self.SEX_PATTERNS.items():
                if pattern.search(value):
                    return sex_name
            return None
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

            # パターン: 2020年3月15日 -> 2020-03-15
            match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", value)
            if match:
                year, month, day = match.groups()
                return f"{year}-{int(month):02d}-{int(day):02d}"

            # パターン: 2020 だけ
            match = re.search(r"(\d{4})", value)
            if match:
                return match.group(1)

            return value
        return None

    def _extract_coat_color(self, soup: BeautifulSoup) -> Optional[str]:
        """毛色を抽出する。"""
        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue

            label = th.get_text(strip=True)
            if "毛色" not in label:
                continue

            return td.get_text(strip=True) or None
        return None

    def _extract_breeder(self, soup: BeautifulSoup) -> Optional[str]:
        """生産者名を抽出する。"""
        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue

            label = th.get_text(strip=True)
            if "生産者" not in label:
                continue

            a = td.find("a")
            if a is not None:
                return a.get_text(strip=True) or None
            return td.get_text(strip=True) or None
        return None

    def _extract_breeder_region(self, soup: BeautifulSoup) -> Optional[str]:
        """生産地域を抽出する。"""
        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue

            label = th.get_text(strip=True)
            if "産地" not in label and "生産地" not in label:
                continue

            return td.get_text(strip=True) or None
        return None

    def _extract_owner(self, soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
        """馬主名と馬主IDを抽出する。"""
        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue

            label = th.get_text(strip=True)
            if "馬主" not in label:
                continue

            a = td.find("a")
            if a is not None:
                name = a.get_text(strip=True) or None
                href = a.get("href", "")
                match = self.OWNER_ID_PATTERN.search(href)
                owner_id = match.group(1) if match else None
                return name, owner_id

            return td.get_text(strip=True) or None, None
        return None, None

    # ------------------------------------------------------------------
    # 血統情報抽出
    # ------------------------------------------------------------------
    def _extract_pedigree(self, soup: BeautifulSoup) -> dict:
        """
        血統テーブルから血統情報を抽出する。

        netkeiba の血統テーブルは以下のような構造:
        - 左側に父系、右側に母系
        - 各行にリンク付きの馬名
        """
        pedigree = {
            "sire_id": None,
            "sire_name": None,
            "dam_id": None,
            "dam_name": None,
            "broodmare_sire_id": None,
            "broodmare_sire_name": None,
            "sire_sire_name": None,
            "sire_dam_name": None,
            "dam_dam_name": None,
        }

        # 血統テーブルを探す
        blood_table = soup.select_one("table.blood_table")
        if not blood_table:
            # 別のセレクタを試す
            blood_table = soup.select_one("table[summary*='血統']")
        if not blood_table:
            # 「血統」というテキストを含むテーブルを探す
            for table in soup.find_all("table"):
                if "血統" in table.get_text():
                    blood_table = table
                    break

        if not blood_table:
            logger.debug("血統テーブルが見つかりませんでした")
            return pedigree

        # 血統テーブル内のリンクを全て収集
        links = blood_table.find_all("a", href=self.HORSE_ID_PATTERN)

        # 典型的なレイアウト:
        # Row 0: 父父, 父
        # Row 1: 父母
        # Row 2: 母父, 母
        # Row 3: 母母

        # リンクを位置で分類
        horse_links = []
        for link in links:
            href = link.get("href", "")
            match = self.HORSE_ID_PATTERN.search(href)
            if match:
                horse_id = match.group(1)
                name = link.get_text(strip=True)
                horse_links.append((horse_id, name))

        # 順序に基づいて割り当て
        # netkeiba の血統テーブルでは通常:
        # [0] = 父父父, [1] = 父父, [2] = 父, [3] = 父父母, [4] = 父母父, [5] = 父母, ...
        # ただしテーブル構造によって異なる場合がある

        # より信頼性の高い方法: td のクラスや位置で特定
        self._extract_pedigree_by_structure(blood_table, pedigree)

        return pedigree

    def _extract_pedigree_by_structure(self, blood_table: Tag, pedigree: dict) -> None:
        """テーブル構造から血統情報を抽出する。"""
        # 行ごとに処理
        rows = blood_table.find_all("tr")

        for row_idx, row in enumerate(rows):
            tds = row.find_all("td")
            for td_idx, td in enumerate(tds):
                a = td.find("a", href=self.HORSE_ID_PATTERN)
                if not a:
                    continue

                href = a.get("href", "")
                match = self.HORSE_ID_PATTERN.search(href)
                if not match:
                    continue

                horse_id = match.group(1)
                name = a.get_text(strip=True)

                # rowspan/colspan を考慮した位置判定
                rowspan = int(td.get("rowspan", 1))

                # 大きい rowspan = 近い世代
                if rowspan >= 4:
                    # 父または母
                    if row_idx < len(rows) / 2:
                        pedigree["sire_id"] = horse_id
                        pedigree["sire_name"] = name
                    else:
                        pedigree["dam_id"] = horse_id
                        pedigree["dam_name"] = name
                elif rowspan >= 2:
                    # 祖父母世代
                    # 位置で判定
                    if row_idx < len(rows) / 2:
                        if not pedigree.get("sire_sire_name"):
                            pedigree["sire_sire_name"] = name
                        else:
                            pedigree["sire_dam_name"] = name
                    else:
                        if not pedigree.get("broodmare_sire_name"):
                            pedigree["broodmare_sire_id"] = horse_id
                            pedigree["broodmare_sire_name"] = name
                        else:
                            pedigree["dam_dam_name"] = name

        # フォールバック: 直接父・母・母父を探す
        if not pedigree.get("sire_name"):
            self._extract_parent_fallback(blood_table, pedigree)

    def _extract_parent_fallback(self, soup: BeautifulSoup, pedigree: dict) -> None:
        """フォールバック: ラベルベースで父・母・母父を探す。"""
        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue

            label = th.get_text(strip=True)

            if label == "父" or label == "Sire":
                a = td.find("a", href=self.HORSE_ID_PATTERN)
                if a:
                    match = self.HORSE_ID_PATTERN.search(a.get("href", ""))
                    if match:
                        pedigree["sire_id"] = match.group(1)
                        pedigree["sire_name"] = a.get_text(strip=True)

            elif label == "母" or label == "Dam":
                a = td.find("a", href=self.HORSE_ID_PATTERN)
                if a:
                    match = self.HORSE_ID_PATTERN.search(a.get("href", ""))
                    if match:
                        pedigree["dam_id"] = match.group(1)
                        pedigree["dam_name"] = a.get_text(strip=True)

            elif label == "母父" or "BMS" in label or "Broodmare" in label:
                a = td.find("a", href=self.HORSE_ID_PATTERN)
                if a:
                    match = self.HORSE_ID_PATTERN.search(a.get("href", ""))
                    if match:
                        pedigree["broodmare_sire_id"] = match.group(1)
                        pedigree["broodmare_sire_name"] = a.get_text(strip=True)

    # ------------------------------------------------------------------
    # 成績情報抽出
    # ------------------------------------------------------------------
    def _extract_total_prize(self, soup: BeautifulSoup) -> Optional[int]:
        """総獲得賞金を抽出する。"""
        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue

            label = th.get_text(strip=True)
            if "獲得賞金" not in label and "総賞金" not in label:
                continue

            value = td.get_text(strip=True)
            # 1,234万円 -> 12340000
            # 1億2345万円 -> 123450000
            try:
                # 億を処理
                oku_match = re.search(r"(\d+)億", value)
                man_match = re.search(r"(\d+(?:,\d+)?)万", value)

                total = 0
                if oku_match:
                    total += int(oku_match.group(1)) * 100000000
                if man_match:
                    man_str = man_match.group(1).replace(",", "")
                    total += int(man_str) * 10000

                return total if total > 0 else None
            except ValueError:
                return None
        return None

    def _extract_career_stats(self, soup: BeautifulSoup) -> tuple[Optional[int], Optional[int]]:
        """通算成績 (出走数, 勝利数) を抽出する。"""
        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue

            label = th.get_text(strip=True)
            if "通算成績" not in label and "成績" not in label:
                continue

            value = td.get_text(strip=True)
            # パターン: 10戦3勝, 10-3-2-5, [10-3-2-5]

            # 10戦3勝 パターン
            match = re.search(r"(\d+)戦(\d+)勝", value)
            if match:
                return int(match.group(1)), int(match.group(2))

            # 10-3-2-5 パターン (出走-1着-2着-3着)
            match = re.search(r"(\d+)-(\d+)-\d+-\d+", value)
            if match:
                return int(match.group(1)), int(match.group(2))

            # [10,3,2,5] パターン
            match = re.search(r"\[?(\d+)[,\s]+(\d+)", value)
            if match:
                return int(match.group(1)), int(match.group(2))

        return None, None


# ------------------------------------------------------------------
# テスト・デバッグ用エントリポイント
# ------------------------------------------------------------------
if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    print("=" * 80)
    print("HorseExtendedParser テスト")
    print("=" * 80)

    # テスト用のHTMLを作成
    test_html = """
    <html>
    <body>
        <div class="db_head">
            <h1>ドウデュース</h1>
        </div>
        <table>
            <tr><th>性齢</th><td>牡4</td></tr>
            <tr><th>生年月日</th><td>2019年5月7日</td></tr>
            <tr><th>毛色</th><td>鹿毛</td></tr>
            <tr><th>生産者</th><td><a href="/breeder/12345/">ノーザンファーム</a></td></tr>
            <tr><th>馬主</th><td><a href="/owner/abc123/">キーファーズ</a></td></tr>
        </table>
        <table class="blood_table">
            <tr>
                <td rowspan="4"><a href="/horse/2010101234/">ハーツクライ</a></td>
            </tr>
            <tr>
                <td rowspan="4"><a href="/horse/2012102345/">ダストアンドダイヤモンズ</a></td>
            </tr>
        </table>
    </body>
    </html>
    """

    soup = BeautifulSoup(test_html, "html.parser")
    parser = HorseExtendedParser()

    try:
        record = parser.parse("2019104385", soup)
        print(f"\nパース成功:")
        print(f"  horse_id: {record.horse_id}")
        print(f"  horse_name: {record.horse_name}")
        print(f"  sex: {record.sex}")
        print(f"  birth_date: {record.birth_date}")
        print(f"  coat_color: {record.coat_color}")
        print(f"  breeder: {record.breeder}")
        print(f"  owner: {record.owner}")
        print(f"  sire_name: {record.sire_name}")
        print(f"  dam_name: {record.dam_name}")
    except Exception as e:
        print(f"\nパース失敗: {e}")

    print("\n" + "=" * 80)
