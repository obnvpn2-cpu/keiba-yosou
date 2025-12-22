# -*- coding: utf-8 -*-
# src/ingestion/parser_pedigree_5gen.py
"""
5代血統表ページ（https://db.netkeiba.com/horse/ped/{horse_id}/）用パーサー。

5代血統表から各祖先の情報を抽出し、正規化された形式で返す。

【血統表の構造】
- 世代1 (gen=1): 父(s), 母(d) の2頭
- 世代2 (gen=2): 父父(ss), 父母(sd), 母父(ds), 母母(dd) の4頭
- 世代3 (gen=3): sss, ssd, sds, sdd, dss, dsd, dds, ddd の8頭
- 世代4 (gen=4): 16頭
- 世代5 (gen=5): 32頭

【position表記】
- "s" = Sire（父）
- "d" = Dam（母）
- "ss" = Sire's Sire（父父）
- "sd" = Sire's Dam（父母）
- "ds" = Dam's Sire（母父）= Broodmare Sire (BMS)
- "dd" = Dam's Dam（母母）
- 以下、同様に s/d を連結

【出力】
各祖先について:
- generation: 世代番号 (1-5)
- position: 位置パス (例: "ss", "dsd")
- ancestor_id: 馬ID（リンクがあれば）
- ancestor_name: 馬名（必須）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import re

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


@dataclass
class PedigreeAncestor:
    """血統表の1セル（祖先1頭）の情報"""
    horse_id: str          # 対象馬のID
    generation: int        # 世代 (1-5)
    position: str          # 位置パス ("s", "d", "ss", "sd", ...)
    ancestor_id: Optional[str]   # 祖先の馬ID（リンクがあれば）
    ancestor_name: str     # 祖先の馬名


class Pedigree5GenParser:
    """
    netkeiba 5代血統表ページ用パーサー

    血統表ページ (https://db.netkeiba.com/horse/ped/{horse_id}/) から
    5世代分の祖先情報を抽出する。

    Example:
        >>> parser = Pedigree5GenParser()
        >>> soup = fetcher.fetch_soup("https://db.netkeiba.com/horse/ped/2019104385/")
        >>> ancestors = parser.parse("2019104385", soup)
        >>> for a in ancestors:
        ...     print(f"Gen{a.generation} {a.position}: {a.ancestor_name}")
    """

    # 馬IDを抽出する正規表現
    HORSE_ID_PATTERN = re.compile(r"/horse/(\d+)")

    # 世代ごとの期待セル数（rowspanの値）
    # 5代血統表は32行あり、各世代のrowspanは:
    # Gen1: 16 (2セル x 16行 = 父/母)
    # Gen2: 8  (4セル x 8行)
    # Gen3: 4  (8セル x 4行)
    # Gen4: 2  (16セル x 2行)
    # Gen5: 1  (32セル x 1行)
    ROWSPAN_TO_GEN = {
        16: 1,
        8: 2,
        4: 3,
        2: 4,
        1: 5,
    }

    def parse(self, horse_id: str, soup: BeautifulSoup) -> List[PedigreeAncestor]:
        """
        5代血統表ページをパースして祖先リストを返す。

        Args:
            horse_id: 対象馬のID
            soup: BeautifulSoup オブジェクト

        Returns:
            PedigreeAncestor のリスト（最大62頭: 2+4+8+16+32）

        Note:
            欠損データがあっても落ちずに、取得できた分だけ返す
        """
        logger.debug("Parsing 5-gen pedigree: horse_id=%s", horse_id)

        ancestors: List[PedigreeAncestor] = []

        # 血統表テーブルを探す
        pedigree_table = self._find_pedigree_table(soup)
        if not pedigree_table:
            logger.warning("horse_id=%s: 血統表テーブルが見つかりませんでした", horse_id)
            return ancestors

        # 方法1: rowspanベースで世代を判定
        ancestors = self._parse_by_rowspan(horse_id, pedigree_table)

        if not ancestors:
            # 方法2: テーブル構造から推測（フォールバック）
            ancestors = self._parse_by_structure(horse_id, pedigree_table)

        logger.info(
            "Parsed pedigree: horse_id=%s, ancestors=%d",
            horse_id, len(ancestors)
        )
        return ancestors

    def _find_pedigree_table(self, soup: BeautifulSoup) -> Optional[Tag]:
        """血統表テーブルを探す。"""
        # パターン1: class="blood_table"
        table = soup.select_one("table.blood_table")
        if table:
            return table

        # パターン2: summary属性に「血統」を含む
        for table in soup.find_all("table"):
            summary = table.get("summary", "")
            if "血統" in summary or "pedigree" in summary.lower():
                return table

        # パターン3: id="pedigree_table" または class に ped を含む
        table = soup.select_one("table#pedigree_table, table[class*='ped']")
        if table:
            return table

        # パターン4: 最も大きなテーブル（行数が多いもの）
        tables = soup.find_all("table")
        if tables:
            max_rows = 0
            best_table = None
            for t in tables:
                rows = len(t.find_all("tr"))
                if rows > max_rows:
                    max_rows = rows
                    best_table = t
            if max_rows >= 16:  # 5代血統表は少なくとも16行以上
                return best_table

        return None

    def _parse_by_rowspan(
        self, horse_id: str, table: Tag
    ) -> List[PedigreeAncestor]:
        """rowspan属性を使って世代を判定しながらパース。"""
        ancestors: List[PedigreeAncestor] = []

        # 各世代のカウンター（位置計算用）
        gen_counters = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        # 全ての td を取得
        tds = table.find_all("td")

        for td in tds:
            rowspan = int(td.get("rowspan", 1))

            # rowspan から世代を判定
            generation = self.ROWSPAN_TO_GEN.get(rowspan)
            if generation is None:
                continue

            # 祖先情報を抽出
            ancestor_name, ancestor_id = self._extract_ancestor_info(td)

            if not ancestor_name:
                # 名前がなければスキップ
                gen_counters[generation] += 1
                continue

            # 位置を計算
            position = self._calc_position(generation, gen_counters[generation])

            ancestors.append(PedigreeAncestor(
                horse_id=horse_id,
                generation=generation,
                position=position,
                ancestor_id=ancestor_id,
                ancestor_name=ancestor_name,
            ))

            gen_counters[generation] += 1

        return ancestors

    def _parse_by_structure(
        self, horse_id: str, table: Tag
    ) -> List[PedigreeAncestor]:
        """テーブル構造から推測してパース（フォールバック）。"""
        ancestors: List[PedigreeAncestor] = []

        # 全てのリンクを収集
        links = table.find_all("a", href=self.HORSE_ID_PATTERN)

        # リンクの位置から世代と位置を推測
        for idx, link in enumerate(links):
            ancestor_id = None
            href = link.get("href", "")
            match = self.HORSE_ID_PATTERN.search(href)
            if match:
                ancestor_id = match.group(1)

            ancestor_name = link.get_text(strip=True)
            if not ancestor_name:
                continue

            # インデックスから世代と位置を推測
            # 順序: Gen1(2), Gen2(4), Gen3(8), Gen4(16), Gen5(32)
            # 累計: 2, 6, 14, 30, 62
            if idx < 2:
                generation = 1
                pos_idx = idx
            elif idx < 6:
                generation = 2
                pos_idx = idx - 2
            elif idx < 14:
                generation = 3
                pos_idx = idx - 6
            elif idx < 30:
                generation = 4
                pos_idx = idx - 14
            elif idx < 62:
                generation = 5
                pos_idx = idx - 30
            else:
                continue

            position = self._calc_position(generation, pos_idx)

            ancestors.append(PedigreeAncestor(
                horse_id=horse_id,
                generation=generation,
                position=position,
                ancestor_id=ancestor_id,
                ancestor_name=ancestor_name,
            ))

        return ancestors

    def _extract_ancestor_info(self, td: Tag) -> Tuple[Optional[str], Optional[str]]:
        """
        tdから祖先の名前とIDを抽出する。

        Returns:
            (ancestor_name, ancestor_id) のタプル
        """
        # リンクがあればそこから取得
        link = td.find("a", href=self.HORSE_ID_PATTERN)
        if link:
            href = link.get("href", "")
            match = self.HORSE_ID_PATTERN.search(href)
            ancestor_id = match.group(1) if match else None
            ancestor_name = link.get_text(strip=True)
            return ancestor_name, ancestor_id

        # リンクがなければテキストだけ
        text = td.get_text(strip=True)
        if text:
            # 余計な文字を除去
            text = re.sub(r"\s+", " ", text)
            return text, None

        return None, None

    def _calc_position(self, generation: int, index: int) -> str:
        """
        世代とインデックスから位置パスを計算する。

        Args:
            generation: 世代 (1-5)
            index: その世代内でのインデックス (0-based)

        Returns:
            位置パス (例: "s", "d", "ss", "sd", ...)

        【計算方法】
        各世代には 2^gen 個の祖先がいる。
        インデックスを2進数で表現し、0='s', 1='d' として変換。

        例: Gen3, index=5 の場合
        - 2進数: 101
        - 変換: s=1, d=0, s=1 → "sds"
        """
        if generation == 0 or index < 0:
            return ""

        # 世代ごとの祖先数
        count = 2 ** generation

        if index >= count:
            logger.warning(
                "Invalid index %d for generation %d (max %d)",
                index, generation, count - 1
            )
            index = index % count

        # インデックスを2進数に変換し、s/d に変換
        path = ""
        for _ in range(generation):
            if index & 1:
                path = "d" + path
            else:
                path = "s" + path
            index >>= 1

        return path


def parse_pedigree_page(horse_id: str, html: str) -> List[PedigreeAncestor]:
    """
    5代血統表ページのHTMLをパースする便利関数。

    Args:
        horse_id: 馬ID
        html: HTMLコンテンツ

    Returns:
        PedigreeAncestor のリスト
    """
    soup = BeautifulSoup(html, "html.parser")
    parser = Pedigree5GenParser()
    return parser.parse(horse_id, soup)


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
    print("Pedigree5GenParser テスト")
    print("=" * 80)

    # テスト用の簡略化されたHTML（2世代分のみ）
    test_html = """
    <html>
    <body>
        <table class="blood_table">
            <tr>
                <td rowspan="16"><a href="/horse/2010101234/">ディープインパクト</a></td>
                <td rowspan="8"><a href="/horse/2000101111/">サンデーサイレンス</a></td>
            </tr>
            <tr>
                <td rowspan="8"><a href="/horse/2000102222/">ウインドインハーヘア</a></td>
            </tr>
            <tr>
                <td rowspan="16"><a href="/horse/2012102345/">ダストアンドダイヤモンズ</a></td>
                <td rowspan="8"><a href="/horse/2001103333/">Vindication</a></td>
            </tr>
            <tr>
                <td rowspan="8"><a href="/horse/2001104444/">Majesty's Crown</a></td>
            </tr>
        </table>
    </body>
    </html>
    """

    soup = BeautifulSoup(test_html, "html.parser")
    parser = Pedigree5GenParser()

    ancestors = parser.parse("2019104385", soup)

    print(f"\nパース結果: {len(ancestors)} 頭の祖先")
    for a in ancestors:
        print(f"  Gen{a.generation} {a.position:5s}: {a.ancestor_name} ({a.ancestor_id})")

    print("\n" + "=" * 80)
    print("位置パス計算テスト")
    print("=" * 80)

    for gen in range(1, 6):
        count = 2 ** gen
        positions = [parser._calc_position(gen, i) for i in range(count)]
        print(f"Gen{gen}: {positions}")

    print("\n" + "=" * 80)
