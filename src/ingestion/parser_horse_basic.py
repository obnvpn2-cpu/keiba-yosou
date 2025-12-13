# -*- coding: utf-8 -*-
# src/ingestion/parser_horse_basic.py
"""
馬の基本情報ページ（https://db.netkeiba.com/horse/{horse_id}/）用パーサー。

現時点で取得するカラム:
- horse_id
- horse_name
- sex         : "牡" / "牝" / "セン" のいずれか（取れなければ None）
- breeder     : 生産者名（取れなければ None）

改善点:
- 正規表現を使った性別抽出の堅牢化
- 馬名抽出のフォールバック強化
- エラーハンドリングの改善
- ログレベルの最適化

NOTE:
- HTML 構造変更に強くするため、特定クラス名だけに依存せず、
  ・見出し (<h1>)
  ・「性齢」「生産者」ラベルの <th> など
  を優先して探索する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class HorseBasicRecord:
    """馬の基本情報（最低限バージョン）"""
    horse_id: str
    horse_name: Optional[str]
    sex: Optional[str]
    breeder: Optional[str]


class HorseBasicParser:
    """
    netkeiba 馬ページ用パーサー
    
    Features:
        - 複数のセレクタパターンでフォールバック
        - 正規表現による性別抽出の堅牢化
        - 文字列の正規化（strip, 空文字チェック）
        - 詳細なログ出力
    
    Example:
        >>> parser = HorseBasicParser()
        >>> soup = fetcher.fetch_soup("https://db.netkeiba.com/horse/2020104385/")
        >>> record = parser.parse("2020104385", soup)
    """

    # 性別抽出用の正規表現パターン
    SEX_PATTERNS = {
        "牡": re.compile(r"^牡"),
        "牝": re.compile(r"^牝"),
        "セン": re.compile(r"^セン"),
    }

    def parse(self, horse_id: str, soup: BeautifulSoup) -> HorseBasicRecord:
        """
        馬ページ HTML をパースして HorseBasicRecord を返す。

        Args:
            horse_id: URL から分かっている馬 ID（10桁想定）
            soup    : BeautifulSoup オブジェクト

        Returns:
            HorseBasicRecord
        
        Raises:
            ValueError: 馬名が取得できない場合（致命的エラー）
        """
        logger.debug("Parsing horse page: horse_id=%s", horse_id)
        
        horse_name = self._extract_horse_name(soup)
        sex = self._extract_sex(soup)
        breeder = self._extract_breeder(soup)

        # 馬名は必須（取れない場合はエラー）
        if not horse_name:
            logger.error("horse_id=%s: 馬名を取得できませんでした（致命的エラー）", horse_id)
            raise ValueError(f"horse_id={horse_id}: 馬名を取得できませんでした")
        
        # 性別・生産者は任意（デバッグログのみ）
        if not sex:
            logger.debug("horse_id=%s: 性別を取得できませんでした", horse_id)
        if not breeder:
            logger.debug("horse_id=%s: 生産者を取得できませんでした", horse_id)

        record = HorseBasicRecord(
            horse_id=horse_id,
            horse_name=horse_name,
            sex=sex,
            breeder=breeder,
        )
        
        logger.info(
            "Parsed horse basic: horse_id=%s, horse_name=%s, sex=%s, breeder=%s",
            horse_id,
            horse_name,
            sex,
            breeder,
        )
        return record

    # ------------------------------------------------------------------
    # 内部ヘルパ
    # ------------------------------------------------------------------
    def _extract_horse_name(self, soup: BeautifulSoup) -> Optional[str]:
        """
        馬名を <h1> 付近から取得する。
        
        複数のパターンを試行して、最初に見つかったものを返す。
        
        Args:
            soup: BeautifulSoup オブジェクト
        
        Returns:
            馬名（str）、または None
        """
        # 優先度順にセレクタを試す
        candidates = [
            # パターン1: div.db_head > h1（最も一般的）
            soup.select_one("div.db_head h1"),
            # パターン2: div.db_main_box > h1
            soup.select_one("div.db_main_box h1"),
            # パターン3: title タグから抽出（最終手段）
            soup.find("title"),
            # パターン4: 任意の h1
            soup.find("h1"),
        ]
        
        for i, el in enumerate(candidates):
            if el:
                text = el.get_text(strip=True)
                if text:
                    # title タグの場合は「| netkeiba.com」などを除去
                    if el.name == "title":
                        text = re.sub(r"\s*[|｜]\s*netkeiba\.com.*", "", text)
                    
                    if text:
                        logger.debug("馬名を取得: パターン%d, text=%s", i + 1, text)
                        return text
        
        logger.warning("馬名を取得できませんでした（全てのパターンで失敗）")
        return None

    def _extract_sex(self, soup: BeautifulSoup) -> Optional[str]:
        """
        性齢(牡3 など) から性別だけを抜き出す。
        
        正規表現を使って堅牢に抽出。
        
        Args:
            soup: BeautifulSoup オブジェクト
        
        Returns:
            性別（"牡" / "牝" / "セン"）、または None
        """
        # 「性齢」ラベルの th を探す
        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue
            
            label = th.get_text(strip=True)
            # 「性齢」「性別」「性」のいずれかにマッチ
            if not any(keyword in label for keyword in ["性齢", "性別", "性"]):
                continue
            
            value = td.get_text(strip=True)
            if not value:
                logger.debug("性齢の値が空でした")
                return None
            
            # 正規表現で性別を抽出
            for sex_name, pattern in self.SEX_PATTERNS.items():
                if pattern.search(value):
                    logger.debug("性別を取得: %s (元の値: %s)", sex_name, value)
                    return sex_name
            
            # パターンにマッチしない場合
            logger.warning("未知の性齢表記: %s", value)
            return None
        
        logger.debug("性齢情報が見つかりませんでした")
        return None

    def _extract_breeder(self, soup: BeautifulSoup) -> Optional[str]:
        """
        生産者名をテーブルから取得する。
        
        Args:
            soup: BeautifulSoup オブジェクト
        
        Returns:
            生産者名（str）、または None
        """
        for tr in soup.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue
            
            label = th.get_text(strip=True)
            if "生産者" not in label:
                continue

            # 生産者はリンクになっていることが多いので a からテキストを拾う
            a = td.find("a")
            if a is not None:
                text = a.get_text(strip=True)
            else:
                text = td.get_text(strip=True)
            
            if text:
                logger.debug("生産者を取得: %s", text)
                return text
            else:
                logger.debug("生産者の値が空でした")
                return None

        logger.debug("生産者情報が見つかりませんでした")
        return None


# ------------------------------------------------------------------
# テスト・デバッグ用エントリポイント
# ------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from pathlib import Path
    
    # ログ設定
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    print("=" * 80)
    print("HorseBasicParser テスト")
    print("=" * 80)
    
    # テスト用のHTMLを作成
    test_html = """
    <html>
    <body>
        <div class="db_head">
            <h1>テストホース</h1>
        </div>
        <table>
            <tr>
                <th>性齢</th>
                <td>牡3</td>
            </tr>
            <tr>
                <th>生産者</th>
                <td><a href="/breeder/12345/">テスト牧場</a></td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    soup = BeautifulSoup(test_html, "html.parser")
    parser = HorseBasicParser()
    
    try:
        record = parser.parse("0000000001", soup)
        print(f"\n✅ パース成功:")
        print(f"  horse_id: {record.horse_id}")
        print(f"  horse_name: {record.horse_name}")
        print(f"  sex: {record.sex}")
        print(f"  breeder: {record.breeder}")
    except Exception as e:
        print(f"\n❌ パース失敗: {e}")
    
    print("\n" + "=" * 80)
