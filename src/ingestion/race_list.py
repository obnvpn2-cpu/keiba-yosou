"""
netkeiba ingestion パイプライン - レース一覧取得

レース検索ページをパースし、race_id のリストを取得する。
ページネーションに対応。
"""

import re
import logging
from typing import Optional

from bs4 import BeautifulSoup

from .scraper import NetkeibaClient, get_client
from .models import JRA_PLACE_CODES

logger = logging.getLogger(__name__)


def fetch_race_ids(
    start_year: int,
    end_year: int,
    start_mon: int = 1,
    end_mon: int = 12,
    place_codes: Optional[list[str]] = None,
    client: Optional[NetkeibaClient] = None,
    max_pages: int = 100,
) -> list[str]:
    """
    指定期間のレースIDを全て取得する。
    
    Args:
        start_year: 開始年
        end_year: 終了年
        start_mon: 開始月（デフォルト: 1）
        end_mon: 終了月（デフォルト: 12）
        place_codes: 場コードのリスト（デフォルト: JRA全場）
        client: NetkeibaClient（Noneの場合はデフォルトを使用）
        max_pages: 最大ページ数（無限ループ防止）
    
    Returns:
        race_id のリスト（12桁文字列）
    """
    if place_codes is None:
        place_codes = JRA_PLACE_CODES
    
    if client is None:
        client = get_client()
    
    all_race_ids: list[str] = []
    page = 1
    
    logger.info(
        f"Fetching race list: {start_year}/{start_mon} - {end_year}/{end_mon}, "
        f"places={place_codes}"
    )
    
    while page <= max_pages:
        logger.info(f"Fetching page {page}...")
        
        try:
            html = client.get_race_list_page(
                start_year=start_year,
                start_mon=start_mon,
                end_year=end_year,
                end_mon=end_mon,
                place_codes=place_codes,
                page=page,
            )
        except Exception as e:
            logger.error(f"Failed to fetch page {page}: {e}")
            break
        
        race_ids = _parse_race_list_page(html)
        
        if not race_ids:
            logger.info(f"No more races found on page {page}. Done.")
            break
        
        all_race_ids.extend(race_ids)
        logger.info(f"Found {len(race_ids)} races on page {page}. Total: {len(all_race_ids)}")
        
        # 次ページが存在するかチェック
        if not _has_next_page(html, page):
            logger.info("No next page. Done.")
            break
        
        page += 1
    
    # 重複を除去して返す
    unique_ids = list(dict.fromkeys(all_race_ids))
    logger.info(f"Total unique race IDs: {len(unique_ids)}")
    
    return unique_ids


def _parse_race_list_page(html: str) -> list[str]:
    """
    レース検索結果ページからrace_idを抽出する。
    
    Args:
        html: HTMLコンテンツ
    
    Returns:
        race_id のリスト
    """
    soup = BeautifulSoup(html, "html.parser")
    race_ids = []
    
    # レースへのリンクを探す
    # <a href="/race/202406050901/">
    # または
    # <a href="https://db.netkeiba.com/race/202406050901/">
    links = soup.select("a[href*='/race/']")
    
    for link in links:
        href = link.get("href", "")
        
        # race_id を抽出（12桁の数値）
        match = re.search(r"/race/(\d{12})/?", href)
        if match:
            race_id = match.group(1)
            if race_id not in race_ids:
                race_ids.append(race_id)
    
    return race_ids


def _has_next_page(html: str, current_page: int) -> bool:
    """
    次のページが存在するかチェックする。
    
    Args:
        html: HTMLコンテンツ
        current_page: 現在のページ番号
    
    Returns:
        次ページが存在する場合はTrue
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # ページネーションリンクを探す
    # 次ページへのリンク: page=N+1
    next_page = current_page + 1
    next_page_pattern = f"page={next_page}"
    
    # ページネーション領域のリンクをチェック
    pager_links = soup.select("a[href*='page=']")
    
    for link in pager_links:
        href = link.get("href", "")
        if next_page_pattern in href:
            return True
    
    # または「次」「>」などのリンクテキストを持つリンク
    for link in pager_links:
        text = link.get_text(strip=True)
        if text in ("次", ">", ">>", "次へ", "Next"):
            return True
    
    return False


def fetch_race_ids_by_year(
    year: int,
    place_codes: Optional[list[str]] = None,
    client: Optional[NetkeibaClient] = None,
) -> list[str]:
    """
    指定年のレースIDを全て取得する（ヘルパー関数）。
    
    Args:
        year: 年
        place_codes: 場コードのリスト（デフォルト: JRA全場）
        client: NetkeibaClient
    
    Returns:
        race_id のリスト
    """
    return fetch_race_ids(
        start_year=year,
        end_year=year,
        start_mon=1,
        end_mon=12,
        place_codes=place_codes,
        client=client,
    )


def fetch_race_ids_by_month(
    year: int,
    month: int,
    place_codes: Optional[list[str]] = None,
    client: Optional[NetkeibaClient] = None,
) -> list[str]:
    """
    指定年月のレースIDを取得する（ヘルパー関数）。
    
    Args:
        year: 年
        month: 月
        place_codes: 場コードのリスト（デフォルト: JRA全場）
        client: NetkeibaClient
    
    Returns:
        race_id のリスト
    """
    return fetch_race_ids(
        start_year=year,
        end_year=year,
        start_mon=month,
        end_mon=month,
        place_codes=place_codes,
        client=client,
    )


def validate_race_id(race_id: str) -> bool:
    """
    race_id の形式をバリデーションする。
    
    Args:
        race_id: 検証するレースID
    
    Returns:
        有効な場合はTrue
    """
    if not re.match(r"^\d{12}$", race_id):
        return False
    
    # 場コードをチェック
    place_code = race_id[4:6]
    if place_code not in JRA_PLACE_CODES:
        # 地方競馬のコードかもしれないので警告だけ
        logger.warning(f"Unknown place code in race_id: {race_id}")
    
    return True


def parse_race_id(race_id: str) -> dict:
    """
    race_id を分解して辞書で返す。
    
    Args:
        race_id: 12桁のレースID
    
    Returns:
        {year, place_code, kai, nichime, race_no}
    """
    if len(race_id) != 12:
        raise ValueError(f"Invalid race_id: {race_id}")
    
    return {
        "year": int(race_id[0:4]),
        "place_code": race_id[4:6],
        "kai": int(race_id[6:8]),
        "nichime": int(race_id[8:10]),
        "race_no": int(race_id[10:12]),
    }
