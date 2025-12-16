"""
netkeiba ingestion パイプライン - HTTPクライアント

Cookie認証付きのrequestsセッションを提供し、
指数バックオフ付きリトライ、ランダムスリープを実装する。

参考: Zenn「競馬予想で始める機械学習〜完全版〜」Chapter 02「スクレイピング」
- User-Agentをリストから毎回ランダムに選択
- 400 Bad Request時はUser-Agentを変更してリトライ
"""

import os
import re
import time
import random
import logging
import json
from typing import Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 設定
BASE_URL = "https://db.netkeiba.com"

# User-Agentリスト（Zenn記事参考）
# 各リクエストごとにランダムに選択することで、ボット検知を回避
USER_AGENTS = [
    # Chrome (Windows/Mac)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Firefox (Windows/Mac)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
    # Safari (Mac)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    # Edge (Windows)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
]

# Cookie名のプレフィックス
COOKIE_PREFIX = "NETKEIBA_COOKIE_"

# リトライ設定
MAX_RETRIES = 5
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
BACKOFF_FACTOR = 1.0  # 1, 2, 4, 8, 16秒 と増える

# 400エラー専用リトライ設定
MAX_400_RETRIES = 3
RETRY_400_SLEEP_MIN = 1.0
RETRY_400_SLEEP_MAX = 2.0

# リクエスト間のスリープ（秒）
# Zenn記事の推奨: 2〜3秒程度
MIN_SLEEP = 2.0
MAX_SLEEP = 3.5


def get_random_user_agent() -> str:
    """ランダムなUser-Agentを取得する。"""
    return random.choice(USER_AGENTS)


def load_cookies_from_env() -> dict[str, str]:
    load_dotenv()
    cookies = {}
    for key, value in os.environ.items():
        if not key.startswith(COOKIE_PREFIX):
            continue

        raw_name = key[len(COOKIE_PREFIX):]

        # Windows だと環境変数名が全部大文字になるので、
        # cookie 名は強制的に小文字にそろえる
        cookie_name = raw_name.lower()

        cookies[cookie_name] = value

    return cookies



def create_session(cookies: Optional[dict[str, str]] = None) -> requests.Session:
    """
    リトライ設定付きのrequests.Sessionを作成する。
    
    Args:
        cookies: セットするCookie辞書（Noneの場合は環境変数から読み込む）
    
    Returns:
        設定済みのrequests.Session
    """
    session = requests.Session()
    
    # リトライ設定（500系エラー用）
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_CODES,
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # デフォルトヘッダー設定（User-Agentは各リクエストで上書き）
    session.headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cache-Control": "max-age=0",
    })
    
    # Cookie設定
    if cookies is None:
        cookies = load_cookies_from_env()
    
    for name, value in cookies.items():
        session.cookies.set(name, value, domain=".netkeiba.com")
    
    return session


class NetkeibaClient:
    """
    netkeiba.com 用のHTTPクライアント。
    
    Cookie認証、リトライ、レート制限を管理する。
    User-Agentを各リクエストでランダムに切り替える。
    """
    
    def __init__(
        self,
        session: Optional[requests.Session] = None,
        min_sleep: float = MIN_SLEEP,
        max_sleep: float = MAX_SLEEP,
    ):
        """
        Args:
            session: 使用するrequests.Session（Noneの場合は新規作成）
            min_sleep: リクエスト間の最小スリープ秒数
            max_sleep: リクエスト間の最大スリープ秒数
        """
        self.session = session or create_session()
        self.min_sleep = min_sleep
        self.max_sleep = max_sleep
        self._last_request_time: Optional[float] = None
        self._current_user_agent: str = get_random_user_agent()
    
    def _get_headers(self, rotate_ua: bool = True) -> dict[str, str]:
        """
        リクエスト用ヘッダーを取得する。
        
        Args:
            rotate_ua: TrueならUser-Agentをローテーション
        
        Returns:
            ヘッダー辞書
        """
        if rotate_ua:
            self._current_user_agent = get_random_user_agent()
        
        return {"User-Agent": self._current_user_agent}
    
    def _wait_if_needed(self) -> None:
        """前回リクエストからの経過時間に応じてスリープする。"""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            target_sleep = random.uniform(self.min_sleep, self.max_sleep)
            if elapsed < target_sleep:
                sleep_time = target_sleep - elapsed
                logger.debug(f"Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
    
    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        timeout: int = 30,
    ) -> requests.Response:
        """
        GETリクエストを実行する。
        
        レート制限とリトライを自動的に処理する。
        400 Bad Requestの場合はUser-Agentを変更してリトライ。
        
        Args:
            url: リクエスト先URL（相対パスも可）
            params: クエリパラメータ
            timeout: タイムアウト秒数
        
        Returns:
            レスポンス
        
        Raises:
            requests.RequestException: リトライ後も失敗した場合
        """
        # 絶対URLに変換
        if not url.startswith("http"):
            url = urljoin(BASE_URL, url)
        
        # レート制限
        self._wait_if_needed()
        
        # 400エラー用リトライループ
        last_error = None
        for attempt in range(MAX_400_RETRIES + 1):
            # 毎回User-Agentを変更
            headers = self._get_headers(rotate_ua=True)
            
            if attempt > 0:
                logger.info(
                    f"Retry {attempt}/{MAX_400_RETRIES} with new User-Agent: "
                    f"{self._current_user_agent[:50]}..."
                )
            
            logger.debug(f"GET {url} (UA: {self._current_user_agent[:30]}...)")
            
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )
                self._last_request_time = time.time()
                
                # 429: Rate Limited
                if response.status_code == 429:
                    logger.warning("Rate limited (429). Waiting 30 seconds...")
                    time.sleep(30)
                    continue
                
                # 400: Bad Request - User-Agentを変えてリトライ
                if response.status_code == 400:
                    logger.warning(
                        f"Bad Request (400) for {url}. "
                        f"Attempt {attempt + 1}/{MAX_400_RETRIES + 1}"
                    )
                    if attempt < MAX_400_RETRIES:
                        # 少し待ってからリトライ
                        sleep_time = random.uniform(RETRY_400_SLEEP_MIN, RETRY_400_SLEEP_MAX)
                        logger.info(f"Waiting {sleep_time:.1f}s before retry...")
                        time.sleep(sleep_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for 400 error: {url}")
                
                # その他のエラー
                if response.status_code >= 400:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                
                return response
                
            except requests.RequestException as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < MAX_400_RETRIES:
                    sleep_time = random.uniform(RETRY_400_SLEEP_MIN, RETRY_400_SLEEP_MAX)
                    time.sleep(sleep_time)
                    continue
                raise
        
        # ここには通常到達しないが、念のため
        if last_error:
            raise last_error
        return response
    
    def get_race_page(self, race_id: str) -> str:
        """
        レース結果ページのHTMLを取得する。
        
        Args:
            race_id: 12桁のレースID
        
        Returns:
            HTMLコンテンツ
        
        Raises:
            ValueError: race_idの形式が不正な場合
            requests.RequestException: リクエスト失敗時
        """
        if not re.match(r"^\d{12}$", race_id):
            raise ValueError(f"Invalid race_id format: {race_id}")
        
        url = f"/race/{race_id}/"
        response = self.get(url)
        response.raise_for_status()
        
        # netkeiba.com は常に EUC-JP
        # apparent_encoding は誤判定（ISO-8859-7等）することがあるため使用しない
        html = self._decode_response(response)
        
        logger.debug(f"Fetched race page {race_id}, length={len(html)}")
        return html
    
    def _decode_response(self, response: requests.Response) -> str:
        """
        レスポンスを適切なエンコーディングでデコードする。
        
        netkeiba.com は EUC-JP を使用しているが、apparent_encoding が
        誤って ISO-8859-7（ギリシャ語）等を返すことがある。
        そのため、明示的に EUC-JP でデコードする。
        
        Args:
            response: HTTPレスポンス
            
        Returns:
            デコード済みHTML文字列
        """
        # 優先順位:
        # 1. Content-Type ヘッダーの charset
        # 2. HTML内の <meta charset> または <meta http-equiv="Content-Type">
        # 3. EUC-JP（netkeiba.comのデフォルト）
        
        content_type = response.headers.get("Content-Type", "")
        
        # Content-Type から charset を抽出
        charset_match = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
        if charset_match:
            charset = charset_match.group(1).strip('"\'')
            logger.debug(f"Charset from Content-Type: {charset}")
            try:
                return response.content.decode(charset)
            except (UnicodeDecodeError, LookupError) as e:
                logger.warning(f"Failed to decode with {charset}: {e}")
        
        # HTML内の meta タグから charset を探す（バイト列のまま検索）
        content_bytes = response.content[:2048]  # 先頭2KBだけ見る
        
        # <meta charset="...">
        meta_match = re.search(
            rb'<meta[^>]+charset=["\']?([^"\'\s>]+)',
            content_bytes,
            re.IGNORECASE
        )
        if meta_match:
            charset = meta_match.group(1).decode('ascii', errors='ignore')
            logger.debug(f"Charset from meta tag: {charset}")
            try:
                return response.content.decode(charset)
            except (UnicodeDecodeError, LookupError) as e:
                logger.warning(f"Failed to decode with {charset}: {e}")
        
        # <meta http-equiv="Content-Type" content="text/html; charset=...">
        meta_match2 = re.search(
            rb'<meta[^>]+content=["\'][^"\']*charset=([^"\'\s;]+)',
            content_bytes,
            re.IGNORECASE
        )
        if meta_match2:
            charset = meta_match2.group(1).decode('ascii', errors='ignore')
            logger.debug(f"Charset from meta http-equiv: {charset}")
            try:
                return response.content.decode(charset)
            except (UnicodeDecodeError, LookupError) as e:
                logger.warning(f"Failed to decode with {charset}: {e}")
        
        # デフォルト: EUC-JP（netkeiba.comの標準）
        logger.debug("Using default charset: EUC-JP")
        try:
            return response.content.decode("euc-jp")
        except UnicodeDecodeError:
            # 最終手段: エラーを無視してデコード
            logger.warning("EUC-JP decode failed, using errors='replace'")
            return response.content.decode("euc-jp", errors="replace")
    
    def fetch_horse_laptime(self, race_id: str) -> Optional[str]:
        """
        各馬ラップタイムをAJAXエンドポイントから取得する。
        
        エンドポイント: /race/ajax_race_result_horse_laptime.html
        レスポンス形式: JSONP
        
        Args:
            race_id: 12桁のレースID
        
        Returns:
            JSONPの括弧内のJSON文字列（取得失敗時はNone）
        """
        if not re.match(r"^\d{12}$", race_id):
            raise ValueError(f"Invalid race_id format: {race_id}")
        
        url = f"{BASE_URL}/race/ajax_race_result_horse_laptime.html"
        
        # コールバック名を生成（jQuery風）
        callback_name = f"jQuery_{int(time.time() * 1000)}"
        
        params = {
            # race_id を明示的に渡す（id パラメータだけだと先頭の1頭しか返らないケースがある）
            "race_id": race_id,
            "id": race_id,
            "callback": callback_name,
            "input": "UTF-8",
            "output": "jsonp",
            "credit": "1",
            "_": str(int(time.time() * 1000)),
        }
        
        logger.debug(f"Fetching horse laptime for race_id={race_id}")
        
        try:
            response = self.get(url, params=params)
            
            if response.status_code != 200:
                logger.warning(f"Horse laptime fetch failed: HTTP {response.status_code}")
                return None
            
            # JSONP レスポンスを UTF-8 でデコード
            content = response.content.decode("utf-8", errors="replace")
            
            # JSONP を剥がす: callbackName(...); → ...
            # パターン: callback_name( ... );
            jsonp_pattern = rf"^{re.escape(callback_name)}\((.*)\);?$"
            match = re.match(jsonp_pattern, content.strip(), re.DOTALL)
            
            if match:
                json_str = match.group(1)
                logger.debug(f"Extracted JSON from JSONP, length={len(json_str)}")
                return json_str
            else:
                # callback名が違う場合も試す（サーバーが別名を返す場合）
                alt_pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*\((.*)\);?$"
                alt_match = re.match(alt_pattern, content.strip(), re.DOTALL)
                if alt_match:
                    json_str = alt_match.group(1)
                    logger.debug(f"Extracted JSON with alt pattern, length={len(json_str)}")
                    return json_str
                
                logger.warning(f"Failed to extract JSON from JSONP response: {content[:200]}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Failed to fetch horse laptime: {e}")
            return None
    
    def get_race_list_page(
        self,
        start_year: int,
        start_mon: int,
        end_year: int,
        end_mon: int,
        place_codes: list[str],
        page: int = 1,
        list_size: int = 100,
    ) -> str:
        """
        レース検索結果ページのHTMLを取得する。
        
        Args:
            start_year: 開始年
            start_mon: 開始月
            end_year: 終了年
            end_mon: 終了月
            place_codes: 場コードのリスト（例: ["01", "02", ...]）
            page: ページ番号（1始まり）
            list_size: 1ページあたりの表示件数
        
        Returns:
            HTMLコンテンツ
        """
        params = {
            "pid": "race_list",
            "word": "",
            "start_year": str(start_year),
            "start_mon": str(start_mon),
            "end_year": str(end_year),
            "end_mon": str(end_mon),
            "kyori_min": "",
            "kyori_max": "",
            "sort": "date",
            "list": str(list_size),
        }
        
        # 場コードを追加
        for i, code in enumerate(place_codes):
            params[f"jyo[{i}]"] = code
        
        # ページ番号（1ページ目は不要）
        if page > 1:
            params["page"] = str(page)
        
        response = self.get(BASE_URL, params=params)
        
        # netkeiba.com は常に EUC-JP
        html = self._decode_response(response)
        
        return html
    
    def is_logged_in(self) -> bool:
        """
        ログイン状態を確認する。
        
        Returns:
            ログインしている場合はTrue
        """
        # nkauth Cookie が存在すればログイン状態と見なす
        return "nkauth" in self.session.cookies
    
    def close(self) -> None:
        """セッションをクローズする。"""
        self.session.close()
    
    def __enter__(self) -> "NetkeibaClient":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# デフォルトクライアント（シングルトン的に使用可能）
_default_client: Optional[NetkeibaClient] = None


def get_client() -> NetkeibaClient:
    """
    デフォルトのNetkeibaClientを取得する。
    
    Returns:
        NetkeibaClient インスタンス
    """
    global _default_client
    if _default_client is None:
        _default_client = NetkeibaClient()
    return _default_client


def reset_client() -> None:
    """デフォルトクライアントをリセットする。"""
    global _default_client
    if _default_client is not None:
        _default_client.close()
        _default_client = None