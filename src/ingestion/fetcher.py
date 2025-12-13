# -*- coding: utf-8 -*-
"""
netkeiba.com スクレイピング用 HTTP クライアント（EUC-JP 対応版）

改良点:
- response.content (bytes) を使用して文字化け防止
- requests の自動エンコード判定を無効化（response.encoding = None）
- BeautifulSoup に bytes のまま渡し、EUC-JP 判定を任せる
- 安定動作用のログ、バックオフ、UAローテーション
- ログ設定を外部化（モジュールレベルの basicConfig を削除）
"""

import time
import random
import logging
from typing import Optional
import requests
from bs4 import BeautifulSoup

# ログ設定は外部で行う（ライブラリとして使いやすくする）
logger = logging.getLogger(__name__)

# ----------------------------------
# 最新 netkeiba 仕様に対応する UA ローテーション
# 2024/11/30 の仕様変更に対応
# ----------------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
]


class NetkeibaFetcher:
    """
    netkeiba のレースページ等を取得する HTTP クライアント
    
    Features:
        - EUC-JP 完全対応（response.content + BeautifulSoup の自動判定）
        - User-Agent ローテーション（2024/11/30 仕様変更対応）
        - 指数バックオフ + ジッター付きリトライ
        - 403/429/503 エラーの適切な対応
    
    Example:
        >>> fetcher = NetkeibaFetcher()
        >>> soup = fetcher.fetch_soup("https://db.netkeiba.com/race/202301010101/")
        >>> fetcher.close()
    """

    def __init__(
        self,
        sleep_min: float = 2.0,
        sleep_max: float = 3.5,
        max_retry: int = 3,
        timeout: int = 10
    ):
        """
        Args:
            sleep_min: リクエスト間隔の最小値（秒）
            sleep_max: リクエスト間隔の最大値（秒）
            max_retry: リトライ回数
            timeout: タイムアウト（秒）
        """
        self.session = requests.Session()
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self.max_retry = max_retry
        self.timeout = timeout

        logger.info(
            f"NetkeibaFetcher initialized: sleep={sleep_min}-{sleep_max}s, "
            f"retry={max_retry}, timeout={timeout}s"
        )

    def _get(self, url: str) -> requests.Response:
        """
        低レベル HTTP GET
        
        Args:
            url: 取得する URL
        
        Returns:
            Response オブジェクト
        """
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        resp = self.session.get(url, headers=headers, timeout=self.timeout)

        # ★★★ 最重要：エンコーディングを requests に判定させない ★★★
        # これにより response.content が正しく EUC-JP のバイト列として取得される
        resp.encoding = None

        return resp

    def fetch_soup(self, url: str) -> BeautifulSoup:
        """
        URL から BeautifulSoup を返す（リトライ込み）
        
        Args:
            url: 取得する URL
        
        Returns:
            BeautifulSoup オブジェクト
        
        Raises:
            RuntimeError: リトライ回数を超えても取得できなかった場合
        
        Example:
            >>> soup = fetcher.fetch_soup("https://db.netkeiba.com/race/202301010101/")
        """
        for attempt in range(1, self.max_retry + 1):
            try:
                resp = self._get(url)

                # 403/429/503 の場合は待機してリトライ
                if resp.status_code in (403, 429, 503):
                    wait = random.uniform(2, 5) * attempt  # 指数的に増加
                    logger.warning(
                        f"HTTP {resp.status_code} at {url}. "
                        f"Retrying in {wait:.1f}s... (attempt {attempt}/{self.max_retry})"
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()

                # ★★★ ここが決定的に重要：content (bytes) を渡す ★★★
                # BeautifulSoup が EUC-JP を自動判定してくれる
                soup = BeautifulSoup(resp.content, "html.parser")

                logger.info(f"Successfully fetched {url}")
                
                # 最低限のインターバル
                time.sleep(random.uniform(self.sleep_min, self.sleep_max))
                
                return soup

            except requests.exceptions.Timeout as e:
                logger.error(
                    f"Timeout fetching {url} (attempt {attempt}/{self.max_retry}): {e}"
                )
                if attempt == self.max_retry:
                    raise RuntimeError(f"Timeout after {self.max_retry} retries: {url}")
                time.sleep(random.uniform(1.0, 3.0))

            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Request failed for {url} (attempt {attempt}/{self.max_retry}): {e}"
                )
                if attempt == self.max_retry:
                    raise RuntimeError(f"Failed to fetch {url} after {self.max_retry} retries: {e}")
                time.sleep(random.uniform(1.0, 3.0))

            except Exception as e:
                logger.error(
                    f"Unexpected error fetching {url} (attempt {attempt}/{self.max_retry}): {e}",
                    exc_info=True
                )
                if attempt == self.max_retry:
                    raise

        raise RuntimeError(f"Failed to fetch {url} after {self.max_retry} attempts")

    def close(self):
        """セッションをクローズ"""
        logger.info("Closing session")
        self.session.close()

    def __enter__(self):
        """Context manager: with 文で使用可能"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: 自動クローズ"""
        self.close()
        return False

    def __del__(self):
        """デストラクタ: 念のためクローズ"""
        try:
            self.close()
        except:
            pass
