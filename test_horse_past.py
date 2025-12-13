# -*- coding: utf-8 -*-
"""
test_horse_past.py

馬の過去走成績パーサーのデバッグ用スクリプト
"""

import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.fetcher import NetkeibaFetcher
from src.ingestion.parser_horse_past import HorsePastRunsParser

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

def test_single_horse(horse_id: str):
    """単一の馬をテスト"""
    url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
    
    print("=" * 80)
    print(f"Testing horse_id: {horse_id}")
    print(f"URL: {url}")
    print("=" * 80)
    
    with NetkeibaFetcher() as fetcher:
        # HTML取得
        print("\n[1/3] Fetching HTML...")
        try:
            soup = fetcher.fetch_soup(url)
            print("✅ HTML fetched successfully")
        except Exception as e:
            print(f"❌ Failed to fetch HTML: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # HTMLを保存
        html_file = f"debug_{horse_id}.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(str(soup.prettify()))
        print(f"✅ HTML saved to: {html_file}")
        
        # テーブル特定
        print("\n[2/3] Finding past runs table...")
        parser = HorsePastRunsParser()
        table = parser._extract_past_runs_table(soup)
        if table is None:
            print("❌ Past runs table not found!")
            print("\nAvailable tables:")
            tables = soup.find_all("table")
            for i, t in enumerate(tables):
                thead = t.find("thead")
                if thead:
                    headers = [th.get_text(strip=True) for th in thead.find_all("th")]
                    print(f"  Table {i}: {', '.join(headers[:5])}...")
                else:
                    print(f"  Table {i}: (no thead)")
            return
        else:
            print("✅ Past runs table found")
            thead = table.find("thead")
            if thead:
                headers = [th.get_text(strip=True) for th in thead.find_all("th")]
                print(f"   Headers: {headers}")
        
        # パース
        print("\n[3/3] Parsing table...")
        try:
            df = parser.parse(soup, horse_id)
            print(f"✅ Successfully parsed {len(df)} rows")
            print("\nDataFrame columns:")
            print(f"  {list(df.columns)}")
            print("\nFirst row:")
            if len(df) > 0:
                print(df.iloc[0].to_dict())
            print("\nDataFrame head:")
            print(df.head())
            
            # CSVに保存
            csv_file = f"debug_{horse_id}.csv"
            df.to_csv(csv_file, index=False, encoding="utf-8-sig")
            print(f"\n✅ Data saved to: {csv_file}")
            
        except Exception as e:
            print(f"❌ Parse error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # 失敗した馬IDでテスト
    test_horse_ids = [
        "2021100159",
        # "2021100265",
        # "2021100648",
    ]
    
    for horse_id in test_horse_ids:
        test_single_horse(horse_id)
        print("\n" + "=" * 80 + "\n")