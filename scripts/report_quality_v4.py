#!/usr/bin/env python3
"""
report_quality_v4.py - CLI for Data Quality Report Generator

Master データの品質レポートを生成します。

【チェック項目】
- テーブル統計: 行数、NULL率
- 参照整合性: race_results vs horses/jockeys/trainers
- 年別カバレッジ: 各年のデータ量
- 血統カバレッジ: 5代血統の世代別網羅率

【出力】
- コンソール: サマリーレポート
- artifacts/quality_report_v4.json: 詳細レポート (JSON)
- artifacts/quality_report_v4.csv: 概要 (CSV)

Usage:
    python scripts/report_quality_v4.py --db netkeiba.db
    python scripts/report_quality_v4.py --db netkeiba.db --output artifacts/
    python scripts/report_quality_v4.py --db netkeiba.db --json-only
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features_v4 import generate_quality_report, MasterQualityReporter


logger = logging.getLogger(__name__)


def print_report_summary(report: dict) -> None:
    """Print formatted report summary to console."""
    print("\n" + "=" * 70)
    print("DATA QUALITY REPORT - FeaturePack v1")
    print("=" * 70)

    # Table Statistics
    print("\n📊 Table Statistics:")
    print("-" * 50)
    for table, stats in report.get("tables", {}).items():
        print(f"  {table}:")
        print(f"    Rows: {stats.get('row_count', 0):,}")
        null_cols = stats.get("null_columns", {})
        if null_cols:
            high_null = [(c, r) for c, r in null_cols.items() if r and r > 0.1]
            if high_null:
                print(f"    High NULL rates (>10%):")
                for col, rate in sorted(high_null, key=lambda x: -x[1])[:5]:
                    print(f"      - {col}: {rate:.1%}")

    # Reference Integrity
    print("\n🔗 Reference Integrity:")
    print("-" * 50)
    refs = report.get("references", {})
    for ref_name, ref_data in refs.items():
        matched = ref_data.get("matched", 0)
        total = ref_data.get("total", 0)
        match_rate = matched / total if total > 0 else 0
        status = "✅" if match_rate >= 0.95 else "⚠️" if match_rate >= 0.8 else "❌"
        print(f"  {status} {ref_name}: {matched:,}/{total:,} ({match_rate:.1%})")

    # Year Coverage
    print("\n📅 Year Coverage:")
    print("-" * 50)
    years = report.get("year_coverage", {})
    if years:
        for year in sorted(years.keys()):
            data = years[year]
            print(f"  {year}: {data.get('races', 0):,} races, {data.get('entries', 0):,} entries")

    # Pedigree Coverage
    print("\n🧬 Pedigree Coverage (5-gen):")
    print("-" * 50)
    pedigree = report.get("pedigree_coverage", {})
    if pedigree:
        total_horses = pedigree.get("total_horses", 0)
        horses_with_pedigree = pedigree.get("horses_with_pedigree", 0)
        coverage = horses_with_pedigree / total_horses if total_horses > 0 else 0
        print(f"  Total horses: {total_horses:,}")
        print(f"  With pedigree: {horses_with_pedigree:,} ({coverage:.1%})")

        gen_coverage = pedigree.get("generation_coverage", {})
        if gen_coverage:
            print("  By generation:")
            for gen in sorted(gen_coverage.keys()):
                rate = gen_coverage[gen]
                bar = "█" * int(rate * 20)
                print(f"    Gen {gen}: {bar} {rate:.1%}")

    print("\n" + "=" * 70)


def save_report(report: dict, output_dir: str) -> None:
    """Save report to JSON and CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_path / f"quality_report_v4_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved JSON report to: {json_path}")

    # Save CSV summary
    csv_path = output_path / f"quality_report_v4_{timestamp}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Metric,Value\n")

        # Table counts
        for table, stats in report.get("tables", {}).items():
            f.write(f"{table}_rows,{stats.get('row_count', 0)}\n")

        # Reference integrity
        for ref_name, ref_data in report.get("references", {}).items():
            matched = ref_data.get("matched", 0)
            total = ref_data.get("total", 0)
            rate = matched / total if total > 0 else 0
            f.write(f"{ref_name}_match_rate,{rate:.4f}\n")

        # Year coverage
        for year, data in report.get("year_coverage", {}).items():
            f.write(f"races_{year},{data.get('races', 0)}\n")
            f.write(f"entries_{year},{data.get('entries', 0)}\n")

        # Pedigree coverage
        pedigree = report.get("pedigree_coverage", {})
        if pedigree:
            total = pedigree.get("total_horses", 0)
            with_ped = pedigree.get("horses_with_pedigree", 0)
            f.write(f"pedigree_coverage,{with_ped / total if total > 0 else 0:.4f}\n")

    logger.info(f"Saved CSV summary to: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Data Quality Report for FeaturePack v1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report and print to console
  python scripts/report_quality_v4.py --db netkeiba.db

  # Save report to artifacts directory
  python scripts/report_quality_v4.py --db netkeiba.db --output artifacts/

  # Generate JSON only (no console output)
  python scripts/report_quality_v4.py --db netkeiba.db --output artifacts/ --json-only
"""
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to SQLite DB (e.g., netkeiba.db)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for report files (optional)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only output JSON (no console summary)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Validate database path
    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    logger.info(f"Generating quality report for: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # Generate report (print_output=False to suppress default output)
        report = generate_quality_report(conn, print_output=False)
        report_dict = report.to_dict()

        # Print summary
        if not args.json_only:
            print_report_summary(report_dict)

        # Save files if output directory specified
        if args.output:
            save_report(report_dict, args.output)
        elif args.json_only:
            # Print JSON to stdout
            print(json.dumps(report_dict, ensure_ascii=False, indent=2))

    finally:
        conn.close()


if __name__ == "__main__":
    main()
