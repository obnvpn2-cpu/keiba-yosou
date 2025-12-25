# -*- coding: utf-8 -*-
"""
quality_report.py - Master Data Quality Report

マスターデータの品質をレポートするモジュール。
以下のチェックを実施:

1. カバレッジ: 各マスターテーブルのレコード数
2. 欠損率: 各カラムの NULL 率
3. 参照整合性: race_results の horse_id/jockey_id/trainer_id がマスターに存在するか
4. 血統カバレッジ: horse_pedigree のカバー率
5. 期間カバレッジ: 年別のレース数・出走数

【出力例】
================================================================================
Master Data Quality Report
================================================================================

[Table Coverage]
  races:        12,345 rows
  race_results: 234,567 rows
  horses:       45,678 rows (coverage: 95.2%)
  jockeys:      1,234 rows (coverage: 98.5%)
  trainers:     567 rows (coverage: 97.8%)
  horse_pedigree: 234,567 rows (5-gen coverage: 82.3%)

[Reference Integrity]
  horse_id in race_results → horses: 95.2%
  jockey_id in race_results → jockeys: 98.5%
  trainer_id in race_results → trainers: 97.8%

[Null Rate by Column]
  horses.birth_date: 2.3%
  horses.sire_id: 5.1%
  jockeys.affiliation: 1.2%

[Year Coverage]
  2021: 3,456 races, 56,789 entries
  2022: 3,567 races, 58,901 entries
  2023: 3,678 races, 60,123 entries
  2024: 3,789 races, 62,345 entries

================================================================================
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TableStats:
    """テーブル統計"""
    table_name: str
    row_count: int
    column_count: int
    null_rates: Dict[str, float]  # column -> null rate


@dataclass
class ReferenceIntegrity:
    """参照整合性"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    match_count: int
    total_count: int
    match_rate: float


@dataclass
class YearCoverage:
    """年別カバレッジ"""
    year: int
    race_count: int
    entry_count: int
    horse_count: int
    jockey_count: int
    trainer_count: int


@dataclass
class InvalidResultStats:
    """無効レコード統計"""
    total_entries: int
    invalid_count: int
    invalid_rate: float
    breakdown: Dict[str, int]  # finish_status -> count


@dataclass
class QualityReport:
    """品質レポート全体"""
    generated_at: str
    table_stats: Dict[str, TableStats]
    reference_integrity: List[ReferenceIntegrity]
    year_coverage: List[YearCoverage]
    pedigree_coverage: Dict[str, Any]
    invalid_results: Optional[InvalidResultStats]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """JSONシリアライズ可能な辞書に変換"""
        return {
            "generated_at": self.generated_at,
            "tables": {
                name: {
                    "table_name": stats.table_name,
                    "row_count": stats.row_count,
                    "column_count": stats.column_count,
                    "null_columns": stats.null_rates,
                }
                for name, stats in self.table_stats.items()
            },
            "references": {
                f"{ref.source_table}.{ref.source_column}_to_{ref.target_table}": {
                    "source_table": ref.source_table,
                    "source_column": ref.source_column,
                    "target_table": ref.target_table,
                    "target_column": ref.target_column,
                    "matched": ref.match_count,
                    "total": ref.total_count,
                    "match_rate": ref.match_rate,
                }
                for ref in self.reference_integrity
            },
            "year_coverage": {
                str(yc.year): {
                    "races": yc.race_count,
                    "entries": yc.entry_count,
                    "horses": yc.horse_count,
                    "jockeys": yc.jockey_count,
                    "trainers": yc.trainer_count,
                }
                for yc in self.year_coverage
            },
            "pedigree_coverage": {
                "total_horses": self.pedigree_coverage.get("total_horses", 0),
                "horses_with_pedigree": self.pedigree_coverage.get("horses_with_pedigree", 0),
                "coverage_rate": self.pedigree_coverage.get("coverage_rate", 0.0),
                "ancestor_count": self.pedigree_coverage.get("ancestor_count", 0),
                "generation_coverage": {
                    gen: data.get("coverage", 0.0) if isinstance(data, dict) else data
                    for gen, data in self.pedigree_coverage.get("generation_coverage", {}).items()
                },
            },
            "invalid_results": {
                "total_entries": self.invalid_results.total_entries,
                "invalid_count": self.invalid_results.invalid_count,
                "invalid_rate": self.invalid_results.invalid_rate,
                "breakdown": self.invalid_results.breakdown,
            } if self.invalid_results else None,
            "warnings": self.warnings,
        }


# =============================================================================
# Quality Reporter Class
# =============================================================================

class MasterQualityReporter:
    """
    マスターデータ品質レポーター

    データベースの各テーブルを分析し、品質レポートを生成する。
    """

    # チェック対象テーブル
    TABLES = [
        "races",
        "race_results",
        "horses",
        "jockeys",
        "trainers",
        "horse_pedigree",
        "fetch_status",
        "lap_times",
        "horse_laps",
    ]

    # 参照整合性チェック
    REFERENCE_CHECKS = [
        ("race_results", "horse_id", "horses", "horse_id"),
        ("race_results", "jockey_id", "jockeys", "jockey_id"),
        ("race_results", "trainer_id", "trainers", "trainer_id"),
        ("horse_pedigree", "horse_id", "horses", "horse_id"),
    ]

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.warnings: List[str] = []

    # =========================================================================
    # Table Existence Check
    # =========================================================================

    def _table_exists(self, table_name: str) -> bool:
        """テーブルが存在するかチェック"""
        cursor = self.conn.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name=?""",
            (table_name,)
        )
        return cursor.fetchone() is not None

    def _get_table_columns(self, table_name: str) -> List[str]:
        """テーブルのカラム一覧を取得"""
        cursor = self.conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]

    # =========================================================================
    # Table Stats
    # =========================================================================

    def get_table_stats(self, table_name: str) -> Optional[TableStats]:
        """テーブルの統計情報を取得"""
        if not self._table_exists(table_name):
            logger.warning("Table %s does not exist", table_name)
            return None

        try:
            # Row count
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            # Column info
            columns = self._get_table_columns(table_name)
            column_count = len(columns)

            # Null rates (サンプリングで高速化)
            null_rates: Dict[str, float] = {}

            if row_count > 0:
                # カラムごとの NULL 率を計算
                for col in columns:
                    cursor = self.conn.execute(f"""
                        SELECT
                            SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
                        FROM {table_name}
                    """)
                    null_rate = cursor.fetchone()[0] or 0.0
                    if null_rate > 0:
                        null_rates[col] = null_rate

            return TableStats(
                table_name=table_name,
                row_count=row_count,
                column_count=column_count,
                null_rates=null_rates,
            )

        except Exception as e:
            logger.error("Failed to get stats for %s: %s", table_name, e)
            self.warnings.append(f"Failed to get stats for {table_name}: {e}")
            return None

    # =========================================================================
    # Reference Integrity
    # =========================================================================

    def check_reference_integrity(
        self,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str,
    ) -> Optional[ReferenceIntegrity]:
        """参照整合性をチェック"""
        if not self._table_exists(source_table):
            return None
        if not self._table_exists(target_table):
            return None

        try:
            # Total count (non-null)
            cursor = self.conn.execute(f"""
                SELECT COUNT(DISTINCT {source_column})
                FROM {source_table}
                WHERE {source_column} IS NOT NULL
            """)
            total_count = cursor.fetchone()[0]

            if total_count == 0:
                return ReferenceIntegrity(
                    source_table=source_table,
                    source_column=source_column,
                    target_table=target_table,
                    target_column=target_column,
                    match_count=0,
                    total_count=0,
                    match_rate=0.0,
                )

            # Match count
            cursor = self.conn.execute(f"""
                SELECT COUNT(DISTINCT s.{source_column})
                FROM {source_table} s
                INNER JOIN {target_table} t ON s.{source_column} = t.{target_column}
                WHERE s.{source_column} IS NOT NULL
            """)
            match_count = cursor.fetchone()[0]

            match_rate = match_count / total_count if total_count > 0 else 0.0

            return ReferenceIntegrity(
                source_table=source_table,
                source_column=source_column,
                target_table=target_table,
                target_column=target_column,
                match_count=match_count,
                total_count=total_count,
                match_rate=match_rate,
            )

        except Exception as e:
            logger.error(
                "Failed to check reference %s.%s → %s.%s: %s",
                source_table, source_column, target_table, target_column, e
            )
            self.warnings.append(
                f"Failed to check reference {source_table}.{source_column} "
                f"→ {target_table}.{target_column}: {e}"
            )
            return None

    # =========================================================================
    # Year Coverage
    # =========================================================================

    def get_year_coverage(self) -> List[YearCoverage]:
        """年別カバレッジを取得"""
        if not self._table_exists("races") or not self._table_exists("race_results"):
            return []

        try:
            sql = """
            SELECT
                CAST(SUBSTR(r.race_id, 1, 4) AS INTEGER) as year,
                COUNT(DISTINCT r.race_id) as race_count,
                COUNT(*) as entry_count,
                COUNT(DISTINCT rr.horse_id) as horse_count,
                COUNT(DISTINCT rr.jockey_id) as jockey_count,
                COUNT(DISTINCT rr.trainer_id) as trainer_count
            FROM races r
            JOIN race_results rr ON r.race_id = rr.race_id
            GROUP BY year
            ORDER BY year
            """

            df = pd.read_sql_query(sql, self.conn)

            return [
                YearCoverage(
                    year=int(row["year"]),
                    race_count=int(row["race_count"]),
                    entry_count=int(row["entry_count"]),
                    horse_count=int(row["horse_count"]),
                    jockey_count=int(row["jockey_count"]),
                    trainer_count=int(row["trainer_count"]),
                )
                for _, row in df.iterrows()
            ]

        except Exception as e:
            logger.error("Failed to get year coverage: %s", e)
            self.warnings.append(f"Failed to get year coverage: {e}")
            return []

    # =========================================================================
    # Invalid Result Stats (中/除/取/降格)
    # =========================================================================

    def get_invalid_result_stats(self) -> Optional[InvalidResultStats]:
        """
        無効レコード統計を取得

        無効レコード:
        - finish_order IS NULL
        - finish_status IN ('中', '除', '取') または '%降%' を含む
        """
        if not self._table_exists("race_results"):
            return None

        try:
            # 総エントリ数
            cursor = self.conn.execute("SELECT COUNT(*) FROM race_results")
            total_entries = cursor.fetchone()[0]

            if total_entries == 0:
                return InvalidResultStats(
                    total_entries=0,
                    invalid_count=0,
                    invalid_rate=0.0,
                    breakdown={},
                )

            # finish_order IS NULL の件数
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM race_results WHERE finish_order IS NULL"
            )
            null_finish_count = cursor.fetchone()[0]

            # finish_status 別の内訳（無効なもののみ）
            # finish_status がテーブルに存在するかチェック
            columns = self._get_table_columns("race_results")
            breakdown: Dict[str, int] = {}

            if "finish_status" in columns:
                cursor = self.conn.execute("""
                    SELECT finish_status, COUNT(*) as cnt
                    FROM race_results
                    WHERE finish_status IN ('中', '除', '取')
                       OR finish_status LIKE '%降%'
                    GROUP BY finish_status
                    ORDER BY cnt DESC
                """)
                for row in cursor.fetchall():
                    status = row[0] or "NULL"
                    breakdown[status] = row[1]

                # finish_order IS NULL で finish_status も取得
                cursor = self.conn.execute("""
                    SELECT finish_status, COUNT(*) as cnt
                    FROM race_results
                    WHERE finish_order IS NULL
                    GROUP BY finish_status
                    ORDER BY cnt DESC
                """)
                null_breakdown = {}
                for row in cursor.fetchall():
                    status = row[0] or "(NULL status)"
                    null_breakdown[status] = row[1]

                # 既存の breakdown と null_breakdown をマージ（重複を避ける）
                # breakdown にはすでに中/除/取が含まれているので、
                # null_finish_count から既に含まれているものを除外
                breakdown["(NULL finish_order)"] = null_finish_count

            else:
                # finish_status カラムがない場合
                breakdown["(NULL finish_order)"] = null_finish_count

            # 総無効件数（finish_order IS NULL or invalid finish_status）
            # 重複を避けるため、SQL で正確にカウント
            if "finish_status" in columns:
                cursor = self.conn.execute("""
                    SELECT COUNT(*) FROM race_results
                    WHERE finish_order IS NULL
                       OR finish_status IN ('中', '除', '取')
                       OR finish_status LIKE '%降%'
                """)
            else:
                cursor = self.conn.execute(
                    "SELECT COUNT(*) FROM race_results WHERE finish_order IS NULL"
                )
            invalid_count = cursor.fetchone()[0]

            invalid_rate = invalid_count / total_entries if total_entries > 0 else 0.0

            return InvalidResultStats(
                total_entries=total_entries,
                invalid_count=invalid_count,
                invalid_rate=invalid_rate,
                breakdown=breakdown,
            )

        except Exception as e:
            logger.error("Failed to get invalid result stats: %s", e)
            self.warnings.append(f"Failed to get invalid result stats: {e}")
            return None

    # =========================================================================
    # Pedigree Coverage
    # =========================================================================

    def get_pedigree_coverage(self) -> Dict[str, Any]:
        """血統カバレッジを取得"""
        result: Dict[str, Any] = {
            "horses_with_pedigree": 0,
            "total_horses": 0,
            "coverage_rate": 0.0,
            "generation_coverage": {},
            "ancestor_count": 0,
        }

        if not self._table_exists("horse_pedigree"):
            return result

        try:
            # 血統登録済み馬数
            cursor = self.conn.execute("""
                SELECT COUNT(DISTINCT horse_id)
                FROM horse_pedigree
            """)
            result["horses_with_pedigree"] = cursor.fetchone()[0]

            # 全馬数
            if self._table_exists("horses"):
                cursor = self.conn.execute("SELECT COUNT(*) FROM horses")
                result["total_horses"] = cursor.fetchone()[0]
            elif self._table_exists("race_results"):
                cursor = self.conn.execute(
                    "SELECT COUNT(DISTINCT horse_id) FROM race_results"
                )
                result["total_horses"] = cursor.fetchone()[0]

            if result["total_horses"] > 0:
                result["coverage_rate"] = (
                    result["horses_with_pedigree"] / result["total_horses"]
                )

            # 世代別カバレッジ
            cursor = self.conn.execute("""
                SELECT generation, COUNT(*) as cnt
                FROM horse_pedigree
                GROUP BY generation
                ORDER BY generation
            """)
            for row in cursor.fetchall():
                gen = row[0]
                cnt = row[1]
                expected = result["horses_with_pedigree"] * (2 ** gen)
                coverage = cnt / expected if expected > 0 else 0.0
                result["generation_coverage"][f"gen{gen}"] = {
                    "count": cnt,
                    "expected": expected,
                    "coverage": coverage,
                }

            # 祖先総数
            cursor = self.conn.execute("SELECT COUNT(*) FROM horse_pedigree")
            result["ancestor_count"] = cursor.fetchone()[0]

        except Exception as e:
            logger.error("Failed to get pedigree coverage: %s", e)
            self.warnings.append(f"Failed to get pedigree coverage: {e}")

        return result

    # =========================================================================
    # Generate Full Report
    # =========================================================================

    def generate_report(self) -> QualityReport:
        """完全な品質レポートを生成"""
        logger.info("Generating quality report...")

        # テーブル統計
        table_stats: Dict[str, TableStats] = {}
        for table in self.TABLES:
            stats = self.get_table_stats(table)
            if stats:
                table_stats[table] = stats

        # 参照整合性
        reference_integrity: List[ReferenceIntegrity] = []
        for src_table, src_col, tgt_table, tgt_col in self.REFERENCE_CHECKS:
            ref = self.check_reference_integrity(src_table, src_col, tgt_table, tgt_col)
            if ref:
                reference_integrity.append(ref)

        # 年別カバレッジ
        year_coverage = self.get_year_coverage()

        # 血統カバレッジ
        pedigree_coverage = self.get_pedigree_coverage()

        # 無効レコード統計
        invalid_results = self.get_invalid_result_stats()

        return QualityReport(
            generated_at=datetime.now().isoformat(),
            table_stats=table_stats,
            reference_integrity=reference_integrity,
            year_coverage=year_coverage,
            pedigree_coverage=pedigree_coverage,
            invalid_results=invalid_results,
            warnings=self.warnings,
        )

    # =========================================================================
    # Print Report
    # =========================================================================

    def print_report(self, report: Optional[QualityReport] = None) -> str:
        """レポートを文字列として出力"""
        if report is None:
            report = self.generate_report()

        lines = []
        lines.append("=" * 80)
        lines.append("Master Data Quality Report")
        lines.append(f"Generated: {report.generated_at}")
        lines.append("=" * 80)
        lines.append("")

        # Table Coverage
        lines.append("[Table Coverage]")
        for name, stats in report.table_stats.items():
            lines.append(f"  {name:20s}: {stats.row_count:>10,} rows, {stats.column_count} columns")
        lines.append("")

        # Reference Integrity
        lines.append("[Reference Integrity]")
        for ref in report.reference_integrity:
            lines.append(
                f"  {ref.source_table}.{ref.source_column} → {ref.target_table}.{ref.target_column}: "
                f"{ref.match_count:,}/{ref.total_count:,} ({ref.match_rate*100:.1f}%)"
            )
        lines.append("")

        # Null Rates (only significant ones)
        lines.append("[Null Rates > 1%]")
        for name, stats in report.table_stats.items():
            for col, rate in stats.null_rates.items():
                if rate > 0.01:  # 1% 以上のみ表示
                    lines.append(f"  {name}.{col}: {rate*100:.1f}%")
        lines.append("")

        # Year Coverage
        lines.append("[Year Coverage]")
        for yc in report.year_coverage:
            lines.append(
                f"  {yc.year}: {yc.race_count:,} races, {yc.entry_count:,} entries, "
                f"{yc.horse_count:,} horses, {yc.jockey_count:,} jockeys"
            )
        lines.append("")

        # Pedigree Coverage
        lines.append("[Pedigree Coverage]")
        pc = report.pedigree_coverage
        lines.append(
            f"  Horses with pedigree: {pc['horses_with_pedigree']:,} / "
            f"{pc['total_horses']:,} ({pc['coverage_rate']*100:.1f}%)"
        )
        lines.append(f"  Total ancestors: {pc['ancestor_count']:,}")
        for gen_name, gen_data in pc.get("generation_coverage", {}).items():
            lines.append(
                f"    {gen_name}: {gen_data['count']:,} / {gen_data['expected']:,} "
                f"({gen_data['coverage']*100:.1f}%)"
            )
        lines.append("")

        # Invalid Results (中/除/取/降格)
        if report.invalid_results:
            ir = report.invalid_results
            lines.append("[Invalid Results (excluded from feature_table_v4)]")
            lines.append(
                f"  Total entries: {ir.total_entries:,}, "
                f"Invalid: {ir.invalid_count:,} ({ir.invalid_rate*100:.2f}%)"
            )
            if ir.breakdown:
                lines.append("  Breakdown by finish_status:")
                for status, count in sorted(ir.breakdown.items(), key=lambda x: -x[1]):
                    lines.append(f"    {status}: {count:,}")
            lines.append("")

        # Warnings
        if report.warnings:
            lines.append("[Warnings]")
            for warning in report.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


# =============================================================================
# Convenience Function
# =============================================================================

def generate_quality_report(conn: sqlite3.Connection, print_output: bool = True) -> QualityReport:
    """
    品質レポートを生成するコンビニエンス関数

    Args:
        conn: SQLite 接続
        print_output: True の場合、レポートを標準出力に表示

    Returns:
        QualityReport オブジェクト
    """
    reporter = MasterQualityReporter(conn)
    report = reporter.generate_report()

    if print_output:
        print(reporter.print_report(report))

    return report


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Generate Master Data Quality Report")
    parser.add_argument("--db", default="netkeiba.db", help="Database path")
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    if not Path(args.db).exists():
        logger.error("Database not found: %s", args.db)
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    try:
        reporter = MasterQualityReporter(conn)
        report = reporter.generate_report()
        output_text = reporter.print_report(report)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_text)
            logger.info("Report saved to %s", args.output)
        else:
            print(output_text)

    finally:
        conn.close()
