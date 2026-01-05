#!/usr/bin/env python3
"""
run_pre_race_day.py - Pre-race Day Operation Runner (Step B)

日次の pre_race 運用を1コマンドでチェック・起動するスクリプト。
素材/モデルの存在確認を行い、不足時は具体的なコマンド案内を表示する。

Usage:
    # 基本: 指定日のチェックとUI起動案内
    python scripts/run_pre_race_day.py --date 2024-12-29

    # DBパス指定
    python scripts/run_pre_race_day.py --db netkeiba.db --date 2024-12-29

    # モデルディレクトリ指定
    python scripts/run_pre_race_day.py --date 2024-12-29 --models models/

    # デモモードの確認
    python scripts/run_pre_race_day.py --demo
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEFAULT_DB_PATH = PROJECT_ROOT / "netkeiba.db"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "pre_race"

# Required models
REQUIRED_MODELS = [
    ("target_win", ["lgbm_target_win_v4.txt", "lgbm_target_win.txt"]),
    ("target_in3", ["lgbm_target_in3_v4.txt", "lgbm_target_in3.txt"]),
]

# UI defaults
UI_HOST = "localhost"
UI_PORT = 8080


# =============================================================================
# Color Output (for CLI)
# =============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)"""
        cls.GREEN = cls.YELLOW = cls.RED = cls.CYAN = cls.BOLD = cls.RESET = ""


def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")


def print_ok(text: str):
    """Print success message"""
    print(f"  {Colors.GREEN}✓{Colors.RESET} {text}")


def print_warn(text: str):
    """Print warning message"""
    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {text}")


def print_error(text: str):
    """Print error message"""
    print(f"  {Colors.RED}✗{Colors.RESET} {text}")


def print_info(text: str):
    """Print info message"""
    print(f"  {Colors.CYAN}→{Colors.RESET} {text}")


def print_cmd(text: str):
    """Print command to run"""
    print(f"    {Colors.CYAN}{text}{Colors.RESET}")


# =============================================================================
# Check Functions
# =============================================================================

def check_db(db_path: Path) -> Tuple[bool, str]:
    """Check if database exists"""
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        return True, f"Database found: {db_path.name} ({size_mb:.1f} MB)"
    return False, f"Database not found: {db_path}"


def check_models(models_dir: Path) -> Tuple[bool, List[str], List[str]]:
    """
    Check if required models exist

    Returns:
        (all_found, found_models, missing_models)
    """
    found = []
    missing = []

    for target, filenames in REQUIRED_MODELS:
        found_any = False
        for filename in filenames:
            model_path = models_dir / filename
            if model_path.exists():
                found.append(f"{target}: {filename}")
                found_any = True
                break
        if not found_any:
            missing.append(target)

    return len(missing) == 0, found, missing


def check_materials(date: str) -> Tuple[bool, Dict[str, any]]:
    """
    Check if pre_race materials exist for the date

    Returns:
        (exists, info_dict)
    """
    date_dir = ARTIFACTS_DIR / date

    if not date_dir.exists():
        return False, {"date_dir": str(date_dir), "exists": False}

    # Check for summary file
    summary_file = date_dir / f"summary_{date}.json"

    # Check for race files
    race_files = list(date_dir.glob("race_*.json"))

    info = {
        "date_dir": str(date_dir),
        "exists": True,
        "summary_exists": summary_file.exists(),
        "n_race_files": len(race_files),
        "race_files": [f.name for f in race_files[:5]],  # First 5
    }

    # At least one race file is required
    if len(race_files) > 0:
        return True, info

    return False, info


def check_demo_data() -> Tuple[bool, Dict[str, any]]:
    """Check if demo data exists"""
    demo_dir = ARTIFACTS_DIR / "demo"

    if not demo_dir.exists():
        return False, {"demo_dir": str(demo_dir), "exists": False}

    race_files = list(demo_dir.glob("race_*.json"))
    summary_file = demo_dir / "summary_demo.json"

    info = {
        "demo_dir": str(demo_dir),
        "exists": True,
        "summary_exists": summary_file.exists(),
        "n_race_files": len(race_files),
    }

    return len(race_files) > 0, info


# =============================================================================
# Main Logic
# =============================================================================

def run_check(
    date: Optional[str],
    db_path: Path,
    models_dir: Path,
    demo_mode: bool = False,
) -> int:
    """
    Run pre-race day checks and show guidance

    Returns:
        0 if all checks pass, 1 otherwise
    """
    all_ok = True
    guidance_needed = []

    print_header("Pre-race Day Check")
    print(f"  Date:       {date or '(demo mode)'}")
    print(f"  Database:   {db_path}")
    print(f"  Models:     {models_dir}")
    print(f"  Artifacts:  {ARTIFACTS_DIR}")

    # =========================================================================
    # 1. Database Check
    # =========================================================================
    print_header("1. Database Check")

    db_ok, db_msg = check_db(db_path)
    if db_ok:
        print_ok(db_msg)
    else:
        print_error(db_msg)
        all_ok = False
        guidance_needed.append("db")

    # =========================================================================
    # 2. Model Check
    # =========================================================================
    print_header("2. Model Check")

    models_ok, found_models, missing_models = check_models(models_dir)

    for m in found_models:
        print_ok(m)

    if not models_ok:
        for m in missing_models:
            print_error(f"{m}: NOT FOUND")
        all_ok = False
        guidance_needed.append("models")
    else:
        print_ok("All required models found")

    # =========================================================================
    # 3. Materials Check
    # =========================================================================
    print_header("3. Pre-race Materials Check")

    if demo_mode:
        # Demo mode check
        demo_ok, demo_info = check_demo_data()
        if demo_ok:
            print_ok(f"Demo data found: {demo_info['n_race_files']} race file(s)")
        else:
            print_error("Demo data not found")
            print_info(f"Expected location: {demo_info['demo_dir']}")
            all_ok = False
            guidance_needed.append("demo")
    elif date:
        # Normal date check
        mat_ok, mat_info = check_materials(date)

        if mat_ok:
            print_ok(f"Date directory found: {mat_info['date_dir']}")
            print_ok(f"Race files: {mat_info['n_race_files']} file(s)")
            if mat_info['summary_exists']:
                print_ok("Summary file exists")
            else:
                print_warn("Summary file not found (optional)")
        else:
            if not mat_info['exists']:
                print_error(f"Date directory not found: {mat_info['date_dir']}")
            else:
                print_error("No race files found in date directory")
            all_ok = False
            guidance_needed.append("materials")

    # =========================================================================
    # 4. Guidance (if needed)
    # =========================================================================
    if guidance_needed:
        print_header("4. 必要なアクション")

        if "db" in guidance_needed:
            print()
            print_warn("データベースがありません")
            print_info("以下のいずれかを実行してください:")
            print()
            print("  (a) 既存DBをコピー:")
            print_cmd(f"cp /path/to/your/netkeiba.db {db_path}")
            print()
            print("  (b) スクレイピングで新規作成:")
            print_cmd("python -m src.ingestion.ingest_runner --start-year 2024 --end-year 2024")

        if "models" in guidance_needed:
            print()
            print_warn("学習済みモデルがありません")
            print_info("以下を実行してモデルを学習してください:")
            print()
            print_cmd(f"python scripts/train_eval_v4.py --db {db_path} --out {models_dir}")
            print()
            print_info("必要なモデル:")
            for target, filenames in REQUIRED_MODELS:
                print(f"    - {target}: {filenames[0]}")

        if "materials" in guidance_needed:
            print()
            print_warn(f"日付 {date} の pre_race 素材がありません")
            print_info("以下を実行して素材を生成してください:")
            print()
            print_cmd(
                f"python scripts/generate_pre_race_materials.py "
                f"--db {db_path} --date {date} --out artifacts/pre_race/{date}"
            )

        if "demo" in guidance_needed:
            print()
            print_warn("デモデータがありません")
            print_info("以下のいずれかを実行してください:")
            print()
            print("  (a) デモデータを作成:")
            print_cmd("python scripts/run_pre_race_day.py --date YYYY-MM-DD  # 実データから生成")
            print()
            print("  (b) 手動でデモディレクトリを作成:")
            print_cmd("mkdir -p artifacts/pre_race/demo")
            print_cmd("cp artifacts/pre_race/YYYY-MM-DD/*.json artifacts/pre_race/demo/")

        return 1

    # =========================================================================
    # 5. Success - Show UI Startup
    # =========================================================================
    print_header("4. 準備完了 - UI起動")

    print_ok("すべてのチェックに合格しました")
    print()
    print_info("以下のコマンドでUIサーバーを起動してください:")
    print()
    print_cmd("python ui/pre_race/server.py")
    print()
    print_info(f"ブラウザで開く: http://{UI_HOST}:{UI_PORT}")
    print()

    if demo_mode:
        print_info("デモモード: 日付選択で 'demo' を選んでください")
    elif date:
        print_info(f"日付 {date} のレースが選択可能です")

    print()
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.GREEN}Ready to run!{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")

    return 0


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pre-race day operation runner - checks materials/models and shows UI startup guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check for a specific date
  python scripts/run_pre_race_day.py --date 2024-12-29

  # Check with custom DB path
  python scripts/run_pre_race_day.py --db /path/to/netkeiba.db --date 2024-12-29

  # Check demo mode
  python scripts/run_pre_race_day.py --demo

  # Check with custom models directory
  python scripts/run_pre_race_day.py --date 2024-12-29 --models models/v4
"""
    )

    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH.name})",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Target date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help=f"Models directory (default: {DEFAULT_MODELS_DIR.name})",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Check demo data instead of specific date",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.demo and not args.date:
        parser.error("Either --date or --demo is required")

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Convert paths
    db_path = Path(args.db).absolute()
    models_dir = Path(args.models).absolute()

    # Run check
    exit_code = run_check(
        date=args.date,
        db_path=db_path,
        models_dir=models_dir,
        demo_mode=args.demo,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
