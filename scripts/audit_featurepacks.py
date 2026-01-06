#!/usr/bin/env python3
"""
audit_featurepacks.py - Feature Audit Runner (Step C/D)

全バージョン横断の特徴量棚卸しを実行するスクリプト。
各バージョン × target × mode を回して、同一フォーマットで成果物を出力する。

【出力成果物】
- artifacts/feature_audit/<version>/<mode>/<target>/
  - used_features.txt: 使用特徴量一覧
  - feature_inventory.json: 特徴量ごとの統計・安全性ラベル
  - importance_gain.csv: gain重要度
  - importance_split.csv: split重要度
  - group_importance.csv: グループ別重要度
  - summary.json: サマリー

- artifacts/feature_audit/index.json: 全runの集約結果

Usage:
    # 基本実行
    python scripts/audit_featurepacks.py --db netkeiba.db

    # 特定バージョンのみ
    python scripts/audit_featurepacks.py --db netkeiba.db --versions v4

    # v3/v2/v1も含める
    python scripts/audit_featurepacks.py --db netkeiba.db --versions v4,v3,v2,v1

    # dry-run（検出と計画表示のみ）
    python scripts/audit_featurepacks.py --db netkeiba.db --dry-run

    # 詳細統計あり（遅い）
    python scripts/audit_featurepacks.py --db netkeiba.db --no-fast

    # モデルがない場合は自動学習
    python scripts/audit_featurepacks.py --db netkeiba.db --auto-train
"""

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_audit import (
    classify_feature_safety,
    classify_features_batch,
    summarize_safety,
    get_available_adapters,
    get_adapter,
    detect_all_versions,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEFAULT_DB_PATH = PROJECT_ROOT / "netkeiba.db"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "feature_audit"

DEFAULT_TARGETS = ["target_win", "target_in3"]
DEFAULT_MODES = ["pre_race"]


# =============================================================================
# Color Output
# =============================================================================

class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        cls.GREEN = cls.YELLOW = cls.RED = cls.CYAN = cls.BOLD = cls.RESET = ""


def print_header(text: str):
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")


def print_ok(text: str):
    print(f"  {Colors.GREEN}✓{Colors.RESET} {text}")


def print_skip(text: str):
    print(f"  {Colors.YELLOW}○{Colors.RESET} {text}")


def print_fail(text: str):
    print(f"  {Colors.RED}✗{Colors.RESET} {text}")


def print_info(text: str):
    print(f"  {Colors.CYAN}→{Colors.RESET} {text}")


# =============================================================================
# Auto-Training
# =============================================================================

LEGACY_VERSIONS = {"v3", "v2", "v1"}


def check_model_exists(models_dir: Path, version: str, target: str) -> bool:
    """Check if model file exists for given version and target."""
    model_patterns = [
        models_dir / f"lgbm_{target}_{version}.txt",
        models_dir / f"feature_columns_{target}_{version}.json",
    ]
    return all(p.exists() for p in model_patterns)


def auto_train_legacy(
    db_path: Path,
    models_dir: Path,
    version: str,
    targets: List[str],
) -> Tuple[bool, str]:
    """
    Run train_eval_legacy.py for missing models.

    Returns:
        (success, error_message) tuple
    """
    script_path = Path(__file__).parent / "train_eval_legacy.py"

    if not script_path.exists():
        msg = f"Training script not found: {script_path}"
        logger.error(msg)
        return False, msg

    print_info(f"Training models for {version}...")

    cmd = [
        sys.executable,
        str(script_path),
        "--db", str(db_path),
        "--version", version,
        "--out", str(models_dir),
        "--targets", *targets,
    ]

    # Windows環境でのencoding問題を回避
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # デコード失敗時は置換
            cwd=str(Path(__file__).parent.parent),
            env=env,
        )

        if result.returncode != 0:
            stderr = result.stderr or "(no stderr)"
            logger.error(f"Training failed for {version}")
            logger.error(f"STDERR: {stderr}")
            print_fail(f"Training failed for {version}")
            return False, stderr

        print_ok(f"Training completed for {version}")
        return True, ""

    except Exception as e:
        msg = str(e)
        logger.exception(f"Error running training for {version}")
        print_fail(f"Training error: {e}")
        return False, msg


# =============================================================================
# Output Writers
# =============================================================================

def write_used_features(output_dir: Path, features: List[str]) -> None:
    """used_features.txt を出力"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "used_features.txt"
    with open(path, "w", encoding="utf-8") as f:
        for feat in features:
            f.write(feat + "\n")


def write_feature_inventory(
    output_dir: Path,
    features: List[str],
    feature_stats: Dict[str, Dict[str, Any]],
) -> None:
    """feature_inventory.json を出力"""
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = []
    for feat in features:
        label, notes = classify_feature_safety(feat)
        stats = feature_stats.get(feat, {})

        entry = {
            "feature": feat,
            "safety_label": label,
            "notes": notes,
            "dtype": stats.get("dtype", "unknown"),
            "missing_rate": stats.get("missing_rate"),
            "n_unique": stats.get("n_unique"),
            "example_values": stats.get("example_values", []),
        }
        inventory.append(entry)

    path = output_dir / "feature_inventory.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)


def write_importance_csv(
    output_dir: Path,
    importance: Dict[str, float],
    filename: str,
) -> None:
    """importance CSV を出力"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ソートして出力
    sorted_items = sorted(importance.items(), key=lambda x: -x[1])

    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write("feature,importance\n")
        for feat, imp in sorted_items:
            f.write(f"{feat},{imp}\n")


def write_group_importance(
    output_dir: Path,
    gain_dict: Dict[str, float],
    split_dict: Dict[str, int],
) -> None:
    """group_importance.csv を出力"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # プレフィックスでグループ化
    groups: Dict[str, Dict[str, Any]] = {}

    for feat in gain_dict:
        # プレフィックス抽出（最初の _ まで）
        if "_" in feat:
            prefix = feat.split("_")[0] + "_"
        else:
            prefix = "other"

        if prefix not in groups:
            groups[prefix] = {
                "gain_sum": 0.0,
                "split_sum": 0,
                "n_features": 0,
            }

        groups[prefix]["gain_sum"] += gain_dict.get(feat, 0)
        groups[prefix]["split_sum"] += split_dict.get(feat, 0)
        groups[prefix]["n_features"] += 1

    # ソートして出力
    sorted_groups = sorted(groups.items(), key=lambda x: -x[1]["gain_sum"])

    path = output_dir / "group_importance.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("group,importance_gain_sum,importance_split_sum,n_features\n")
        for group, data in sorted_groups:
            f.write(f"{group},{data['gain_sum']:.4f},{data['split_sum']},{data['n_features']}\n")


def write_summary(
    output_dir: Path,
    version: str,
    mode: str,
    target: str,
    features: List[str],
    gain_dict: Dict[str, float],
    warnings: List[str],
) -> None:
    """summary.json を出力"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 安全性サマリー
    safety = summarize_safety(features)

    # top features (gain上位20)
    sorted_gain = sorted(gain_dict.items(), key=lambda x: -x[1])
    top_features = [f for f, _ in sorted_gain[:20]]

    # missing top（feature_statsがあれば）- 今回はスキップ

    summary = {
        "version": version,
        "mode": mode,
        "target": target,
        "n_features": len(features),
        "top_features": top_features,
        "unsafe_features_found": safety["unsafe_features"],
        "warn_features_found": safety["warn_features"],
        "n_unsafe": safety["n_unsafe"],
        "n_warn": safety["n_warn"],
        "n_safe": safety["n_safe"],
        "warnings": warnings,
        "generated_at": datetime.now().isoformat(),
    }

    path = output_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def write_index(
    output_dir: Path,
    results: List[Dict[str, Any]],
) -> None:
    """index.json を出力"""
    output_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "generated_at": datetime.now().isoformat(),
        "n_runs": len(results),
        "runs": results,
    }

    path = output_dir / "index.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


# =============================================================================
# Main Logic
# =============================================================================

def run_audit(
    db_path: Path,
    models_dir: Path,
    output_dir: Path,
    versions: Optional[List[str]] = None,
    targets: Optional[List[str]] = None,
    modes: Optional[List[str]] = None,
    fast: bool = True,
    dry_run: bool = False,
    auto_train: bool = False,
) -> int:
    """
    棚卸しを実行

    Returns:
        終了コード (0=成功, 1=一部失敗, 2=全失敗)
    """
    targets = targets or DEFAULT_TARGETS
    modes = modes or DEFAULT_MODES

    print_header("Feature Audit Runner (Step C/D)")
    print(f"  Database:   {db_path}")
    print(f"  Models:     {models_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Targets:    {', '.join(targets)}")
    print(f"  Modes:      {', '.join(modes)}")
    print(f"  Fast mode:  {fast}")
    print(f"  Dry run:    {dry_run}")
    print(f"  Auto-train: {auto_train}")

    # ==========================================================================
    # 1. バージョン検出
    # ==========================================================================
    print_header("1. Version Detection")

    all_versions = detect_all_versions(db_path, models_dir)
    available_versions = [(v, r) for v, ok, r in all_versions if ok]
    skipped_versions = [(v, r) for v, ok, r in all_versions if not ok]

    for version, reason in available_versions:
        print_ok(f"{version}: {reason}")

    for version, reason in skipped_versions:
        print_skip(f"{version}: {reason}")

    # フィルタリング
    if versions:
        available_versions = [(v, r) for v, r in available_versions if v in versions]

    if not available_versions:
        print_fail("No available versions to audit")
        return 2

    # ==========================================================================
    # 1.5. Auto-Training (if requested)
    # ==========================================================================
    train_errors: Dict[str, str] = {}  # version -> error message

    if auto_train and not dry_run:
        print_header("1.5. Auto-Training Check")

        for version, _ in available_versions:
            if version in LEGACY_VERSIONS:
                missing_targets = []
                for target in targets:
                    if not check_model_exists(models_dir, version, target):
                        missing_targets.append(target)

                if missing_targets:
                    print_info(f"{version}: Missing models for {missing_targets}")
                    success, error_msg = auto_train_legacy(db_path, models_dir, version, missing_targets)
                    if not success:
                        train_errors[version] = error_msg
                else:
                    print_ok(f"{version}: All models exist")

    # ==========================================================================
    # 2. 実行計画
    # ==========================================================================
    print_header("2. Execution Plan")

    plan = []
    for version, _ in available_versions:
        for mode in modes:
            for target in targets:
                plan.append((version, mode, target))
                print_info(f"{version} / {mode} / {target}")

    print(f"\n  Total runs: {len(plan)}")

    if dry_run:
        print_header("Dry Run Complete")
        print("  No files written (--dry-run specified)")
        return 0

    # ==========================================================================
    # 3. 実行
    # ==========================================================================
    print_header("3. Running Audits")

    index_results = []

    # DB接続
    conn = sqlite3.connect(str(db_path))

    try:
        for version, mode, target in plan:
            run_key = f"{version}/{mode}/{target}"
            print(f"\n  {Colors.CYAN}Running: {run_key}{Colors.RESET}")

            adapter = get_adapter(version, db_path, models_dir)
            if adapter is None:
                print_fail(f"  Adapter not found: {version}")
                index_results.append({
                    "version": version,
                    "mode": mode,
                    "target": target,
                    "status": "FAIL",
                    "reason": "Adapter not found",
                    "output_path": None,
                    "timestamp": datetime.now().isoformat(),
                })
                continue

            # 出力ディレクトリ
            run_output_dir = output_dir / version / mode / target

            try:
                result = adapter.run_audit(conn, mode=mode, target=target, fast=fast)

                if not result.success:
                    print_fail(f"  Failed: {result.error_message}")
                    index_results.append({
                        "version": version,
                        "mode": mode,
                        "target": target,
                        "status": "FAIL",
                        "reason": result.error_message,
                        "output_path": None,
                        "timestamp": datetime.now().isoformat(),
                    })
                    continue

                # 成果物出力
                write_used_features(run_output_dir, result.used_features)
                write_feature_inventory(run_output_dir, result.used_features, result.feature_stats)

                if result.importance_gain:
                    write_importance_csv(run_output_dir, result.importance_gain, "importance_gain.csv")
                if result.importance_split:
                    write_importance_csv(
                        run_output_dir,
                        {k: float(v) for k, v in result.importance_split.items()},
                        "importance_split.csv",
                    )
                if result.importance_gain and result.importance_split:
                    write_group_importance(run_output_dir, result.importance_gain, result.importance_split)

                write_summary(
                    run_output_dir,
                    version, mode, target,
                    result.used_features,
                    result.importance_gain,
                    result.warnings,
                )

                print_ok(f"  Done: {len(result.used_features)} features")

                index_results.append({
                    "version": version,
                    "mode": mode,
                    "target": target,
                    "status": "OK",
                    "reason": "",
                    "n_features": len(result.used_features),
                    "output_path": str(run_output_dir.relative_to(PROJECT_ROOT).as_posix()),
                    "timestamp": datetime.now().isoformat(),
                })

            except Exception as e:
                logger.exception(f"Error in {run_key}")
                print_fail(f"  Error: {e}")
                index_results.append({
                    "version": version,
                    "mode": mode,
                    "target": target,
                    "status": "FAIL",
                    "reason": str(e),
                    "output_path": None,
                    "timestamp": datetime.now().isoformat(),
                })

    finally:
        conn.close()

    # ==========================================================================
    # 4. Index出力
    # ==========================================================================
    print_header("4. Writing Index")

    # SKIPされたバージョンも記録
    for version, reason in skipped_versions:
        for mode in modes:
            for target in targets:
                index_results.append({
                    "version": version,
                    "mode": mode,
                    "target": target,
                    "status": "SKIP",
                    "reason": reason,
                    "output_path": None,
                    "timestamp": datetime.now().isoformat(),
                })

    write_index(output_dir, index_results)
    print_ok(f"Index written: {output_dir / 'index.json'}")

    # ==========================================================================
    # 5. サマリー
    # ==========================================================================
    print_header("5. Summary")

    n_ok = sum(1 for r in index_results if r["status"] == "OK")
    n_skip = sum(1 for r in index_results if r["status"] == "SKIP")
    n_fail = sum(1 for r in index_results if r["status"] == "FAIL")

    print(f"  {Colors.GREEN}OK:   {n_ok}{Colors.RESET}")
    print(f"  {Colors.YELLOW}SKIP: {n_skip}{Colors.RESET}")
    print(f"  {Colors.RED}FAIL: {n_fail}{Colors.RESET}")

    if n_ok > 0:
        print(f"\n{Colors.GREEN}Audit completed successfully!{Colors.RESET}")
        return 0 if n_fail == 0 else 1
    else:
        print(f"\n{Colors.RED}All audits failed or skipped.{Colors.RESET}")
        return 2


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feature Audit Runner - Audit all feature pack versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run (fast mode)
  python scripts/audit_featurepacks.py --db netkeiba.db

  # Specific versions only
  python scripts/audit_featurepacks.py --db netkeiba.db --versions v4,legacy

  # Dry run (detection only)
  python scripts/audit_featurepacks.py --db netkeiba.db --dry-run

  # With detailed statistics (slower)
  python scripts/audit_featurepacks.py --db netkeiba.db --no-fast

  # Custom output directory
  python scripts/audit_featurepacks.py --db netkeiba.db --out out/audit
"""
    )

    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH.name})",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help=f"Models directory (default: {DEFAULT_MODELS_DIR.name})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT).as_posix()})",
    )
    parser.add_argument(
        "--versions",
        type=str,
        default=None,
        help="Comma-separated list of versions to audit (default: all detected)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(DEFAULT_TARGETS),
        help=f"Comma-separated list of targets (default: {','.join(DEFAULT_TARGETS)})",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help=f"Comma-separated list of modes (default: {','.join(DEFAULT_MODES)})",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Fast mode (skip heavy computations, default: True)",
    )
    parser.add_argument(
        "--no-fast",
        action="store_true",
        help="Disable fast mode (compute detailed statistics)",
    )
    parser.add_argument(
        "--with-permutation",
        action="store_true",
        help="Include permutation importance (not yet implemented)",
    )
    parser.add_argument(
        "--with-segments",
        action="store_true",
        help="Include segment analysis (not yet implemented)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detection and planning only, no file output",
    )
    parser.add_argument(
        "--auto-train",
        action="store_true",
        help="Auto-train missing models for legacy versions (v3/v2/v1)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Parse lists
    versions = args.versions.split(",") if args.versions else None
    targets = args.targets.split(",") if args.targets else DEFAULT_TARGETS
    modes = args.modes.split(",") if args.modes else DEFAULT_MODES

    # Fast mode
    fast = args.fast and not args.no_fast

    # Run
    exit_code = run_audit(
        db_path=Path(args.db).absolute(),
        models_dir=Path(args.models).absolute(),
        output_dir=Path(args.out).absolute(),
        versions=versions,
        targets=targets,
        modes=modes,
        fast=fast,
        dry_run=args.dry_run,
        auto_train=args.auto_train,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
