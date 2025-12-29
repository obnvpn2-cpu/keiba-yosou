#!/usr/bin/env python3
"""
Pre-race Scenario UI Server

シナリオUIを提供するローカルサーバー。
race_<race_id>.json を読み込み、シナリオ補正を適用し、結果を保存する。

起動方法:
    python ui/pre_race/server.py

Usage:
    1. サーバー起動後、http://localhost:8080 にアクセス
    2. レースJSONファイルを選択（artifacts/pre_race/<date>/race_<race_id>.json）
    3. シナリオを入力し「適用」をクリック
    4. 補正後のランキングとΔを確認
    5. 「保存」で結果をJSONに出力
"""

import json
import logging
import os
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse
import socketserver
import threading

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Scenario Adjustment Logic (Simplified for UI)
# =============================================================================

# Pace adjustment coefficients
PACE_COEF = {
    "S": {  # Slow pace
        "逃げ": {"win": 1.20, "in3": 1.10},
        "先行": {"win": 1.10, "in3": 1.05},
        "差し": {"win": 0.95, "in3": 0.98},
        "追込": {"win": 0.85, "in3": 0.92},
        "不明": {"win": 1.00, "in3": 1.00},
    },
    "M": {  # Medium pace
        "逃げ": {"win": 1.00, "in3": 1.00},
        "先行": {"win": 1.00, "in3": 1.00},
        "差し": {"win": 1.00, "in3": 1.00},
        "追込": {"win": 1.00, "in3": 1.00},
        "不明": {"win": 1.00, "in3": 1.00},
    },
    "H": {  # High pace
        "逃げ": {"win": 0.80, "in3": 0.90},
        "先行": {"win": 0.90, "in3": 0.95},
        "差し": {"win": 1.10, "in3": 1.05},
        "追込": {"win": 1.20, "in3": 1.12},
        "不明": {"win": 1.00, "in3": 1.00},
    },
}

# Track condition adjustment
TRACK_COEF = {
    "良": 1.00,
    "稍重": 1.00,
    "重": 0.98,
    "不良": 0.95,
}

# Bias adjustment (based on umaban position)
BIAS_COEF = {
    "内": {  # Inner bias
        "inner": {"win": 1.05, "in3": 1.03},
        "middle": {"win": 1.00, "in3": 1.00},
        "outer": {"win": 0.97, "in3": 0.98},
    },
    "外": {  # Outer bias
        "inner": {"win": 0.97, "in3": 0.98},
        "middle": {"win": 1.00, "in3": 1.00},
        "outer": {"win": 1.05, "in3": 1.03},
    },
    "フラット": {  # No bias
        "inner": {"win": 1.00, "in3": 1.00},
        "middle": {"win": 1.00, "in3": 1.00},
        "outer": {"win": 1.00, "in3": 1.00},
    },
}


def estimate_run_style(umaban: int, front_runner_ids: List[str], horse_id: str) -> str:
    """
    馬番と指定から脚質を推定（仮実装）
    """
    if horse_id in front_runner_ids:
        return "逃げ"
    # 仮: 馬番が若いほど前寄りと推定
    if umaban <= 3:
        return "先行"
    elif umaban <= 10:
        return "差し"
    else:
        return "追込"


def estimate_lane(umaban: int, field_size: int) -> str:
    """
    馬番からレーンを推定
    """
    if field_size <= 8:
        inner_max = 2
        outer_min = 7
    else:
        inner_max = max(2, int(field_size * 0.3))
        outer_min = field_size - max(2, int(field_size * 0.3)) + 1

    if umaban <= inner_max:
        return "inner"
    elif umaban >= outer_min:
        return "outer"
    else:
        return "middle"


def apply_scenario_adjustment(
    race_data: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    """
    シナリオ補正を適用

    Args:
        race_data: race_<race_id>.json の内容
        scenario: シナリオ入力
            - pace: "S" / "M" / "H"
            - track_condition: "良" / "稍重" / "重" / "不良"
            - bias: "内" / "外" / "フラット"
            - front_runner_ids: ["h001", ...]
            - notes: "自由記述"

    Returns:
        補正結果
    """
    entries = race_data.get("entries", [])
    field_size = len(entries)

    pace = scenario.get("pace", "M")
    track_condition = scenario.get("track_condition", "良")
    bias = scenario.get("bias", "フラット")
    front_runner_ids = scenario.get("front_runner_ids", [])

    # 補正結果を格納
    adjusted_entries = []

    for entry in entries:
        horse_id = entry.get("horse_id", "")
        umaban = entry.get("umaban", 0) or 0
        name = entry.get("name", "")

        base_p_win = entry.get("p_win", 0.0) or 0.0
        base_p_in3 = entry.get("p_in3", 0.0) or 0.0
        base_rank_win = entry.get("rank_win", 99)
        base_rank_in3 = entry.get("rank_in3", 99)

        # 脚質推定
        run_style = estimate_run_style(umaban, front_runner_ids, horse_id)

        # レーン推定
        lane = estimate_lane(umaban, field_size)

        # 補正係数を計算
        pace_coef = PACE_COEF.get(pace, PACE_COEF["M"]).get(run_style, {"win": 1.0, "in3": 1.0})
        track_coef = TRACK_COEF.get(track_condition, 1.0)
        bias_coef = BIAS_COEF.get(bias, BIAS_COEF["フラット"]).get(lane, {"win": 1.0, "in3": 1.0})

        # 補正後確率
        adj_p_win = base_p_win * pace_coef["win"] * track_coef * bias_coef["win"]
        adj_p_in3 = base_p_in3 * pace_coef["in3"] * track_coef * bias_coef["in3"]

        # 補正理由を生成
        reasons = []
        if pace != "M":
            if pace_coef["win"] > 1.0:
                reasons.append(f"ペース{pace}で{run_style}有利")
            elif pace_coef["win"] < 1.0:
                reasons.append(f"ペース{pace}で{run_style}不利")

        if bias != "フラット":
            if bias_coef["win"] > 1.0:
                reasons.append(f"{bias}伸びバイアス有利")
            elif bias_coef["win"] < 1.0:
                reasons.append(f"{bias}伸びバイアス不利")

        if horse_id in front_runner_ids:
            reasons.append("逃げ想定馬")

        if track_coef < 1.0:
            reasons.append(f"馬場{track_condition}で減衰")

        adjusted_entries.append({
            "horse_id": horse_id,
            "umaban": umaban,
            "name": name,
            "base_p_win": round(base_p_win, 4),
            "base_p_in3": round(base_p_in3, 4),
            "base_rank_win": base_rank_win,
            "base_rank_in3": base_rank_in3,
            "adj_p_win": round(adj_p_win, 6),
            "adj_p_in3": round(adj_p_in3, 6),
            "run_style": run_style,
            "lane": lane,
            "reasons": reasons,
            "jockey": entry.get("jockey"),
            "trainer": entry.get("trainer"),
        })

    # 正規化
    total_win = sum(e["adj_p_win"] for e in adjusted_entries)
    total_in3 = sum(e["adj_p_in3"] for e in adjusted_entries)

    if total_win > 0:
        for e in adjusted_entries:
            e["adj_p_win"] = round(e["adj_p_win"] / total_win, 4)

    if total_in3 > 3.0:
        scale = 3.0 / total_in3
        for e in adjusted_entries:
            e["adj_p_in3"] = round(e["adj_p_in3"] * scale, 4)

    # 補正後ランキングを計算
    sorted_by_win = sorted(adjusted_entries, key=lambda x: x["adj_p_win"], reverse=True)
    for rank, entry in enumerate(sorted_by_win, 1):
        entry["adj_rank_win"] = rank
        entry["delta_rank_win"] = entry["base_rank_win"] - rank

    sorted_by_in3 = sorted(adjusted_entries, key=lambda x: x["adj_p_in3"], reverse=True)
    for rank, entry in enumerate(sorted_by_in3, 1):
        entry["adj_rank_in3"] = rank
        entry["delta_rank_in3"] = entry["base_rank_in3"] - rank

    # umaban順に並べ直す
    adjusted_entries.sort(key=lambda x: x["umaban"])

    return {
        "race_id": race_data.get("race_id"),
        "race_name": race_data.get("name"),
        "date": race_data.get("date"),
        "place": race_data.get("place"),
        "race_no": race_data.get("race_no"),
        "distance": race_data.get("distance"),
        "course": race_data.get("course"),
        "scenario": scenario,
        "entries": adjusted_entries,
        "adjusted_at": datetime.now().isoformat(),
    }


# =============================================================================
# HTTP Server
# =============================================================================

class ScenarioUIHandler(SimpleHTTPRequestHandler):
    """シナリオUIのHTTPハンドラー"""

    def __init__(self, *args, **kwargs):
        # UI directory as the base
        self.ui_dir = Path(__file__).parent
        super().__init__(*args, directory=str(self.ui_dir), **kwargs)

    def do_GET(self):
        """GET リクエスト処理"""
        parsed = urlparse(self.path)

        if parsed.path == "/api/list-races":
            self._handle_list_races(parsed)
        elif parsed.path == "/api/load-race":
            self._handle_load_race(parsed)
        else:
            # 静的ファイル配信
            super().do_GET()

    def do_POST(self):
        """POST リクエスト処理"""
        parsed = urlparse(self.path)

        if parsed.path == "/api/apply-scenario":
            self._handle_apply_scenario()
        elif parsed.path == "/api/save-result":
            self._handle_save_result()
        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data: Any, status: int = 200):
        """JSON レスポンスを送信"""
        response = json.dumps(data, ensure_ascii=False, indent=2)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(response.encode("utf-8")))
        self.end_headers()
        self.wfile.write(response.encode("utf-8"))

    def _handle_list_races(self, parsed):
        """レースファイル一覧を返す"""
        query = parse_qs(parsed.query)

        # artifacts/pre_race/ 以下のディレクトリを探索
        artifacts_dir = PROJECT_ROOT / "artifacts" / "pre_race"

        races = []
        if artifacts_dir.exists():
            for date_dir in sorted(artifacts_dir.iterdir(), reverse=True):
                if date_dir.is_dir():
                    for race_file in sorted(date_dir.glob("race_*.json")):
                        races.append({
                            "path": str(race_file.relative_to(PROJECT_ROOT)),
                            "date": date_dir.name,
                            "race_id": race_file.stem.replace("race_", ""),
                        })

        self._send_json({"races": races[:50]})  # 最新50件

    def _handle_load_race(self, parsed):
        """レースデータを読み込む"""
        query = parse_qs(parsed.query)
        race_path = query.get("path", [""])[0]

        if not race_path:
            self._send_json({"error": "path parameter required"}, 400)
            return

        full_path = PROJECT_ROOT / race_path
        if not full_path.exists():
            self._send_json({"error": f"File not found: {race_path}"}, 404)
            return

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._send_json(data)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_apply_scenario(self):
        """シナリオ補正を適用"""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body.decode("utf-8"))
            race_data = request.get("race_data", {})
            scenario = request.get("scenario", {})

            result = apply_scenario_adjustment(race_data, scenario)
            self._send_json(result)
        except Exception as e:
            logger.exception("Error applying scenario")
            self._send_json({"error": str(e)}, 500)

    def _handle_save_result(self):
        """結果を保存"""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body.decode("utf-8"))
            result = request.get("result", {})

            # 保存先ディレクトリ
            output_dir = PROJECT_ROOT / "artifacts" / "ui_runs"
            output_dir.mkdir(parents=True, exist_ok=True)

            # ファイル名: timestamp_race_id.json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            race_id = result.get("race_id", "unknown")
            filename = f"{timestamp}_{race_id}.json"

            output_path = output_dir / filename
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            self._send_json({
                "status": "saved",
                "path": str(output_path.relative_to(PROJECT_ROOT)),
            })
        except Exception as e:
            logger.exception("Error saving result")
            self._send_json({"error": str(e)}, 500)

    def log_message(self, format, *args):
        """ログ出力をカスタマイズ"""
        logger.info("%s - %s", self.address_string(), format % args)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """スレッド対応HTTPサーバー"""
    allow_reuse_address = True


def main():
    """メイン関数"""
    port = int(os.environ.get("PORT", 8080))

    server = ThreadedHTTPServer(("", port), ScenarioUIHandler)

    logger.info("=" * 60)
    logger.info("Pre-race Scenario UI Server")
    logger.info("=" * 60)
    logger.info("  URL: http://localhost:%d", port)
    logger.info("  UI:  %s", Path(__file__).parent / "index.html")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
