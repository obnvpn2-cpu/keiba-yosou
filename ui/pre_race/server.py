#!/usr/bin/env python3
"""
Pre-race Scenario UI Server (v2.0)

シナリオUIを提供するローカルサーバー。
race_<race_id>.json を読み込み、シナリオ補正を適用し、結果を保存する。

v2.0 更新:
- トラックバイアスを2軸に分離（進路バイアス + 脚質バイアス）
- メモ解析API追加（提案チップ生成）
- レース一覧にメタ情報付与
- 全体コメント生成
- 馬タグ生成

起動方法:
    python ui/pre_race/server.py

Usage:
    1. サーバー起動後、http://localhost:8080 にアクセス
    2. レースJSONファイルを選択（artifacts/pre_race/<date>/race_<race_id>.json）
    3. シナリオを入力し「適用」をクリック
    4. 補正後のランキングと変化量を確認
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_pre_race.memo_parser import MemoParser, parse_memo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Scenario Adjustment Logic (v2.0 - Two-axis Bias)
# =============================================================================

# Pace adjustment coefficients (脚質への影響)
PACE_COEF = {
    "S": {  # Slow pace - 前が有利
        "逃げ": {"win": 1.20, "in3": 1.12},
        "先行": {"win": 1.12, "in3": 1.08},
        "差し": {"win": 0.92, "in3": 0.95},
        "追込": {"win": 0.82, "in3": 0.88},
        "不明": {"win": 1.00, "in3": 1.00},
    },
    "M": {  # Medium pace - フラット
        "逃げ": {"win": 1.00, "in3": 1.00},
        "先行": {"win": 1.00, "in3": 1.00},
        "差し": {"win": 1.00, "in3": 1.00},
        "追込": {"win": 1.00, "in3": 1.00},
        "不明": {"win": 1.00, "in3": 1.00},
    },
    "H": {  # High pace - 差しが有利
        "逃げ": {"win": 0.80, "in3": 0.88},
        "先行": {"win": 0.90, "in3": 0.94},
        "差し": {"win": 1.12, "in3": 1.08},
        "追込": {"win": 1.22, "in3": 1.14},
        "不明": {"win": 1.00, "in3": 1.00},
    },
}

# Track condition adjustment
TRACK_COEF = {
    "良": 1.00,
    "稍重": 0.99,
    "重": 0.97,
    "不良": 0.94,
}

# 進路バイアス係数 (lane_bias)
# 「どのラインが伸びるか」に基づく補正
# 重要: 枠へ直結しない。推定される通り道（lane）への作用
LANE_BIAS_COEF = {
    "inner": {  # 内が伸びる馬場
        # 推定される通り道に対する補正
        "inner_lane": {"win": 1.06, "in3": 1.04},   # 内を通る馬
        "middle_lane": {"win": 1.00, "in3": 1.00}, # 中を通る馬
        "outer_lane": {"win": 0.96, "in3": 0.97},  # 外を回す馬
    },
    "middle": {  # 真ん中が良い
        "inner_lane": {"win": 1.00, "in3": 1.00},
        "middle_lane": {"win": 1.03, "in3": 1.02},
        "outer_lane": {"win": 1.00, "in3": 1.00},
    },
    "outer": {  # 外が伸びる馬場
        "inner_lane": {"win": 0.96, "in3": 0.97},
        "middle_lane": {"win": 1.00, "in3": 1.00},
        "outer_lane": {"win": 1.06, "in3": 1.04},
    },
    "flat": {  # バイアスなし
        "inner_lane": {"win": 1.00, "in3": 1.00},
        "middle_lane": {"win": 1.00, "in3": 1.00},
        "outer_lane": {"win": 1.00, "in3": 1.00},
    },
}

# 脚質バイアス係数 (style_bias)
# 「前が残るか/差しが届くか」に基づく補正
STYLE_BIAS_COEF = {
    "front": {  # 前が残りやすい
        "逃げ": {"win": 1.10, "in3": 1.06},
        "先行": {"win": 1.06, "in3": 1.04},
        "差し": {"win": 0.95, "in3": 0.97},
        "追込": {"win": 0.90, "in3": 0.94},
        "不明": {"win": 1.00, "in3": 1.00},
    },
    "closer": {  # 差しが届きやすい
        "逃げ": {"win": 0.90, "in3": 0.94},
        "先行": {"win": 0.95, "in3": 0.97},
        "差し": {"win": 1.06, "in3": 1.04},
        "追込": {"win": 1.10, "in3": 1.06},
        "不明": {"win": 1.00, "in3": 1.00},
    },
    "flat": {  # バイアスなし
        "逃げ": {"win": 1.00, "in3": 1.00},
        "先行": {"win": 1.00, "in3": 1.00},
        "差し": {"win": 1.00, "in3": 1.00},
        "追込": {"win": 1.00, "in3": 1.00},
        "不明": {"win": 1.00, "in3": 1.00},
    },
}


def estimate_run_style(
    umaban: int,
    front_runner_ids: List[str],
    horse_id: str,
    entry: Optional[Dict[str, Any]] = None
) -> str:
    """
    脚質を推定

    優先順位:
    1. JSONに脚質情報があればそれを使用
    2. front_runner_ids に含まれていれば逃げ
    3. 馬番から弱い推定（不確実）
    """
    # 1. JSONに脚質情報があれば使用
    if entry:
        run_style = entry.get("run_style") or entry.get("style")
        if run_style and run_style in ["逃げ", "先行", "差し", "追込"]:
            return run_style

    # 2. 指定された逃げ馬
    if horse_id in front_runner_ids:
        return "逃げ"

    # 3. 馬番から弱い推定（あくまで補助）
    # 内枠は先行寄り、外枠は差し寄りの傾向はあるが確実ではない
    if umaban <= 4:
        return "先行"
    elif umaban <= 10:
        return "差し"
    else:
        return "追込"


def estimate_lane(
    umaban: int,
    field_size: int,
    run_style: str,
    entry: Optional[Dict[str, Any]] = None
) -> str:
    """
    推定される通り道（lane）を推定

    重要: 枠番 ≠ 通り道
    - 内枠の差し馬は外へ持ち出すことがある
    - 外枠の逃げ馬は内に切れ込む
    - これらを考慮した「推定」を行う

    Returns:
        "inner_lane", "middle_lane", "outer_lane"
    """
    # JSONにlane情報があれば使用
    if entry:
        lane = entry.get("estimated_lane")
        if lane and lane in ["inner_lane", "middle_lane", "outer_lane"]:
            return lane

    # 枠と脚質から推定
    if field_size <= 8:
        inner_max = 2
        outer_min = 7
    else:
        inner_max = max(2, int(field_size * 0.25))
        outer_min = field_size - max(2, int(field_size * 0.25)) + 1

    # 基本は枠から推定
    if umaban <= inner_max:
        base_lane = "inner"
    elif umaban >= outer_min:
        base_lane = "outer"
    else:
        base_lane = "middle"

    # 脚質による調整（弱い推定）
    # 差し・追込は外へ持ち出す傾向がある（ただし確実ではない）
    if run_style in ["差し", "追込"] and base_lane == "inner":
        # 内枠の差し馬は中~外へ持ち出す可能性
        return "middle_lane"
    elif run_style in ["逃げ", "先行"] and base_lane == "outer":
        # 外枠の逃げ馬は内へ切れ込む可能性
        return "middle_lane"

    return f"{base_lane}_lane"


def generate_horse_tags(
    entry: Dict[str, Any],
    run_style: str,
    lane: str,
    is_front_runner: bool
) -> List[str]:
    """馬ごとのタグを生成"""
    tags = []

    # 脚質タグ
    if is_front_runner:
        tags.append("逃げ想定")
    elif run_style:
        tags.append(f"{run_style}想定")

    # 推定進路タグ（弱い推定であることを示す）
    lane_labels = {
        "inner_lane": "内目通過想定",
        "middle_lane": "中通過想定",
        "outer_lane": "外目通過想定",
    }
    if lane in lane_labels:
        tags.append(lane_labels[lane])

    # 特徴量ベースのタグ（JSONにあれば）
    if entry.get("h_days_since_last") is not None:
        days = entry.get("h_days_since_last", 0)
        if days > 90:
            tags.append("休み明け")
        elif days > 60:
            tags.append("間隔あり")

    if entry.get("h_recent3_avg_finish") is not None:
        avg_finish = entry.get("h_recent3_avg_finish", 10)
        if avg_finish <= 2.0:
            tags.append("近走好調")
        elif avg_finish >= 8.0:
            tags.append("近走不振")

    if entry.get("h_recent3_avg_last3f") is not None:
        avg_last3f = entry.get("h_recent3_avg_last3f", 35)
        if avg_last3f <= 34.0:
            tags.append("末脚○")
        elif avg_last3f >= 37.0:
            tags.append("末脚△")

    return tags


def generate_race_comment(
    race_data: Dict[str, Any],
    scenario: Dict[str, Any],
    adjusted_entries: List[Dict[str, Any]]
) -> str:
    """
    全体コメントを生成

    入力情報のみから説明可能な範囲で生成（捏造禁止）
    """
    parts = []

    race_name = race_data.get("name") or race_data.get("race_name") or ""
    distance = race_data.get("distance", 0)
    course = race_data.get("course", "")

    # レース基本情報
    if race_name:
        parts.append(f"【{race_name}】")
    if distance and course:
        parts.append(f"{course}{distance}m。")

    # ペース予想
    pace = scenario.get("pace", "M")
    pace_labels = {"S": "スローペース", "M": "平均ペース", "H": "ハイペース"}
    parts.append(f"予想ペースは{pace_labels.get(pace, '不明')}。")

    # 脚質バイアス
    style_bias = scenario.get("style_bias", "flat")
    if style_bias == "front":
        parts.append("前残り傾向で逃げ・先行馬に有利な展開が見込まれる。")
    elif style_bias == "closer":
        parts.append("差し・追込が届きやすい展開が見込まれる。")

    # 進路バイアス
    lane_bias = scenario.get("lane_bias", "flat")
    if lane_bias == "inner":
        parts.append("内めのラインが伸びる馬場状態。")
    elif lane_bias == "outer":
        parts.append("外めのラインが伸びる馬場状態。")
    elif lane_bias == "middle":
        parts.append("馬場中央が良い状態。")

    # 馬場状態
    track = scenario.get("track_condition", "良")
    if track in ["重", "不良"]:
        parts.append(f"馬場は{track}で、パワーが求められる。")
    elif track == "稍重":
        parts.append("やや時計がかかる馬場。")

    # 逃げ馬想定
    front_runners = [e for e in adjusted_entries if e.get("run_style") == "逃げ"]
    if len(front_runners) == 1:
        parts.append(f"逃げは{front_runners[0].get('name', '?')}が想定される単騎逃げ見込み。")
    elif len(front_runners) >= 2:
        names = "、".join([e.get("name", "?") for e in front_runners[:3]])
        parts.append(f"逃げ候補は{names}など複数おり、競り合いの可能性あり。")

    # メモからの補足
    notes = scenario.get("notes", "")
    if notes:
        parts.append(f"補足：{notes[:100]}")

    return " ".join(parts)


def apply_scenario_adjustment(
    race_data: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    """
    シナリオ補正を適用（v2.0）

    Args:
        race_data: race_<race_id>.json の内容
        scenario: シナリオ入力
            - pace: "S" / "M" / "H"
            - track_condition: "良" / "稍重" / "重" / "不良"
            - lane_bias: "inner" / "middle" / "outer" / "flat"
            - style_bias: "front" / "closer" / "flat"
            - front_runner_ids: ["h001", ...]
            - notes: "自由記述"

    Returns:
        補正結果
    """
    entries = race_data.get("entries", [])
    field_size = len(entries)

    pace = scenario.get("pace", "M")
    track_condition = scenario.get("track_condition", "良")
    lane_bias = scenario.get("lane_bias", "flat")
    style_bias = scenario.get("style_bias", "flat")
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

        is_front_runner = horse_id in front_runner_ids

        # 脚質推定
        run_style = estimate_run_style(umaban, front_runner_ids, horse_id, entry)

        # 進路推定（枠番 + 脚質から）
        lane = estimate_lane(umaban, field_size, run_style, entry)

        # 補正係数を計算
        pace_coef = PACE_COEF.get(pace, PACE_COEF["M"]).get(run_style, {"win": 1.0, "in3": 1.0})
        track_coef = TRACK_COEF.get(track_condition, 1.0)
        lane_coef = LANE_BIAS_COEF.get(lane_bias, LANE_BIAS_COEF["flat"]).get(lane, {"win": 1.0, "in3": 1.0})
        style_coef = STYLE_BIAS_COEF.get(style_bias, STYLE_BIAS_COEF["flat"]).get(run_style, {"win": 1.0, "in3": 1.0})

        # 補正後確率
        adj_p_win = base_p_win * pace_coef["win"] * track_coef * lane_coef["win"] * style_coef["win"]
        adj_p_in3 = base_p_in3 * pace_coef["in3"] * track_coef * lane_coef["in3"] * style_coef["in3"]

        # 補正理由を生成
        reasons = []

        # ペース補正
        if pace != "M":
            if pace_coef["win"] > 1.0:
                reasons.append(f"ペース{pace}で{run_style}有利 (+{(pace_coef['win']-1)*100:.0f}%)")
            elif pace_coef["win"] < 1.0:
                reasons.append(f"ペース{pace}で{run_style}不利 ({(pace_coef['win']-1)*100:.0f}%)")

        # 脚質バイアス補正
        if style_bias != "flat":
            style_label = {"front": "前残り", "closer": "差し向き"}.get(style_bias, style_bias)
            if style_coef["win"] > 1.0:
                reasons.append(f"{style_label}バイアスで有利 (+{(style_coef['win']-1)*100:.0f}%)")
            elif style_coef["win"] < 1.0:
                reasons.append(f"{style_label}バイアスで不利 ({(style_coef['win']-1)*100:.0f}%)")

        # 進路バイアス補正
        if lane_bias != "flat":
            lane_label = {"inner": "内伸び", "middle": "中央", "outer": "外伸び"}.get(lane_bias, lane_bias)
            est_lane_label = {"inner_lane": "内目", "middle_lane": "中", "outer_lane": "外目"}.get(lane, lane)
            if lane_coef["win"] > 1.0:
                reasons.append(f"{lane_label}馬場で{est_lane_label}通過有利 (+{(lane_coef['win']-1)*100:.0f}%)")
            elif lane_coef["win"] < 1.0:
                reasons.append(f"{lane_label}馬場で{est_lane_label}通過不利 ({(lane_coef['win']-1)*100:.0f}%)")

        # 逃げ想定馬
        if is_front_runner:
            reasons.append("逃げ想定馬")

        # 馬場状態
        if track_coef < 1.0:
            reasons.append(f"馬場{track_condition}で減衰 ({(track_coef-1)*100:.0f}%)")

        # タグ生成
        tags = generate_horse_tags(entry, run_style, lane, is_front_runner)

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
            "tags": tags,
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
        entry["rank_change_win"] = entry["base_rank_win"] - rank  # 正=上昇、負=下降

    sorted_by_in3 = sorted(adjusted_entries, key=lambda x: x["adj_p_in3"], reverse=True)
    for rank, entry in enumerate(sorted_by_in3, 1):
        entry["adj_rank_in3"] = rank
        entry["rank_change_in3"] = entry["base_rank_in3"] - rank

    # umaban順に並べ直す
    adjusted_entries.sort(key=lambda x: x["umaban"])

    # 全体コメント生成
    race_comment = generate_race_comment(race_data, scenario, adjusted_entries)

    return {
        "race_id": race_data.get("race_id"),
        "race_name": race_data.get("name") or race_data.get("race_name"),
        "date": race_data.get("date"),
        "place": race_data.get("place"),
        "race_no": race_data.get("race_no"),
        "distance": race_data.get("distance"),
        "course": race_data.get("course"),
        "grade": race_data.get("grade"),
        "scenario": scenario,
        "entries": adjusted_entries,
        "race_comment": race_comment,
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
        elif parsed.path == "/api/parse-memo":
            self._handle_parse_memo()
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
        """レースファイル一覧を返す（メタ情報付き）"""
        query = parse_qs(parsed.query)

        # artifacts/pre_race/ 以下のディレクトリを探索
        artifacts_dir = PROJECT_ROOT / "artifacts" / "pre_race"

        races = []
        if artifacts_dir.exists():
            for date_dir in sorted(artifacts_dir.iterdir(), reverse=True):
                if date_dir.is_dir():
                    for race_file in sorted(date_dir.glob("race_*.json")):
                        # JSONを読み込んでメタ情報を取得
                        try:
                            with open(race_file, "r", encoding="utf-8") as f:
                                data = json.load(f)

                            race_id = data.get("race_id") or race_file.stem.replace("race_", "")
                            race_name = data.get("name") or data.get("race_name") or ""
                            place = data.get("place") or ""
                            race_no = data.get("race_no")
                            distance = data.get("distance")
                            course = data.get("course") or ""
                            grade = data.get("grade") or ""
                            head_count = data.get("head_count") or len(data.get("entries", []))

                            # 表示用ラベルを構築
                            label_parts = [date_dir.name]
                            if place:
                                label_parts.append(place)
                            if race_no:
                                label_parts.append(f"{race_no}R")
                            if race_name:
                                label_parts.append(race_name)
                            if grade:
                                label_parts.append(f"({grade})")
                            if course and distance:
                                label_parts.append(f"{course}{distance}m")
                            if head_count:
                                label_parts.append(f"{head_count}頭")

                            races.append({
                                "path": str(race_file.relative_to(PROJECT_ROOT)),
                                "date": date_dir.name,
                                "race_id": race_id,
                                "label": " ".join(label_parts),
                                "meta": {
                                    "race_name": race_name,
                                    "place": place,
                                    "race_no": race_no,
                                    "distance": distance,
                                    "course": course,
                                    "grade": grade,
                                    "head_count": head_count,
                                }
                            })
                        except Exception as e:
                            logger.warning(f"Failed to read {race_file}: {e}")
                            # フォールバック：最小限の情報
                            races.append({
                                "path": str(race_file.relative_to(PROJECT_ROOT)),
                                "date": date_dir.name,
                                "race_id": race_file.stem.replace("race_", ""),
                                "label": f"{date_dir.name} - {race_file.stem}",
                                "meta": {},
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

    def _handle_parse_memo(self):
        """メモをパースして提案チップを返す"""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body.decode("utf-8"))
            memo_text = request.get("memo", "")

            result = parse_memo(memo_text)
            self._send_json(result)
        except Exception as e:
            logger.exception("Error parsing memo")
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
    logger.info("Pre-race Scenario UI Server v2.0")
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
