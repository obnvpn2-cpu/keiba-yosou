"""
netkeiba ingestion パイプライン - HTMLパーサー

BeautifulSoupを使用してレース結果ページをパースし、
構造化データ（dataclass）に変換する。
"""

import re
import json
import logging
from typing import Optional
from datetime import date

from bs4 import BeautifulSoup, Tag

from .models import (
    Race,
    RaceResult,
    Payout,
    Corner,
    LapTime,
    HorseLap,
    HorseShortComment,
    ParsedRaceData,
    PLACE_CODE_MAP,
)

logger = logging.getLogger(__name__)


def parse_race_page(html: str, race_id: str) -> ParsedRaceData:
    """
    レース結果ページのHTMLをパースする。
    
    Args:
        html: HTMLコンテンツ
        race_id: 12桁のレースID
    
    Returns:
        パース済みデータ
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # 各セクションをパース
    race = _parse_race_info(soup, race_id)
    results = _parse_race_results(soup, race_id)
    payouts = _parse_payouts(soup, race_id)
    corner = _parse_corners(soup, race_id)
    lap_times = _parse_race_lap_times(soup, race_id)
    
    # 馬場指数・コメント
    baba_info = _parse_baba_info(soup)
    if baba_info:
        race.baba_index, race.baba_comment = baba_info
    
    # レース分析コメント
    race.analysis_comment = _parse_analysis_comment(soup)
    
    # 出走頭数を結果から設定
    race.head_count = len(results)
    
    # 短評（マスター会員限定）
    short_comments = _parse_short_comments(soup, race_id)
    
    # horse_laps は AJAX エンドポイントから取得するため、ここでは空リスト
    # ingest_runner.py で別途取得・パース
    horse_laps = []
    
    return ParsedRaceData(
        race=race,
        results=results,
        payouts=payouts,
        corner=corner,
        lap_times=lap_times,
        horse_laps=horse_laps,
        short_comments=short_comments,
    )


def parse_horse_laptime_json(json_str: str, race_id: str) -> list[HorseLap]:
    """
    AJAXエンドポイントから取得したJSONをパースしてHorseLapリストを返す。
    """
    horse_laps: list[HorseLap] = []

    if not json_str:
        logger.debug(f"[{race_id}] Empty JSON string for horse laptime")
        return horse_laps

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"[{race_id}] Failed to parse horse laptime JSON: {e}")
        return horse_laps

    # netkeiba の馬ラップ API は「HTML を JSON の文字列として返す」形式
    # → json.loads(...) の結果が str だったら、そのまま HTML としてパースする
    if isinstance(data, str):
        logger.debug(f"[{race_id}] Horse laptime JSON is HTML string, parsing as HTML")
        return _parse_horse_laps_from_html(data, race_id)

    # ここから下は「本当に JSON で来た場合」の汎用処理（将来フォーマット変更多分用）
    if isinstance(data, dict):
        lap_data = data.get("data") or data.get("result") or data.get("laps")

        html_content = data.get("html") or data.get("result_html")
        if html_content and isinstance(html_content, str):
            return _parse_horse_laps_from_html(html_content, race_id)

        if lap_data is None:
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    lap_data = value
                    break

        if lap_data is None:
            logger.debug(f"[{race_id}] No lap data found in JSON: keys={list(data.keys())}")
            return horse_laps
    elif isinstance(data, list):
        lap_data = data
    else:
        logger.warning(f"[{race_id}] Unexpected JSON structure: {type(data)}")
        return horse_laps

    # （将来 JSON 直パースが必要になったとき用の汎用ロジックはここに残しておく）
    # 現状の netkeiba ではここに来ない想定だが、安全のため残す。
    if isinstance(lap_data, list):
        for item in lap_data:
            if not isinstance(item, dict):
                continue

            horse_id = item.get("horse_id") or item.get("horseId") or item.get("id")
            laps = item.get("laps") or item.get("lap_times") or item.get("times")

            if not horse_id or not isinstance(laps, list):
                continue

            horse_id = str(horse_id)

            for lap_item in laps:
                if not isinstance(lap_item, dict):
                    continue

                section_m = lap_item.get("distance") or lap_item.get("section") or lap_item.get("m")
                time_sec = lap_item.get("time") or lap_item.get("time_sec") or lap_item.get("sec")
                position = lap_item.get("position") or lap_item.get("rank")

                try:
                    section_m_val = int(section_m) if section_m is not None else None
                    time_sec_val = float(time_sec) if time_sec is not None else None
                except (TypeError, ValueError):
                    continue

                if section_m_val is None or time_sec_val is None:
                    continue

                horse_laps.append(
                    HorseLap(
                        race_id=race_id,
                        horse_id=horse_id,
                        section_m=section_m_val,
                        time_sec=time_sec_val,
                        position=int(position)
                        if isinstance(position, (int, float, str)) and str(position).isdigit()
                        else None,
                    )
                )

    logger.debug(f"[{race_id}] Parsed {len(horse_laps)} horse lap records")
    return horse_laps


def _parse_horse_laps_from_html(html: str, race_id: str) -> list[HorseLap]:
    """
    AJAX レスポンスに含まれる HTML から horse_laps をパースする。

    構造イメージ:
      <table id="lap_summary" class="LapSummary_Table">
        <thead>
          <!-- 最下段の Header 行に th[data-furlong] が並ぶ -->
        </thead>
        <tbody>
          <tr class="HorseList">
            <td class="Horse_Info Horse_Link">
              <a href="/horse/2021104514/">ハリウッドブルース</a>
            </td>
            ...
            <td data-laptime="13.2" class="CellDataWrap">13.2</td>
            <td class="PositionCell" data-position="..."></td>
            <td data-laptime="11.5" class="CellDataWrap">11.5</td>
            <td class="PositionCell" data-position="..."></td>
            ...
          </tr>
        </tbody>
      </table>
    """
    horse_laps: list[HorseLap] = []

    soup = BeautifulSoup(html, "html.parser")

    # 個別ラップテーブルを特定
    table = soup.select_one("table#lap_summary") or soup.select_one("table.LapSummary_Table")
    if not table:
        logger.debug(f"[{race_id}] Lap summary table not found in horse laptime HTML")
        return horse_laps

    # ヘッダー最下段の th[data-furlong] から距離一覧を取得。
    # 2024年末時点の HTML では th.data-furlong が欠落しているケースがあるため、
    # テキスト(1F,2F...)やセル側の data-* も併用して復元する。
    header_rows = table.select("thead tr.Header")
    if not header_rows:
        logger.debug(f"[{race_id}] No header rows in lap summary table")
        return horse_laps

    last_header = header_rows[-1]
    dist_cells = last_header.select("th[data-furlong]") or last_header.select(
        "th"
    )

    distances: list[int] = []
    for idx, cell in enumerate(dist_cells, start=1):
        furlong = cell.get("data-furlong") or cell.get("data-distance")
        if not furlong:
            text = cell.get_text(strip=True)
            # 1F,2F... を 200m 単位に変換
            m = re.match(r"(\d+)[Ff]", text)
            if m:
                try:
                    furlong = int(m.group(1)) * 200
                except ValueError:
                    furlong = None
        if not furlong:
            # data-* もテキストも無い場合はヘッダー順に 200m ずつ積む仮距離
            furlong = idx * 200
        try:
            d = int(furlong)
        except ValueError:
            continue
        distances.append(d)

    # ヘッダーから距離が取れない場合、各 data-laptime セルに付与されている data-furlong などを頼る
    if not distances:
        logger.debug(f"[{race_id}] No distances found in lap summary header, trying cell attrs")

    # 各馬の行をパース
    body_rows = table.select("tbody tr.HorseList") or table.select("tbody tr")
    logger.debug(f"[{race_id}] Found {len(body_rows)} horse rows in lap summary")

    # data-furlong が見つからない場合、最初に距離列を探索し直す
    if not distances:
        for row in body_rows:
            lap_cells = _collect_lap_cells(row)
            for cell in lap_cells:
                furlong = _extract_distance_from_cell(cell)
                if furlong is None:
                    continue
                distances.append(furlong)
            if distances:
                break

    if not distances:
        logger.debug(f"[{race_id}] No distances found in lap summary (header or cells)")
        return horse_laps

    logger.debug(f"[{race_id}] Lap distances (m): {distances}")

    for row in body_rows:
        # 馬IDをリンクから取得
        horse_link = row.select_one("td.Horse_Info a[href*='/horse/']") or row.select_one(
            "a[href*='/horse/']"
        )
        if not horse_link:
            continue

        href = horse_link.get("href", "")
        m = re.search(r"/horse/(\d+)/", href)
        if not m:
            continue
        horse_id = m.group(1)

        # ラップタイムとポジションのセルを取得
        lap_cells = _collect_lap_cells(row)
        pos_cells = row.select("td.PositionCell")

        if not lap_cells:
            # ラップデータが無い馬はスキップ
            continue

        # th[data-furlong] の数と data-laptime の数を合わせる
        n = min(len(distances), len(lap_cells)) if distances else len(lap_cells)

        for i in range(n):
            cell = lap_cells[i]
            laptime_str = _extract_laptime(cell)
            if laptime_str is None:
                continue

            try:
                time_sec = float(laptime_str)
            except ValueError:
                continue

            section_m = (
                distances[i]
                if distances and i < len(distances)
                else _extract_distance_from_cell(cell) or (i + 1) * 200
            )

            # セクションごとのポジション（あれば）
            position: Optional[int] = None
            if i < len(pos_cells):
                raw_pos = pos_cells[i].get("data-position") or pos_cells[i].get_text(strip=True)
                if raw_pos and str(raw_pos).isdigit():
                    try:
                        position = int(raw_pos)
                    except ValueError:
                        position = None

            horse_laps.append(
                HorseLap(
                    race_id=race_id,
                    horse_id=horse_id,
                    section_m=section_m,
                    time_sec=time_sec,
                    position=position,
                )
            )

    logger.debug(f"[{race_id}] Parsed {len(horse_laps)} horse lap records from HTML")
    return horse_laps


def _collect_lap_cells(row: Tag) -> list[Tag]:
    """ラップタイムが入っているセル候補を広めに集める。"""

    lap_cells = row.select("td[data-laptime]")
    if lap_cells:
        return lap_cells

    candidates = []
    for td in row.select("td"):
        classes = td.get("class", [])
        if td.get("data-furlong") or td.get("data-distance"):
            candidates.append(td)
            continue
        if any(cls in {"CellDataWrap", "LapTimeCell", "TimeCell"} for cls in classes):
            candidates.append(td)
    return candidates


def _extract_laptime(cell: Tag) -> Optional[str]:
    """セルからラップタイム文字列を取得する。"""

    val = cell.get("data-laptime")
    if val:
        return val

    text = cell.get_text(strip=True)
    if text:
        return text

    inner = cell.select_one("*[data-laptime]")
    if inner and inner.get("data-laptime"):
        return inner.get("data-laptime")

    return None


def _extract_distance_from_cell(cell: Tag) -> Optional[int]:
    """セルの属性やテキストから距離(m)を推定する。"""

    furlong = cell.get("data-furlong") or cell.get("data-distance")
    if not furlong:
        text = cell.get_text(strip=True)
        m = re.match(r"(\d+)[Ff]", text)
        if m:
            try:
                furlong = int(m.group(1)) * 200
            except ValueError:
                furlong = None
    if furlong:
        try:
            return int(furlong)
        except ValueError:
            return None
    return None


def _parse_race_info(soup: BeautifulSoup, race_id: str) -> Race:
    """レース基本情報をパースする。"""
    race = Race(race_id=race_id)
    
    # レース番号を race_id から取得（末尾2桁）
    race.race_no = int(race_id[-2:])
    
    # レース名・グレード
    # <h1>第68回有馬記念(GI)</h1>
    h1 = soup.select_one("div.data_intro dl.racedata dd h1")
    if h1:
        race_name_text = h1.get_text(strip=True)
        race.name = race_name_text
        race.grade = _extract_grade(race_name_text)
        logger.debug(f"[{race_id}] Race name: {race.name}, grade: {race.grade}")
    else:
        logger.warning(f"[{race_id}] Could not find race name (h1 not found)")
    
    # コース情報・天候・馬場状態・発走時刻
    # <span>芝右2500m / 天候 : 晴 / 芝 : 良 / 発走 : 15:40</span>
    course_span = soup.select_one("div.data_intro dl.racedata dd p span")
    if course_span:
        course_text = course_span.get_text(strip=True)
        _parse_course_text(race, course_text)
        logger.debug(f"[{race_id}] Course: {race.course_type}, {race.distance}m, {race.track_condition}")
    else:
        logger.warning(f"[{race_id}] Could not find course info (span not found)")
    
    # 開催日・回次・クラス
    # <p class="smalltxt">2023年12月24日 5回中山8日目 3歳以上オープン (国際)(指)(定量)</p>
    smalltxt = soup.select_one("div.data_intro p.smalltxt")
    if smalltxt:
        smalltxt_text = smalltxt.get_text(strip=True)
        _parse_smalltxt(race, smalltxt_text)
        logger.debug(f"[{race_id}] Date: {race.date}, Place: {race.place}, Class: {race.race_class}")
    else:
        logger.warning(f"[{race_id}] Could not find smalltxt (date/place info)")
    
    return race


def _extract_grade(text: str) -> Optional[str]:
    """レース名からグレードを抽出する。"""
    # (GI), (GII), (GIII), (G1), (G2), (G3) など
    grade_patterns = [
        (r"\(GI\)|\(G1\)", "G1"),
        (r"\(GII\)|\(G2\)", "G2"),
        (r"\(GIII\)|\(G3\)", "G3"),
        (r"\(L\)|\(Listed\)", "Listed"),
        (r"\(OP\)|\(オープン\)", "OP"),
    ]
    
    for pattern, grade in grade_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return grade
    
    return None


def _parse_course_text(race: Race, text: str) -> None:
    """コース情報テキストをパースする。"""
    # 例: "芝右2500m / 天候 : 晴 / 芝 : 良 / 発走 : 15:40"
    
    # コースタイプ・回り・距離
    course_match = re.search(r"(芝|ダ|ダート|障)(右|左|直)?(\d+)m", text)
    if course_match:
        course_type_raw = course_match.group(1)
        if course_type_raw in ("芝",):
            race.course_type = "turf"
        elif course_type_raw in ("ダ", "ダート"):
            race.course_type = "dirt"
        elif course_type_raw in ("障",):
            race.course_type = "steeple"
        
        turn = course_match.group(2)
        if turn == "右":
            race.course_turn = "right"
        elif turn == "左":
            race.course_turn = "left"
        elif turn == "直":
            race.course_turn = "straight"
        
        race.distance = int(course_match.group(3))
    
    # 内/外コース
    if "内" in text:
        race.course_inout = "inner"
    elif "外" in text:
        race.course_inout = "outer"
    
    # 天候
    weather_match = re.search(r"天候\s*:\s*(\S+)", text)
    if weather_match:
        race.weather = weather_match.group(1)
    
    # 馬場状態
    track_match = re.search(r"(芝|ダ|ダート)\s*:\s*(良|稍重|稍|重|不良|不)", text)
    if track_match:
        condition = track_match.group(2)
        if condition == "稍":
            condition = "稍重"
        elif condition == "不":
            condition = "不良"
        race.track_condition = condition
    
    # 発走時刻
    start_match = re.search(r"発走\s*:\s*(\d{1,2}:\d{2})", text)
    if start_match:
        race.start_time = start_match.group(1)


def _parse_smalltxt(race: Race, text: str) -> None:
    """開催情報テキストをパースする。"""
    # 例: "2023年12月24日 5回中山8日目 3歳以上オープン (国際)(指)(定量)"
    
    # 日付
    date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if date_match:
        race.date = date(
            int(date_match.group(1)),
            int(date_match.group(2)),
            int(date_match.group(3)),
        )
    
    # 回次・場名・日目
    kai_match = re.search(r"(\d+)回([\u4e00-\u9fff]+?)(\d+)日目", text)
    if kai_match:
        race.kai = int(kai_match.group(1))
        race.place = kai_match.group(2)
        race.nichime = int(kai_match.group(3))
    
    # クラス情報（オープン、1勝クラス、未勝利など）
    class_patterns = [
        r"(新馬)",
        r"(未勝利)",
        r"(1勝クラス|500万下)",
        r"(2勝クラス|1000万下)",
        r"(3勝クラス|1600万下)",
        r"(オープン|OP)",
        r"(\d歳以上\d勝クラス)",
        r"(\d歳\d勝クラス)",
    ]
    
    for pattern in class_patterns:
        class_match = re.search(pattern, text)
        if class_match:
            race.race_class = class_match.group(1)
            break


def _parse_race_results(soup: BeautifulSoup, race_id: str) -> list[RaceResult]:
    """レース結果テーブルをパースする。"""
    results = []
    
    # <table class="race_table_01 nk_tb_common">
    table = soup.select_one("table.race_table_01")
    if not table:
        logger.warning(f"[{race_id}] Could not find race_table_01")
        return results
    
    rows = table.select("tbody tr")
    if not rows:
        # tbody がない場合
        rows = table.select("tr")[1:]  # ヘッダー行をスキップ
    
    for row in rows:
        cells = row.select("td")
        if len(cells) < 10:
            continue
        
        result = _parse_result_row(cells, race_id)
        if result:
            results.append(result)
    
    logger.debug(f"[{race_id}] Parsed {len(results)} race results")
    return results


def _parse_result_row(cells: list[Tag], race_id: str) -> Optional[RaceResult]:
    """結果テーブルの1行をパースする。"""
    try:
        # 着順（1列目）
        finish_order_text = cells[0].get_text(strip=True)
        finish_order = None
        finish_status = None
        
        if finish_order_text.isdigit():
            finish_order = int(finish_order_text)
        else:
            # 取消、除外、中止、失格など
            finish_status = finish_order_text
        
        # 枠番（2列目）
        frame_no_text = cells[1].get_text(strip=True)
        frame_no = int(frame_no_text) if frame_no_text.isdigit() else None
        
        # 馬番（3列目）
        horse_no_text = cells[2].get_text(strip=True)
        horse_no = int(horse_no_text) if horse_no_text.isdigit() else None
        
        # 馬名・馬ID（4列目）
        horse_cell = cells[3]
        horse_link = horse_cell.select_one("a[href*='/horse/']")
        horse_name = None
        horse_id = None
        
        if horse_link:
            horse_name = horse_link.get_text(strip=True)
            href = horse_link.get("href", "")
            horse_id_match = re.search(r"/horse/(\d+)", href)
            if horse_id_match:
                horse_id = horse_id_match.group(1)
        
        if not horse_id:
            return None
        
        # 性齢（5列目）
        sex_age_text = cells[4].get_text(strip=True)
        sex = None
        age = None
        sex_age_match = re.match(r"([牡牝セ騸])(\d+)", sex_age_text)
        if sex_age_match:
            sex = sex_age_match.group(1)
            age = int(sex_age_match.group(2))
        
        # 斤量（6列目）
        weight_text = cells[5].get_text(strip=True)
        weight = None
        try:
            weight = float(weight_text)
        except ValueError:
            pass
        
        # 騎手（7列目）
        jockey_cell = cells[6]
        jockey_link = jockey_cell.select_one("a[href*='/jockey/']")
        jockey_name = None
        jockey_id = None
        
        if jockey_link:
            jockey_name = jockey_link.get_text(strip=True)
            href = jockey_link.get("href", "")
            # /jockey/result/recent/00666/ や /jockey/00666/ に対応
            jockey_id_match = re.search(r"/jockey/(?:result/recent/)?(\d+)", href)
            if jockey_id_match:
                jockey_id = jockey_id_match.group(1)
        
        # タイム（8列目）
        time_text = cells[7].get_text(strip=True)
        time_str = time_text if time_text else None
        time_sec = _parse_time_to_seconds(time_text)
        
        # 着差（9列目）
        margin_text = cells[8].get_text(strip=True)
        margin = margin_text if margin_text else None
        
        # 以降の列は存在しない場合がある
        # 通過順（10列目）
        passing_order = None
        if len(cells) > 10:
            passing_order = cells[10].get_text(strip=True) or None
        
        # 上がり3F（11列目）
        last_3f = None
        if len(cells) > 11:
            last_3f_text = cells[11].get_text(strip=True)
            try:
                last_3f = float(last_3f_text)
            except ValueError:
                pass
        
        # 人気・オッズ（単勝人気、単勝オッズ）
        popularity = None
        win_odds = None
        
        # 人気（13列目あたり）
        if len(cells) > 13:
            pop_text = cells[13].get_text(strip=True)
            if pop_text.isdigit():
                popularity = int(pop_text)
        
        # オッズ（12列目あたり）
        if len(cells) > 12:
            odds_text = cells[12].get_text(strip=True)
            try:
                win_odds = float(odds_text)
            except ValueError:
                pass
        
        # 馬体重・増減（14列目あたり）
        body_weight = None
        body_weight_diff = None
        
        if len(cells) > 14:
            bw_text = cells[14].get_text(strip=True)
            # "468(-2)" のような形式
            bw_match = re.match(r"(\d+)\s*\(([+-]?\d+)\)", bw_text)
            if bw_match:
                body_weight = int(bw_match.group(1))
                body_weight_diff = int(bw_match.group(2))
            elif bw_text.isdigit():
                body_weight = int(bw_text)
        
        # タイム指数（スペースで囲まれた値、存在する場合）
        time_index = None
        if len(cells) > 9:
            ti_text = cells[9].get_text(strip=True)
            try:
                time_index = float(ti_text)
            except ValueError:
                pass
        
        # 調教師（17列目あたり）
        trainer_id = None
        trainer_name = None
        trainer_region = None
        
        for cell in cells[15:]:
            trainer_link = cell.select_one("a[href*='/trainer/']")
            if trainer_link:
                trainer_name = trainer_link.get_text(strip=True)
                href = trainer_link.get("href", "")
                # /trainer/result/recent/01061/ や /trainer/01061/ に対応
                trainer_id_match = re.search(r"/trainer/(?:result/recent/)?(\d+)", href)
                if trainer_id_match:
                    trainer_id = trainer_id_match.group(1)
                
                # 調教師名の前に地域マークがある場合
                cell_text = cell.get_text(strip=True)
                if "[栗]" in cell_text or "栗" in cell_text:
                    trainer_region = "栗東"
                elif "[美]" in cell_text or "美" in cell_text:
                    trainer_region = "美浦"
                break
        
        # 馬主（リンクから取得）
        owner_id = None
        owner_name = None
        
        for cell in cells[15:]:
            owner_link = cell.select_one("a[href*='/owner/']")
            if owner_link:
                owner_name = owner_link.get_text(strip=True)
                href = owner_link.get("href", "")
                # /owner/002803/ に対応
                owner_id_match = re.search(r"/owner/(?:result/recent/)?(\d+)", href)
                if owner_id_match:
                    owner_id = owner_id_match.group(1)
                break
        
        # 賞金（最後の列付近）
        prize_money = None
        for cell in reversed(cells[-3:]):
            prize_text = cell.get_text(strip=True)
            # "5,200.0" のような形式
            prize_clean = prize_text.replace(",", "")
            try:
                prize_money = float(prize_clean)
                break
            except ValueError:
                continue
        
        # 備考（remark）
        remark_text = None
        # 通常は備考列がある場合にパース
        
        return RaceResult(
            race_id=race_id,
            horse_id=horse_id,
            finish_order=finish_order,
            finish_status=finish_status,
            frame_no=frame_no,
            horse_no=horse_no,
            horse_name=horse_name,
            sex=sex,
            age=age,
            weight=weight,
            jockey_id=jockey_id,
            jockey_name=jockey_name,
            time_str=time_str,
            time_sec=time_sec,
            margin=margin,
            passing_order=passing_order,
            last_3f=last_3f,
            win_odds=win_odds,
            popularity=popularity,
            body_weight=body_weight,
            body_weight_diff=body_weight_diff,
            time_index=time_index,
            trainer_id=trainer_id,
            trainer_name=trainer_name,
            trainer_region=trainer_region,
            owner_id=owner_id,
            owner_name=owner_name,
            prize_money=prize_money,
            remark_text=remark_text,
        )
        
    except Exception as e:
        logger.warning(f"[{race_id}] Failed to parse result row: {e}")
        return None


def _parse_time_to_seconds(time_str: str) -> Optional[float]:
    """タイム文字列を秒に変換する。"""
    if not time_str:
        return None
    
    # "1:34.5" → 94.5
    # "59.8" → 59.8
    time_str = time_str.strip()
    
    if ":" in time_str:
        parts = time_str.split(":")
        try:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        except (ValueError, IndexError):
            return None
    else:
        try:
            return float(time_str)
        except ValueError:
            return None


def _parse_payouts(soup: BeautifulSoup, race_id: str) -> list[Payout]:
    """払い戻しテーブルをパースする。"""
    payouts = []
    
    # <table class="pay_table_01">
    tables = soup.select("table.pay_table_01")
    
    for table in tables:
        rows = table.select("tr")
        
        for row in rows:
            th = row.select_one("th")
            tds = row.select("td")
            
            if not th or len(tds) < 2:
                continue
            
            bet_type_text = th.get_text(strip=True)
            bet_type = _normalize_bet_type(bet_type_text)
            
            if not bet_type:
                continue
            
            # 組み合わせ・配当・人気
            # <br>で区切られている場合がある
            comb_td = tds[0]
            payout_td = tds[1]
            pop_td = tds[2] if len(tds) > 2 else None
            
            combinations = _split_br(comb_td)
            payout_strs = _split_br(payout_td)
            pop_strs = _split_br(pop_td) if pop_td else []
            
            # リストの長さを揃える
            max_len = max(len(combinations), len(payout_strs))
            combinations = _pad_list(combinations, max_len)
            payout_strs = _pad_list(payout_strs, max_len)
            pop_strs = _pad_list(pop_strs, max_len)
            
            for comb, payout_str, pop_str in zip(combinations, payout_strs, pop_strs):
                if not comb:
                    continue
                
                # 組み合わせを正規化（全角→半角、スペース除去）
                comb = comb.replace("　", "-").replace(" ", "-").replace("→", "-").replace("－", "-")
                comb = re.sub(r"\s+", "", comb)
                
                # 払い戻し金額
                payout_int = _parse_payout_value(payout_str)
                if payout_int is None:
                    continue
                
                pop_int = None
                if pop_str and pop_str.isdigit():
                    pop_int = int(pop_str)
                
                payouts.append(Payout(
                    race_id=race_id,
                    bet_type=bet_type,
                    combination=comb,
                    payout=payout_int,
                    popularity=pop_int,
                ))
    
    return payouts


def _normalize_bet_type(text: str) -> Optional[str]:
    """券種名を正規化する。"""
    mapping = {
        "単勝": "単勝",
        "複勝": "複勝",
        "枠連": "枠連",
        "馬連": "馬連",
        "ワイド": "ワイド",
        "馬単": "馬単",
        "三連複": "三連複",
        "三連単": "三連単",
    }
    for key, value in mapping.items():
        if key in text:
            return value
    return None


def _split_br(td: Tag) -> list[str]:
    """td要素内の<br>で区切られたテキストを分割する。"""
    # brタグを改行文字に置換してから分割
    for br in td.find_all("br"):
        br.replace_with("\n")
    
    text = td.get_text()
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts


def _pad_list(lst: list, length: int) -> list:
    """リストを指定長にパディングする。"""
    while len(lst) < length:
        lst.append("")
    return lst


def _parse_payout_value(text: str) -> Optional[int]:
    """配当テキストを整数に変換する。"""
    # "2,730" -> 2730
    cleaned = text.replace(",", "").strip()
    if cleaned.isdigit():
        return int(cleaned)
    return None


def _parse_corners(soup: BeautifulSoup, race_id: str) -> Optional[Corner]:
    """コーナー通過順位をパースする。"""
    # <table summary="コーナー通過順位">
    table = soup.select_one("table[summary='コーナー通過順位']")
    if not table:
        return None
    
    corner = Corner(race_id=race_id)
    
    rows = table.select("tr")
    for row in rows:
        th = row.select_one("th")
        td = row.select_one("td")
        if not th or not td:
            continue
        
        corner_name = th.get_text(strip=True)
        corner_value = td.get_text(strip=True)
        
        if "1" in corner_name:
            corner.corner_1 = corner_value
        elif "2" in corner_name:
            corner.corner_2 = corner_value
        elif "3" in corner_name:
            corner.corner_3 = corner_value
        elif "4" in corner_name:
            corner.corner_4 = corner_value
    
    return corner


def _parse_race_lap_times(soup: BeautifulSoup, race_id: str) -> list[LapTime]:
    """レース全体のラップタイムをパースする。"""
    lap_times = []
    
    # <table summary="ラップタイム">
    table = soup.select_one("table[summary='ラップタイム']")
    if not table:
        return lap_times
    
    # ヘッダー行（距離）とデータ行（タイム）
    rows = table.select("tr")
    if len(rows) < 2:
        return lap_times
    
    # 距離ヘッダー
    header_cells = rows[0].select("th, td")
    distances = []
    for cell in header_cells:
        text = cell.get_text(strip=True)
        # "100m", "200", "1C" など
        dist_match = re.search(r"(\d+)", text)
        if dist_match:
            distances.append(int(dist_match.group(1)))
    
    # タイムデータ
    data_cells = rows[1].select("td")
    
    for i, (dist, cell) in enumerate(zip(distances, data_cells)):
        time_text = cell.get_text(strip=True)
        try:
            time_sec = float(time_text)
            lap_times.append(LapTime(
                race_id=race_id,
                lap_index=i,
                distance_m=dist,
                time_sec=time_sec,
            ))
        except ValueError:
            pass
    
    return lap_times


def _parse_baba_info(soup: BeautifulSoup) -> Optional[tuple[Optional[int], Optional[str]]]:
    """馬場指数・コメントをパースする。"""
    # <div class="data_intro"> 内の馬場情報
    # 馬場指数: <span>-9</span>
    baba_div = soup.select_one("div.race_date")
    
    baba_index = None
    baba_comment = None
    
    if baba_div:
        text = baba_div.get_text(strip=True)
        # 馬場指数を探す
        index_match = re.search(r"馬場指数[：:]?\s*([+-]?\d+)", text)
        if index_match:
            baba_index = int(index_match.group(1))
    
    # data_intro 内で探す
    data_intro = soup.select_one("div.data_intro")
    if data_intro:
        text = data_intro.get_text()
        index_match = re.search(r"馬場指数[：:]?\s*([+-]?\d+)", text)
        if index_match:
            baba_index = int(index_match.group(1))
    
    return (baba_index, baba_comment) if baba_index is not None else None


def _parse_analysis_comment(soup: BeautifulSoup) -> Optional[str]:
    """レース分析コメントをパースする。"""
    # <div class="race_analysis"> や類似のセレクタ
    analysis_div = soup.select_one("div.race_analysis")
    if analysis_div:
        return analysis_div.get_text(strip=True)
    
    return None


def _parse_short_comments(soup: BeautifulSoup, race_id: str) -> list[HorseShortComment]:
    """注目馬の短評をパースする（マスター会員限定）。"""
    comments = []
    
    # 短評テーブルを探す
    # <table class="short_comment_table"> のような形式
    table = soup.select_one("table.short_comment_table")
    if not table:
        # 別のセレクタを試す
        table = soup.select_one("div.short_comment table")
    
    if not table:
        return comments
    
    rows = table.select("tr")
    for row in rows:
        cells = row.select("td")
        if len(cells) < 2:
            continue
        
        # 馬名
        horse_cell = cells[0]
        horse_link = horse_cell.select_one("a")
        if not horse_link:
            continue
        
        horse_name = horse_link.get_text(strip=True)
        
        # 着順（あれば）
        finish_order = None
        order_cell = horse_cell.select_one("span.order")
        if order_cell:
            order_text = order_cell.get_text(strip=True)
            if order_text.isdigit():
                finish_order = int(order_text)
        
        # コメント
        comment_text = cells[1].get_text(strip=True) if len(cells) > 1 else None
        
        if horse_name and comment_text:
            comments.append(HorseShortComment(
                race_id=race_id,
                horse_id=None,  # 後で補完
                horse_name=horse_name,
                finish_order=finish_order,
                comment=comment_text,
            ))
    
    return comments