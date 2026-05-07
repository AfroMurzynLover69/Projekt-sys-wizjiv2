import base64
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from html import escape
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import AI_PROFILES, APP_TITLE, DANGEROUS_PLATES_LIST
from .config import DEFAULT_AI_PROFILE, RUNTIME_DEVICE, get_ai_profile
from .pipeline import DetectionEvent, DetectionResult, analyze_frame, run_detection
from .plates import clean_plate_text, extract_polish_root

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DATA_DIR = BASE_DIR / "web_data"
UPLOADS_DIR = WEB_DATA_DIR / "uploads"
RESULTS_DIR = WEB_DATA_DIR / "results"
HISTORY_THUMBS_DIR = WEB_DATA_DIR / "history"
HISTORY_FILE = WEB_DATA_DIR / "history.json"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_THUMBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_TITLE)
app.mount("/files", StaticFiles(directory=str(WEB_DATA_DIR)), name="files")
LIVE_HISTORY_KEYS: dict[str, set[tuple[str, int]]] = {}
LAST_CPU_SAMPLE: tuple[int, int] | None = None
LAST_PROCESS_SAMPLE: tuple[float, int] | None = None


def _read_dangerous_plates() -> set[str]:
    if not DANGEROUS_PLATES_LIST.exists():
        return set()
    try:
        lines = DANGEROUS_PLATES_LIST.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    values: set[str] = set()
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        cleaned = clean_plate_text(raw)
        if cleaned:
            values.add(cleaned)
    return values


def _write_dangerous_plates(plates: set[str]) -> None:
    DANGEROUS_PLATES_LIST.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Watchlista tablic "niebezpiecznych" kierowcow.',
        "# Format: jedna rejestracja na linie.",
        "# Dozwolone sa tez wpisy z myslnikiem lub spacja (np. WR-3804R).",
        "",
        *sorted(plates),
        "",
    ]
    DANGEROUS_PLATES_LIST.write_text("\n".join(lines), encoding="utf-8")


def _plate_region(plate: str) -> str:
    return extract_polish_root(plate) or "INNE"


def _build_ranking(history: list[dict]) -> tuple[list[tuple[str, int]], list[dict]]:
    region_counts: dict[str, int] = {}
    dangerous_plates = _read_dangerous_plates()
    dangerous_summary: dict[str, dict] = {}

    for item in history:
        plate = clean_plate_text(str(item.get("plate", "")))
        if not plate:
            continue

        region = _plate_region(plate)
        region_counts[region] = region_counts.get(region, 0) + 1

        if plate not in dangerous_plates:
            continue

        entry = dangerous_summary.setdefault(
            plate,
            {"plate": plate, "count": 0, "events": []},
        )
        entry["count"] += 1
        entry["events"].append(
            {
                "source": str(item.get("source", "")),
                "time": str(item.get("time", "")),
            }
        )

    ranked_regions = sorted(
        region_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    ranked_dangerous = sorted(
        dangerous_summary.values(),
        key=lambda item: (-item["count"], item["plate"]),
    )
    return ranked_regions, ranked_dangerous


def _parse_active_filters(active_filter: str) -> set[str]:
    filters: set[str] = set()
    for raw in str(active_filter).split(","):
        value = raw.strip().lower()
        if not value or value == "normal":
            continue
        filters.add(value)
    return filters


def _read_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    cleaned_history: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        plate = clean_plate_text(str(item.get("plate", "")))
        if not plate:
            continue
        cleaned_history.append(
            {
                **item,
                "plate": plate,
            }
        )
    return cleaned_history


def _write_history(entries: list[dict]) -> None:
    HISTORY_FILE.write_text(
        json.dumps(entries[:120], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _thumbnail_url(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        relative = path.relative_to(WEB_DATA_DIR)
    except ValueError:
        return None
    return "/files/" + "/".join(relative.parts)


def _event_to_entry(event: DetectionEvent) -> dict:
    return {
        "id": uuid4().hex,
        "plate": event.plate_text,
        "source": event.source_name,
        "time": event.time_label,
        "seconds": round(event.time_seconds, 2),
        "thumbnail_url": _thumbnail_url(event.thumbnail_path),
    }


def _append_history(events: list[DetectionEvent]) -> list[dict]:
    history = _read_history()
    items = [_event_to_entry(event) for event in events]
    if items:
        history = items + history
        _write_history(history)
    return history[:120]


def _append_live_history(session_id: str, events: list[DetectionEvent]) -> None:
    if not events:
        return
    seen = LIVE_HISTORY_KEYS.setdefault(session_id, set())
    fresh: list[DetectionEvent] = []
    for event in events:
        key = (event.plate_text, int(event.time_seconds // 60))
        if key in seen:
            continue
        seen.add(key)
        fresh.append(event)
    _append_history(fresh)


def _render_history(history: list[dict]) -> str:
    if not history:
        return '<div class="history-empty">Brak wykryc</div>'

    chunks: list[str] = []
    for item in history:
        thumb = item.get("thumbnail_url")
        thumb_html = f'<img src="{escape(thumb)}" alt="{escape(item["plate"])}">' if thumb else ""
        chunks.append(
            f"""
<article class="history-item">
  <div class="history-thumb">{thumb_html}</div>
  <div class="history-copy">
    <strong>{escape(item["plate"])}</strong>
    <span>{escape(item["source"])}</span>
    <span>{escape(item["time"])}</span>
  </div>
</article>
"""
        )
    return "".join(chunks)


def _render_region_ranking(rows: list[tuple[str, int]]) -> str:
    if not rows:
        return '<div class="history-empty">Brak danych</div>'

    chunks: list[str] = []
    for index, (region, count) in enumerate(rows[:12], start=1):
        chunks.append(
            f"""
<article class="rank-item">
  <span class="rank-index">{index:02d}</span>
  <strong>{escape(region)}</strong>
  <span>{count}</span>
</article>
"""
        )
    return "".join(chunks)


def _render_dangerous_ranking(rows: list[dict]) -> str:
    if not rows:
        return '<div class="history-empty">Brak czerwonych tablic</div>'

    chunks: list[str] = []
    for item in rows[:20]:
        moments = "".join(
            f'<span class="danger-chip">{escape(event["source"])} {escape(event["time"])}</span>'
            for event in item["events"][:8]
        )
        chunks.append(
            f"""
<article class="danger-item">
  <div class="danger-head">
    <strong>{escape(item["plate"])}</strong>
    <span>{item["count"]}</span>
  </div>
  <div class="danger-times">{moments}</div>
</article>
"""
        )
    return "".join(chunks)


def _build_analysis(history: list[dict]) -> dict:
    dangerous_plates = _read_dangerous_plates()
    plates = [clean_plate_text(str(item.get("plate", ""))) for item in history]
    plates = [plate for plate in plates if plate]
    plate_counts = Counter(plates)
    region_counts = Counter(_plate_region(plate) for plate in plates)
    dangerous_hits = [plate for plate in plates if plate in dangerous_plates]
    repeated = [plate for plate, count in plate_counts.items() if count > 1]
    top_region = region_counts.most_common(1)[0] if region_counts else ("-", 0)

    return {
        "total": len(plates),
        "unique": len(plate_counts),
        "repeated": len(repeated),
        "dangerous_hits": len(dangerous_hits),
        "dangerous_unique": len(set(dangerous_hits)),
        "top_region": top_region,
        "plate_counts": plate_counts.most_common(12),
        "dangerous_plates": dangerous_plates,
    }


def _render_stat_cards(analysis: dict) -> str:
    region, region_count = analysis["top_region"]
    cards = [
        ("Wykrycia", str(analysis["total"]), "wszystkie odczyty"),
        ("Unikalne", str(analysis["unique"]), "rozne tablice"),
        ("Powtorzenia", str(analysis["repeated"]), "tablice widziane wiecej niz raz"),
        ("Watchlista", str(analysis["dangerous_unique"]), f'{analysis["dangerous_hits"]} trafien'),
        ("Region", str(region), f"{region_count} wykryc"),
    ]
    chunks: list[str] = []
    for label, value, caption in cards:
        chunks.append(
            f"""
<article class="metric-card">
  <span>{escape(label)}</span>
  <strong>{escape(value)}</strong>
  <small>{escape(caption)}</small>
</article>
"""
        )
    return "".join(chunks)


def _render_plate_frequency(rows: list[tuple[str, int]], dangerous_plates: set[str]) -> str:
    if not rows:
        return '<div class="empty-state">Brak danych do analizy</div>'

    max_count = max(count for _plate, count in rows) or 1
    chunks: list[str] = []
    for plate, count in rows:
        width = max(8, round((count / max_count) * 100))
        danger_class = " danger" if plate in dangerous_plates else ""
        badge = '<span class="analysis-badge danger">watchlista</span>' if plate in dangerous_plates else ""
        chunks.append(
            f"""
<article class="plate-row{danger_class}">
  <div class="plate-row-main">
    <strong>{escape(plate)}</strong>
    {badge}
    <span>{count}x</span>
  </div>
  <div class="plate-bar"><i style="width:{width}%"></i></div>
</article>
"""
        )
    return "".join(chunks)


def _render_analysis_history(history: list[dict], dangerous_plates: set[str]) -> str:
    if not history:
        return '<div class="empty-state">Brak wykryc. Wgraj plik albo uruchom przechwytywanie ekranu.</div>'

    chunks: list[str] = []
    for item in history[:80]:
        plate = clean_plate_text(str(item.get("plate", "")))
        if not plate:
            continue
        thumb = item.get("thumbnail_url")
        thumb_html = f'<img src="{escape(thumb)}" alt="{escape(plate)}">' if thumb else ""
        is_dangerous = plate in dangerous_plates
        row_class = "analysis-event danger" if is_dangerous else "analysis-event"
        badge = '<span class="analysis-badge danger">watchlista</span>' if is_dangerous else '<span class="analysis-badge">normalna</span>'
        region = _plate_region(plate)
        chunks.append(
            f"""
<article class="{row_class}" data-plate="{escape(plate)}">
  <div class="event-thumb">{thumb_html}</div>
  <div class="event-main">
    <div class="event-title">
      <strong>{escape(plate)}</strong>
      {badge}
    </div>
    <span>{escape(str(item.get("source", "")))}</span>
  </div>
  <div class="event-meta">
    <strong>{escape(region)}</strong>
    <span>{escape(str(item.get("time", "")))}</span>
  </div>
</article>
"""
        )
    return "".join(chunks) if chunks else '<div class="empty-state">Brak poprawnych wykryc</div>'


def _render_watchlist_items(plates: set[str]) -> str:
    if not plates:
        return '<div class="empty-state">Lista jest pusta</div>'

    chunks: list[str] = []
    for plate in sorted(plates):
        region = _plate_region(plate)
        chunks.append(
            f"""
<article class="watchlist-item">
  <div>
    <strong>{escape(plate)}</strong>
    <span>Region: {escape(region)}</span>
  </div>
  <form action="/watchlist/remove" method="post">
    <input type="hidden" name="plate" value="{escape(plate)}">
    <button class="watchlist-remove" type="submit" aria-label="Usun {escape(plate)}">Usun</button>
  </form>
</article>
"""
        )
    return "".join(chunks)


def _render_search_plate_buttons(history: list[dict], dangerous_plates: set[str]) -> str:
    counts = Counter(clean_plate_text(str(item.get("plate", ""))) for item in history)
    counts.pop("", None)
    if not counts:
        return '<div class="empty-state" id="searchPlateEmpty">Brak tablic w historii</div>'

    chunks: list[str] = []
    for plate, count in sorted(counts.items()):
        state = "danger" if plate in dangerous_plates else "normal"
        label = "black" if plate in dangerous_plates else "normal"
        chunks.append(
            f"""
<button class="search-plate-button {state}" type="button" data-search-plate="{escape(plate)}" hidden>
  <strong>{escape(plate)}</strong>
  <span>{count} zdjec | {label}</span>
</button>
"""
        )
    return "".join(chunks)


def _render_search_results(history: list[dict], dangerous_plates: set[str]) -> str:
    if not history:
        return '<div class="empty-state">Brak wykryc. Najpierw uruchom analize pliku albo ekranu.</div>'

    chunks: list[str] = []
    for item in history:
        plate = clean_plate_text(str(item.get("plate", "")))
        if not plate:
            continue
        is_dangerous = plate in dangerous_plates
        state = "danger" if is_dangerous else "normal"
        label = "blacklista" if is_dangerous else "normalna"
        thumb = item.get("thumbnail_url")
        thumb_html = f'<img src="{escape(thumb)}" alt="{escape(plate)}">' if thumb else '<span>brak zdjecia</span>'
        chunks.append(
            f"""
<article class="search-card {state}" data-search-card data-search-plate="{escape(plate)}" hidden>
  <div class="search-thumb">{thumb_html}</div>
  <div class="search-card-copy">
    <div>
      <strong>{escape(plate)}</strong>
      <span class="search-state {state}">{label}</span>
    </div>
    <span>{escape(str(item.get("time", "")))}</span>
    <small>{escape(str(item.get("source", "")))}</small>
  </div>
</article>
"""
        )
    return "".join(chunks) if chunks else '<div class="empty-state">Brak poprawnych tablic w historii</div>'


def _render_result(result: DetectionResult, original_name: str) -> str:
    if result.exit_code != 0 or result.output_path is None:
        return f"""
<section class="panel result-panel">
  <div class="result-empty">
    <div class="preview-message">
      <strong>Nie udalo sie przeanalizowac pliku</strong>
      <span>{escape(original_name)}</span>
    </div>
  </div>
</section>
"""

    result_url = escape(_thumbnail_url(result.output_path) or "")
    if result.media_kind == "image":
        preview = f'<img class="preview-media" src="{result_url}" alt="{escape(original_name)}">'
    else:
        preview = f'<video class="preview-media" src="{result_url}" controls></video>'

    return f"""
<section class="panel result-panel">
  {preview}
</section>
"""


def _read_cpu_sample() -> tuple[int, int] | None:
    try:
        first_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    parts = first_line.split()
    if not parts or parts[0] != "cpu":
        return None
    try:
        values = [int(value) for value in parts[1:]]
    except ValueError:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total, idle


def _cpu_percent() -> int | None:
    global LAST_CPU_SAMPLE
    sample = _read_cpu_sample()
    if sample is None:
        try:
            load_1m = os.getloadavg()[0]
            cpu_count = os.cpu_count() or 1
            return round(min(100.0, (load_1m / cpu_count) * 100.0))
        except OSError:
            return None

    if LAST_CPU_SAMPLE is None:
        LAST_CPU_SAMPLE = sample
        return None

    total_delta = sample[0] - LAST_CPU_SAMPLE[0]
    idle_delta = sample[1] - LAST_CPU_SAMPLE[1]
    LAST_CPU_SAMPLE = sample
    if total_delta <= 0:
        return None
    busy = 100.0 * (1.0 - (idle_delta / total_delta))
    return round(max(0.0, min(100.0, busy)))


def _ram_percent() -> int | None:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in lines:
        key, _, value = line.partition(":")
        raw_number = value.strip().split(" ")[0]
        try:
            values[key] = int(raw_number)
        except ValueError:
            continue
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return None
    used = total - available
    return round(max(0.0, min(100.0, (used / total) * 100.0)))


def _read_process_cpu_ticks() -> int | None:
    try:
        stat = Path("/proc/self/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    _, _separator, tail = stat.rpartition(") ")
    if not tail:
        return None
    parts = tail.split()
    try:
        utime = int(parts[11])
        stime = int(parts[12])
    except (IndexError, ValueError):
        return None
    return utime + stime


def _process_cpu_percent() -> int | None:
    global LAST_PROCESS_SAMPLE
    ticks = _read_process_cpu_ticks()
    if ticks is None:
        return None

    now = time.monotonic()
    if LAST_PROCESS_SAMPLE is None:
        LAST_PROCESS_SAMPLE = (now, ticks)
        return None

    previous_time, previous_ticks = LAST_PROCESS_SAMPLE
    LAST_PROCESS_SAMPLE = (now, ticks)
    elapsed = now - previous_time
    if elapsed <= 0:
        return None

    tick_rate = os.sysconf("SC_CLK_TCK")
    cpu_seconds = (ticks - previous_ticks) / max(1, tick_rate)
    return round(max(0.0, cpu_seconds / elapsed * 100.0))


def _process_status() -> dict:
    process_path = ""
    try:
        process_path = str(Path("/proc/self/exe").resolve())
    except OSError:
        process_path = ""

    return {
        "pid": os.getpid(),
        "cpu": _process_cpu_percent(),
        "path": process_path,
    }


def _gpu_status() -> dict:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {
            "available": False,
            "label": "brak",
            "utilization": None,
            "memory": None,
        }

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.2,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "available": False,
            "label": "blad",
            "utilization": None,
            "memory": None,
        }

    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 4:
        return {
            "available": False,
            "label": "brak danych",
            "utilization": None,
            "memory": None,
        }
    try:
        utilization = int(parts[0])
        memory_used = int(parts[1])
        memory_total = int(parts[2])
    except ValueError:
        utilization = None
        memory_used = None
        memory_total = None

    memory = None
    if memory_used is not None and memory_total:
        memory = round((memory_used / memory_total) * 100.0)

    return {
        "available": True,
        "label": parts[3],
        "utilization": utilization,
        "memory": memory,
    }


def _render_nav(active_view: str) -> str:
    items = [
        ("/", "Media", active_view == "media"),
        ("/analysis", "Analiza", active_view == "analysis"),
        ("/search", "Szukaj", active_view == "search"),
        ("/watchlist", "Watchlista", active_view == "watchlist"),
    ]
    links: list[str] = []
    for href, label, is_active in items:
        class_name = "task-link active" if is_active else "task-link"
        links.append(f'<a class="{class_name}" href="{href}">{label}</a>')
    return "".join(links)


def _page(
    content: str,
    active_view: str,
    active_filter: str,
    active_profile: str = DEFAULT_AI_PROFILE,
) -> HTMLResponse:
    profile = get_ai_profile(active_profile)
    profile_json = escape(json.dumps(AI_PROFILES), quote=True)
    return HTMLResponse(
        f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{APP_TITLE}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f2ec;
      --panel: rgba(255,255,255,0.94);
      --line: rgba(17,17,17,0.08);
      --ink: #111111;
      --muted: #686868;
      --accent: #111111;
      --shadow: 0 18px 40px rgba(17,17,17,0.08);
      --taskbar-height: 62px;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "DejaVu Sans", "Noto Sans", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(17,17,17,0.06), transparent 24%),
        linear-gradient(180deg, #f7f6f2 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .taskbar {{
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 50;
      min-height: var(--taskbar-height);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 7px 12px;
      border-radius: 0;
      background: rgba(10,10,10,0.94);
      color: #fff;
      border-top: 1px solid rgba(255,255,255,0.1);
      box-shadow: 0 -10px 32px rgba(0,0,0,0.18);
      backdrop-filter: blur(14px);
    }}
    .serial-monitor {{
      position: fixed;
      right: 14px;
      bottom: calc(var(--taskbar-height) + 14px);
      z-index: 45;
      width: min(440px, calc(100vw - 28px));
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(10,10,10,0.92);
      color: #d7ffd9;
      box-shadow: 0 16px 40px rgba(0,0,0,0.22);
      backdrop-filter: blur(12px);
      font-family: "DejaVu Sans Mono", "Noto Sans Mono", monospace;
    }}
    .serial-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 34px;
      padding: 8px 10px;
      border-bottom: 1px solid rgba(255,255,255,0.1);
      color: rgba(255,255,255,0.72);
      font-size: 0.76rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .monitor-state {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      color: rgba(255,255,255,0.62);
      font-size: 0.72rem;
      letter-spacing: 0;
      text-transform: none;
    }}
    .serial-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #1f9d55;
      box-shadow: 0 0 12px rgba(31,157,85,0.8);
    }}
    .serial-line {{
      min-height: 42px;
      display: flex;
      align-items: center;
      padding: 10px;
      color: #a8ffb0;
      font-size: 0.86rem;
      line-height: 1.35;
      word-break: break-word;
    }}
    .task-brand {{
      min-width: 64px;
      padding: 0 8px;
      font-weight: 800;
      letter-spacing: 0.05em;
    }}
    .task-nav {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .task-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 46px;
      padding: 0 16px;
      border-radius: 6px;
      color: rgba(255,255,255,0.78);
      text-decoration: none;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.06);
      font-weight: 700;
      transition: background 0.2s ease, color 0.2s ease;
    }}
    .task-link:hover {{
      background: rgba(255,255,255,0.1);
      color: #fff;
    }}
    .task-link.active {{
      background: #fff;
      color: #111;
    }}
    .task-left,
    .task-status {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .task-status {{
      justify-content: flex-end;
      flex-wrap: wrap;
    }}
    .status-chip {{
      display: inline-grid;
      grid-template-columns: auto auto;
      align-items: center;
      column-gap: 8px;
      min-height: 46px;
      padding: 6px 10px;
      border-radius: 6px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .status-icon {{
      width: 28px;
      height: 28px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 4px;
      background: rgba(255,255,255,0.12);
      color: #fff;
    }}
    .status-icon svg {{
      width: 18px;
      height: 18px;
      display: block;
      stroke: currentColor;
    }}
    .status-copy {{
      display: grid;
      gap: 2px;
      min-width: 46px;
    }}
    .status-chip.process .status-copy {{
      min-width: 260px;
      max-width: 420px;
    }}
    .process-path {{
      display: block;
      max-width: 420px;
      overflow: hidden;
      color: rgba(255,255,255,0.68);
      font-size: 0.72rem;
      line-height: 1.2;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .status-copy span {{
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }}
    .status-copy strong {{
      color: #fff;
      font-size: 0.88rem;
      line-height: 1;
      white-space: nowrap;
    }}
    .status-chip.online .status-icon {{
      background: #1f9d55;
    }}
    .status-chip.warning .status-icon {{
      background: #b7791f;
    }}
    .status-chip.offline .status-icon {{
      background: #8f1111;
    }}
    .history-list {{
      display: grid;
      gap: 12px;
    }}
    .history-item {{
      display: grid;
      grid-template-columns: 74px 1fr;
      gap: 12px;
      padding: 10px;
      border-radius: 16px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.06);
    }}
    .history-thumb {{
      width: 74px;
      height: 74px;
      border-radius: 12px;
      overflow: hidden;
      background: rgba(255,255,255,0.08);
    }}
    .history-thumb img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .history-copy {{
      min-width: 0;
      display: grid;
      align-content: center;
      gap: 5px;
    }}
    .history-copy strong {{
      font-size: 1rem;
      letter-spacing: 0.04em;
    }}
    .history-copy span {{
      color: rgba(255,255,255,0.68);
      font-size: 0.9rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .history-empty {{
      padding: 16px;
      border-radius: 16px;
      background: rgba(255,255,255,0.06);
      color: rgba(255,255,255,0.62);
    }}
    .split-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .rank-list {{
      display: grid;
      gap: 12px;
    }}
    .rank-item {{
      display: grid;
      grid-template-columns: 64px 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(17,17,17,0.04);
      border: 1px solid rgba(17,17,17,0.08);
    }}
    .rank-index {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 48px;
      height: 48px;
      border-radius: 14px;
      background: rgba(17,17,17,0.92);
      color: #fff;
      font-weight: 700;
      letter-spacing: 0.06em;
    }}
    .danger-list {{
      display: grid;
      gap: 12px;
    }}
    .danger-item {{
      display: grid;
      gap: 12px;
      padding: 16px;
      border-radius: 18px;
      background: rgba(157,17,17,0.06);
      border: 1px solid rgba(157,17,17,0.14);
    }}
    .danger-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .danger-head strong {{
      font-size: 1.05rem;
      letter-spacing: 0.05em;
      color: #8f1111;
    }}
    .danger-head span {{
      min-width: 44px;
      padding: 8px 12px;
      border-radius: 999px;
      background: #8f1111;
      color: #fff;
      text-align: center;
      font-weight: 700;
    }}
    .danger-times {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .danger-chip {{
      display: inline-flex;
      align-items: center;
      min-height: 36px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(17,17,17,0.06);
      color: var(--ink);
      font-size: 0.92rem;
    }}
    .panel-head {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .panel-head p {{
      margin: 0 0 4px;
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .panel-head h2 {{
      margin: 0;
      font-size: 1.35rem;
    }}
    .analysis-layout {{
      display: grid;
      gap: 18px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric-card {{
      min-height: 116px;
      display: grid;
      align-content: space-between;
      gap: 8px;
      padding: 16px;
      border: 1px solid rgba(17,17,17,0.08);
      border-radius: 8px;
      background: rgba(255,255,255,0.7);
    }}
    .metric-card span,
    .metric-card small {{
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .metric-card strong {{
      font-size: 2rem;
      line-height: 1;
    }}
    .analysis-columns {{
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(360px, 0.95fr);
      gap: 18px;
      align-items: start;
    }}
    .plate-list,
    .analysis-events {{
      display: grid;
      gap: 10px;
    }}
    .plate-row {{
      display: grid;
      gap: 10px;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid rgba(17,17,17,0.08);
      background: rgba(17,17,17,0.035);
    }}
    .plate-row.danger {{
      background: rgba(157,17,17,0.08);
      border-color: rgba(157,17,17,0.18);
    }}
    .plate-row-main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 10px;
      align-items: center;
    }}
    .plate-row-main strong {{
      letter-spacing: 0.05em;
    }}
    .plate-bar {{
      height: 7px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(17,17,17,0.1);
    }}
    .plate-bar i {{
      display: block;
      height: 100%;
      border-radius: inherit;
      background: rgba(17,17,17,0.82);
    }}
    .analysis-badge {{
      min-height: 28px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 5px 9px;
      border-radius: 999px;
      background: rgba(17,17,17,0.08);
      color: var(--ink);
      font-size: 0.78rem;
      font-weight: 800;
      white-space: nowrap;
    }}
    .analysis-badge.danger {{
      background: #8f1111;
      color: #fff;
    }}
    .analysis-event {{
      display: grid;
      grid-template-columns: 78px minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid rgba(17,17,17,0.08);
      background: rgba(255,255,255,0.68);
    }}
    .analysis-event.danger {{
      border-color: rgba(157,17,17,0.2);
      background: rgba(157,17,17,0.07);
    }}
    .event-thumb {{
      width: 78px;
      height: 58px;
      border-radius: 6px;
      overflow: hidden;
      background: #111;
    }}
    .event-thumb img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .event-main {{
      min-width: 0;
      display: grid;
      gap: 5px;
    }}
    .event-title {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}
    .event-title strong {{
      letter-spacing: 0.05em;
    }}
    .event-main span,
    .event-meta span {{
      color: var(--muted);
      font-size: 0.9rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .event-meta {{
      display: grid;
      gap: 5px;
      justify-items: end;
    }}
    .analysis-search {{
      width: min(320px, 100%);
      min-height: 42px;
      padding: 0 12px;
      border: 1px solid rgba(17,17,17,0.12);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    .watchlist-layout {{
      display: grid;
      grid-template-columns: minmax(0, 0.9fr) minmax(380px, 1.1fr);
      gap: 18px;
      align-items: start;
    }}
    .watchlist-form {{
      display: grid;
      gap: 12px;
    }}
    .watchlist-form label {{
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 0.86rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .watchlist-input {{
      width: 100%;
      min-height: 52px;
      padding: 0 14px;
      border: 1px solid rgba(17,17,17,0.12);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 1.05rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .watchlist-submit,
    .watchlist-remove {{
      appearance: none;
      min-height: 44px;
      border: 0;
      border-radius: 8px;
      padding: 0 15px;
      background: rgba(17,17,17,0.92);
      color: #fff;
      cursor: pointer;
      font: inherit;
      font-weight: 800;
    }}
    .watchlist-remove {{
      background: #8f1111;
    }}
    .watchlist-message {{
      padding: 12px 14px;
      border-radius: 8px;
      background: rgba(31,157,85,0.1);
      border: 1px solid rgba(31,157,85,0.22);
      color: #17633b;
      font-weight: 700;
    }}
    .watchlist-message.error {{
      background: rgba(157,17,17,0.08);
      border-color: rgba(157,17,17,0.2);
      color: #8f1111;
    }}
    .watchlist-list {{
      display: grid;
      gap: 10px;
    }}
    .watchlist-item {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid rgba(17,17,17,0.08);
      background: rgba(255,255,255,0.68);
    }}
    .watchlist-item div {{
      min-width: 0;
      display: grid;
      gap: 5px;
    }}
    .watchlist-item strong {{
      letter-spacing: 0.06em;
    }}
    .watchlist-item span {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .search-layout {{
      display: grid;
      grid-template-columns: minmax(300px, 0.8fr) minmax(0, 1.2fr);
      gap: 18px;
      align-items: start;
    }}
    .search-box {{
      display: grid;
      gap: 14px;
    }}
    .search-input {{
      width: 100%;
      min-height: 58px;
      padding: 0 15px;
      border: 2px solid rgba(17,17,17,0.14);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 1.2rem;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .search-counter {{
      color: var(--muted);
      font-size: 0.92rem;
      font-weight: 700;
    }}
    .search-plate-list {{
      display: grid;
      gap: 9px;
      max-height: 520px;
      overflow: auto;
      padding-right: 4px;
    }}
    .search-plate-button {{
      appearance: none;
      width: 100%;
      min-height: 58px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      border-radius: 8px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.7);
      color: var(--ink);
      cursor: pointer;
      font: inherit;
      text-align: left;
    }}
    .search-plate-button.normal {{
      border: 2px solid #1f9d55;
    }}
    .search-plate-button.danger {{
      border: 2px solid #8f1111;
    }}
    .search-plate-button.active {{
      background: rgba(17,17,17,0.92);
      color: #fff;
      border-color: rgba(17,17,17,0.92);
    }}
    .search-plate-button strong {{
      letter-spacing: 0.06em;
    }}
    .search-plate-button span {{
      color: inherit;
      font-size: 0.86rem;
      opacity: 0.72;
    }}
    .search-results {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 12px;
    }}
    .search-card {{
      overflow: hidden;
      border-radius: 8px;
      background: rgba(255,255,255,0.78);
      box-shadow: 0 12px 24px rgba(17,17,17,0.08);
    }}
    .search-card.normal {{
      border: 4px solid #1f9d55;
    }}
    .search-card.danger {{
      border: 4px solid #8f1111;
    }}
    .search-thumb {{
      aspect-ratio: 16 / 10;
      display: grid;
      place-items: center;
      background: #111;
      color: rgba(255,255,255,0.62);
      font-weight: 800;
    }}
    .search-thumb img {{
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
    }}
    .search-card-copy {{
      display: grid;
      gap: 7px;
      padding: 12px;
    }}
    .search-card-copy div {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }}
    .search-card-copy strong {{
      letter-spacing: 0.06em;
    }}
    .search-card-copy span,
    .search-card-copy small {{
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .search-state {{
      min-height: 26px;
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 8px;
      color: #fff !important;
      font-size: 0.76rem;
      font-weight: 900;
    }}
    .search-state.normal {{
      background: #1f9d55;
    }}
    .search-state.danger {{
      background: #8f1111;
    }}
    .empty-state {{
      padding: 16px;
      border-radius: 8px;
      background: rgba(17,17,17,0.04);
      color: var(--muted);
    }}
    .page {{
      min-height: 100vh;
      padding: 34px 20px 86px;
    }}
    .main {{
      max-width: 1380px;
      display: grid;
      gap: 18px;
      margin: 0 auto;
    }}
    .hero {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: 4rem;
      line-height: 0.96;
      letter-spacing: 0;
    }}
    .hero p {{
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 560px;
    }}
    .grid {{
      display: grid;
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}
    .media-shell {{
      display: grid;
      gap: 18px;
    }}
    .media-actions {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .source-tab {{
      appearance: none;
      border: 0;
      border-radius: 18px;
      padding: 15px 22px;
      background: rgba(17,17,17,0.92);
      color: #fff;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
    }}
    .profile-switch {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-left: auto;
    }}
    .profile-tab {{
      appearance: none;
      min-height: 48px;
      border: 1px solid rgba(17,17,17,0.12);
      border-radius: 8px;
      padding: 9px 13px;
      background: rgba(17,17,17,0.06);
      color: var(--ink);
      cursor: pointer;
      font: inherit;
      font-weight: 800;
    }}
    .profile-tab.active {{
      background: #1f9d55;
      color: #fff;
      border-color: transparent;
    }}
    .profile-tab[data-ai-profile-value="gpu_heavy"].active {{
      background: #8f1111;
    }}
    .upload-form {{
      display: none;
    }}
    .file-input {{
      display: none;
    }}
    .result-panel {{
      min-height: 620px;
      padding: 0;
      overflow: hidden;
      background: #000;
      border-color: rgba(255,255,255,0.04);
      box-shadow: 0 18px 40px rgba(0,0,0,0.18);
    }}
    .preview-media {{
      width: 100%;
      height: 100%;
      min-height: 620px;
      border-radius: 18px;
      display: block;
      background: #000;
      object-fit: contain;
    }}
    .preview-stack {{
      position: relative;
      width: 100%;
      min-height: 620px;
      background: #000;
    }}
    .preview-layer {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      min-height: 620px;
      border-radius: 18px;
      background: #000;
      object-fit: contain;
    }}
    .analysis-overlay {{
      position: absolute;
      left: 18px;
      bottom: 18px;
      z-index: 3;
      max-width: min(520px, calc(100% - 36px));
      display: grid;
      gap: 5px;
      padding: 12px 14px;
      border-radius: 8px;
      background: rgba(0,0,0,0.72);
      color: #fff;
      border: 1px solid rgba(255,255,255,0.12);
    }}
    .analysis-overlay strong {{
      font-size: 0.95rem;
    }}
    .analysis-overlay span {{
      color: rgba(255,255,255,0.68);
      font-size: 0.86rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .file-analysis-frame {{
      z-index: 2;
      pointer-events: none;
    }}
    .file-analysis-frame:not(.ready) {{
      display: none;
    }}
    #rawPreview {{
      display: none;
    }}
    #rawPreview.ready {{
      display: block;
    }}
    #livePreview {{
      display: none;
    }}
    #livePreview.ready {{
      display: block;
    }}
    #livePlaceholder.hidden {{
      display: none;
    }}
    .result-empty {{
      width: 100%;
      min-height: 620px;
      background: #000;
      display: grid;
      place-items: center;
      color: rgba(255,255,255,0.7);
    }}
    .preview-message {{
      display: grid;
      gap: 8px;
      justify-items: center;
      padding: 20px;
      text-align: center;
    }}
    .preview-message strong {{
      color: #fff;
      font-size: 1.05rem;
    }}
    .preview-message span {{
      color: rgba(255,255,255,0.62);
      max-width: 520px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .preview-controls {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .filter-tab {{
      appearance: none;
      border: 1px solid rgba(17,17,17,0.12);
      border-radius: 14px;
      padding: 11px 16px;
      background: rgba(17,17,17,0.06);
      color: var(--ink);
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
    }}
    .filter-tab.active {{
      background: rgba(17,17,17,0.92);
      color: #fff;
      border-color: transparent;
    }}
    @media (max-width: 980px) {{
      .page {{
        padding: 22px 14px 150px;
      }}
      .result-panel,
      .preview-media,
      .preview-stack,
      .preview-layer,
      .result-empty {{
        min-height: 460px;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
      .split-grid {{
        grid-template-columns: 1fr;
      }}
      .metric-grid,
      .analysis-columns,
      .search-layout,
      .watchlist-layout {{
        grid-template-columns: 1fr;
      }}
      .panel-head {{
        align-items: start;
        flex-direction: column;
      }}
      .analysis-event {{
        grid-template-columns: 70px minmax(0, 1fr);
      }}
      .event-meta {{
        grid-column: 2;
        justify-items: start;
      }}
      body {{
        overflow-x: hidden;
      }}
      .taskbar {{
        left: 0;
        right: 0;
        bottom: 0;
        align-items: stretch;
        flex-direction: column;
        gap: 7px;
        padding: 8px;
      }}
      .serial-monitor {{
        right: 8px;
        bottom: 150px;
      }}
      .task-left {{
        justify-content: space-between;
      }}
      .task-nav {{
        flex: 1;
      }}
      .task-link {{
        flex: 1;
        min-width: 0;
        padding: 0 10px;
      }}
      .task-status {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .status-chip {{
        min-width: 0;
      }}
      .status-chip.process .status-copy,
      .process-path {{
        max-width: none;
        min-width: 0;
      }}
      .hero h1 {{
        font-size: 2.5rem;
      }}
    }}
  </style>
</head>
<body data-filter-mode="{escape(active_filter)}" data-ai-profile="{escape(profile['id'])}" data-ai-profiles="{profile_json}">
  <div class="page">
    <main class="main">
      {content}
    </main>
  </div>
  <aside class="serial-monitor" aria-live="polite">
    <div class="serial-head">
      <span>Monitor</span>
      <span class="monitor-state" title="Zielona kropka oznacza aktywny monitor aplikacji">
        aktywny
        <i class="serial-dot"></i>
      </span>
    </div>
    <div class="serial-line" id="serialLine">&gt; gotowy</div>
  </aside>
  <footer class="taskbar">
    <div class="task-left">
      <div class="task-brand">{APP_TITLE}</div>
      <nav class="task-nav">{_render_nav(active_view)}</nav>
    </div>
    <div class="task-status" aria-label="Status systemu">
      <div class="status-chip warning" id="gpuStatus">
        <span class="status-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="6" width="14" height="11" rx="2"></rect>
            <path d="M17 10h3a1 1 0 0 1 1 1v3"></path>
            <path d="M21 8v8"></path>
            <circle cx="8" cy="11.5" r="2"></circle>
            <circle cx="13" cy="11.5" r="2"></circle>
            <path d="M6 20h8"></path>
            <path d="M7 17v3"></path>
            <path d="M11 17v3"></path>
            <path d="M15 17v3"></path>
          </svg>
        </span>
        <span class="status-copy"><span>Grafika</span><strong>...</strong></span>
      </div>
      <div class="status-chip" id="cpuStatus">
        <span class="status-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="6" y="6" width="12" height="12" rx="2"></rect>
            <rect x="10" y="10" width="4" height="4" rx="1"></rect>
            <path d="M9 1v3"></path>
            <path d="M12 1v3"></path>
            <path d="M15 1v3"></path>
            <path d="M9 20v3"></path>
            <path d="M12 20v3"></path>
            <path d="M15 20v3"></path>
            <path d="M1 9h3"></path>
            <path d="M1 12h3"></path>
            <path d="M1 15h3"></path>
            <path d="M20 9h3"></path>
            <path d="M20 12h3"></path>
            <path d="M20 15h3"></path>
          </svg>
        </span>
        <span class="status-copy"><span>Procesor</span><strong>...</strong></span>
      </div>
      <div class="status-chip" id="ramStatus">
        <span class="status-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="7" width="18" height="9" rx="2"></rect>
            <path d="M6 11h2"></path>
            <path d="M10 11h2"></path>
            <path d="M14 11h2"></path>
            <path d="M18 11h1"></path>
            <path d="M6 16v3"></path>
            <path d="M9 16v3"></path>
            <path d="M12 16v3"></path>
            <path d="M15 16v3"></path>
            <path d="M18 16v3"></path>
          </svg>
        </span>
        <span class="status-copy"><span>Pamiec</span><strong>...</strong></span>
      </div>
      <div class="status-chip process" id="processStatus">
        <span class="status-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="4" width="18" height="16" rx="2"></rect>
            <path d="M7 8h10"></path>
            <path d="M7 12h4"></path>
            <path d="M7 16h7"></path>
          </svg>
        </span>
        <span class="status-copy">
          <span>Proces Python</span>
          <strong>PID ...</strong>
          <code class="process-path" id="processPath">...</code>
        </span>
      </div>
      <div class="status-chip" id="aiStatus">
        <span class="status-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 3v3"></path>
            <path d="M12 18v3"></path>
            <path d="M3 12h3"></path>
            <path d="M18 12h3"></path>
            <circle cx="12" cy="12" r="5"></circle>
            <circle cx="12" cy="12" r="1"></circle>
          </svg>
        </span>
        <span class="status-copy"><span>Tryb</span><strong>{escape(str(profile["label"]).upper())}</strong></span>
      </div>
    </div>
  </footer>
  <script>
    const body = document.body;
    const fileButton = document.getElementById("fileButton");
    const fileInput = document.getElementById("fileInput");
    const filterInput = document.getElementById("filterModeInput");
    const aiProfileInput = document.getElementById("aiProfileInput");
    const profileButtons = Array.from(document.querySelectorAll("[data-ai-profile-value]"));
    const filterButtons = Array.from(document.querySelectorAll("[data-filter-mode-value]"));
    const screenButton = document.getElementById("screenButton");
    const rawPreview = document.getElementById("rawPreview");
    const previewNode = document.getElementById("livePreview");
    const livePlaceholder = document.getElementById("livePlaceholder");
    let screenStream = null;
    let previewVideo = null;
    let frameCanvas = null;
    let frameLoop = null;
    const liveSessionId = crypto?.randomUUID ? crypto.randomUUID() : String(Date.now());
    let liveStartedAt = 0;
    let liveBusy = false;
    let serialTimer = null;
    let activePreviewUrl = null;
    let filePreviewMedia = null;
    let fileAnalysisImage = null;
    let fileFrameCanvas = null;
    let fileFrameLoop = null;
    let fileFrameBusy = false;
    let fileSourceName = "Plik";
    let fileAnalysisSessionId = "";
    let currentAiProfile = body.dataset.aiProfile || "cpu_lite";
    let profileConfig = {{}};
    try {{
      profileConfig = JSON.parse(body.dataset.aiProfiles || "{{}}");
    }} catch (_error) {{}}
    let currentFilterModes = new Set(
      (body.dataset.filterMode || "normal")
        .split(",")
        .map((value) => value.trim().toLowerCase())
        .filter((value) => value && value !== "normal")
    );

    const serialLine = document.getElementById("serialLine");
    const getCurrentProfileConfig = () => profileConfig[currentAiProfile] || profileConfig.cpu_lite || {{}};
    const getLiveInterval = () => Number(getCurrentProfileConfig().live_interval_ms) || 1000;
    const getJpegQuality = () => Number(getCurrentProfileConfig().live_jpeg_quality) || 0.72;
    const getLiveMaxWidth = () => Number(getCurrentProfileConfig().live_max_width) || 960;
    const shouldRunFullFileAnalysis = () => getCurrentProfileConfig().full_file_analysis !== false;
    const serialWrite = (message) => {{
      if (!serialLine) return;
      serialLine.textContent = `> ${{message}}`;
    }};

    const serialSequence = (messages, delay = 900) => {{
      if (serialTimer) {{
        clearInterval(serialTimer);
        serialTimer = null;
      }}
      let index = 0;
      serialWrite(messages[index] || "gotowy");
      serialTimer = setInterval(() => {{
        index += 1;
        if (index >= messages.length) {{
          clearInterval(serialTimer);
          serialTimer = null;
          return;
        }}
        serialWrite(messages[index]);
      }}, delay);
    }};

    const htmlEscape = (value) => String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

    const setPreviewMessage = (title, detail = "") => {{
      const resultPanel = document.querySelector(".result-panel");
      if (!resultPanel) return;
      resultPanel.innerHTML = `
        <div class="result-empty">
          <div class="preview-message">
            <strong>${{htmlEscape(title)}}</strong>
            <span>${{htmlEscape(detail)}}</span>
          </div>
        </div>
      `;
    }};

    const drawScaledFrame = (media, canvas, sourceWidth, sourceHeight) => {{
      const maxWidth = getLiveMaxWidth();
      const scale = sourceWidth > maxWidth ? maxWidth / sourceWidth : 1;
      const targetWidth = Math.max(1, Math.round(sourceWidth * scale));
      const targetHeight = Math.max(1, Math.round(sourceHeight * scale));
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      const context = canvas.getContext("2d");
      if (!context) return false;
      context.drawImage(media, 0, 0, targetWidth, targetHeight);
      return true;
    }};

    const stopFileFrameLoop = () => {{
      if (fileFrameLoop) {{
        clearInterval(fileFrameLoop);
        fileFrameLoop = null;
      }}
      filePreviewMedia = null;
      fileAnalysisImage = null;
      fileFrameCanvas = null;
      fileFrameBusy = false;
      fileAnalysisSessionId = "";
    }};

    const sendFileFrame = async () => {{
      if (!filePreviewMedia || !fileFrameCanvas || !fileAnalysisImage || fileFrameBusy) return;
      const isVideo = filePreviewMedia.tagName === "VIDEO";
      if (isVideo && (filePreviewMedia.readyState < 2 || filePreviewMedia.paused)) return;
      const width = isVideo ? filePreviewMedia.videoWidth : filePreviewMedia.naturalWidth;
      const height = isVideo ? filePreviewMedia.videoHeight : filePreviewMedia.naturalHeight;
      if (!width || !height) return;

      if (!drawScaledFrame(filePreviewMedia, fileFrameCanvas, width, height)) return;
      const blob = await new Promise((resolve) => fileFrameCanvas.toBlob(resolve, "image/jpeg", getJpegQuality()));
      if (!blob) return;

      fileFrameBusy = true;
      try {{
        const data = new FormData();
        data.append("frame", new File([blob], "file-frame.jpg", {{ type: "image/jpeg" }}));
        data.append("source_name", fileSourceName);
        data.append("seconds", String(isVideo ? filePreviewMedia.currentTime : 0));
        data.append("session_id", fileAnalysisSessionId || liveSessionId);
        data.append("filter_mode", serializeFilterModes());
        data.append("ai_profile", currentAiProfile);
        const response = await fetch("/analyze-live-frame", {{ method: "POST", body: data }});
        if (!response.ok) return;
        const payload = await response.json();
        if (payload.image) {{
          fileAnalysisImage.src = payload.image;
          fileAnalysisImage.classList.add("ready");
          if (!fileAnalysisImage.dataset.firstFrame) {{
            fileAnalysisImage.dataset.firstFrame = "1";
            serialWrite("podglad live: odebrano klatke z boxami");
          }}
        }}
      }} catch (_error) {{
      }} finally {{
        fileFrameBusy = false;
      }}
    }};

    const startFileFrameAnalysis = (media, file, analyzedImage) => {{
      stopFileFrameLoop();
      filePreviewMedia = media;
      fileAnalysisImage = analyzedImage;
      fileFrameCanvas = document.createElement("canvas");
      fileSourceName = file.name;
      fileAnalysisSessionId = crypto?.randomUUID ? crypto.randomUUID() : `file-${{Date.now()}}`;

      if (media.tagName === "IMG") {{
        media.addEventListener("load", sendFileFrame, {{ once: true }});
        return;
      }}

      media.addEventListener("loadeddata", () => {{
        sendFileFrame();
        fileFrameLoop = setInterval(sendFileFrame, getLiveInterval());
      }}, {{ once: true }});
      media.addEventListener("ended", () => {{
        serialWrite("podglad live: koniec odtwarzania pliku");
      }});
    }};

    const setLocalFilePreview = (file) => {{
      const resultPanel = document.querySelector(".result-panel");
      if (!resultPanel) return;
      stopFileFrameLoop();
      if (activePreviewUrl) {{
        URL.revokeObjectURL(activePreviewUrl);
      }}
      activePreviewUrl = URL.createObjectURL(file);

      const stack = document.createElement("div");
      stack.className = "preview-stack";

      const media = file.type.startsWith("image/") ? document.createElement("img") : document.createElement("video");
      media.className = "preview-layer ready";
      media.src = activePreviewUrl;
      media.style.display = "block";
      if (media.tagName === "VIDEO") {{
        media.controls = true;
        media.autoplay = true;
        media.muted = true;
        media.loop = true;
        media.playsInline = true;
      }} else {{
        media.alt = file.name;
      }}

      const analyzedImage = document.createElement("img");
      analyzedImage.className = "preview-layer file-analysis-frame";
      analyzedImage.alt = "Analizowana klatka z ramkami";

      const overlay = document.createElement("div");
      overlay.className = "analysis-overlay";
      const title = document.createElement("strong");
      title.textContent = "Podglad z analiza live";
      const detail = document.createElement("span");
      detail.textContent = "Klatki sa wysylane do backendu i nakladane z boxami.";
      overlay.append(title, detail);

      stack.append(media, analyzedImage, overlay);
      resultPanel.replaceChildren(stack);
      startFileFrameAnalysis(media, file, analyzedImage);
      media.play?.().catch(() => {{}});
    }};

    const setStatus = (id, value, state = "") => {{
      const node = document.getElementById(id);
      const strong = node?.querySelector("strong");
      if (strong) {{
        strong.textContent = value;
      }}
      if (node) {{
        node.classList.remove("online", "warning", "offline");
        if (state) {{
          node.classList.add(state);
        }}
      }}
    }};

    const setProcessPath = (value) => {{
      const node = document.getElementById("processPath");
      if (!node) return;
      node.textContent = value || "brak sciezki";
      node.title = value || "brak sciezki";
    }};

    const updateSystemStatus = async () => {{
      try {{
        const response = await fetch("/system-status", {{ cache: "no-store" }});
        if (!response.ok) return;
        const status = await response.json();
        const cpu = Number.isFinite(status.cpu) ? `${{status.cpu}}%` : "...";
        const ram = Number.isFinite(status.ram) ? `${{status.ram}}%` : "...";
        const cpuState = Number.isFinite(status.cpu) ? (status.cpu >= 85 ? "warning" : "online") : "warning";
        const ramState = Number.isFinite(status.ram) ? (status.ram >= 85 ? "warning" : "online") : "warning";
        setStatus("cpuStatus", cpu, cpuState);
        setStatus("ramStatus", ram, ramState);
        const processCpu = Number.isFinite(status.process?.cpu) ? `${{status.process.cpu}}%` : "...";
        const processPid = status.process?.pid ? `PID ${{status.process.pid}}` : "PID ?";
        const processState = Number.isFinite(status.process?.cpu) && status.process.cpu >= 85 ? "warning" : "online";
        setStatus("processStatus", `${{processPid}} ${{processCpu}}`, processState);
        setProcessPath(status.process?.path || "");

        if (status.gpu?.available) {{
          const gpuValue = Number.isFinite(status.gpu.utilization)
            ? `${{status.gpu.utilization}}%`
            : "ON";
          setStatus("gpuStatus", gpuValue, "online");
        }} else {{
          setStatus("gpuStatus", "OFF", "offline");
        }}
      }} catch (_error) {{}}
    }};

    updateSystemStatus();
    setInterval(updateSystemStatus, 2500);

    const syncProfileUi = () => {{
      if (aiProfileInput) {{
        aiProfileInput.value = currentAiProfile;
      }}
      const profile = getCurrentProfileConfig();
      setStatus("aiStatus", String(profile.label || currentAiProfile).toUpperCase(), currentAiProfile === "gpu_heavy" ? "warning" : "online");
      profileButtons.forEach((button) => {{
        button.classList.toggle("active", button.dataset.aiProfileValue === currentAiProfile);
      }});
    }};

    profileButtons.forEach((button) => {{
      button.addEventListener("click", () => {{
        currentAiProfile = button.dataset.aiProfileValue || "cpu_lite";
        body.dataset.aiProfile = currentAiProfile;
        syncProfileUi();
        const profile = getCurrentProfileConfig();
        serialWrite(`profil: ${{profile.label || currentAiProfile}}`);
        if (fileFrameLoop && filePreviewMedia?.tagName === "VIDEO") {{
          clearInterval(fileFrameLoop);
          fileFrameLoop = setInterval(sendFileFrame, getLiveInterval());
        }}
      }});
    }});

    syncProfileUi();

    const analysisSearch = document.getElementById("analysisSearch");
    analysisSearch?.addEventListener("input", () => {{
      const query = analysisSearch.value.trim().toUpperCase().replace(/[^A-Z0-9]/g, "");
      document.querySelectorAll("[data-plate]").forEach((row) => {{
        const plate = row.getAttribute("data-plate") || "";
        row.hidden = Boolean(query) && !plate.includes(query);
      }});
    }});

    const plateSearchInput = document.getElementById("plateSearchInput");
    const searchCards = Array.from(document.querySelectorAll("[data-search-card]"));
    const searchButtons = Array.from(document.querySelectorAll("[data-search-plate]"))
      .filter((node) => node.tagName === "BUTTON");
    const searchCount = document.getElementById("searchCount");
    const searchEmpty = document.getElementById("searchEmpty");
    const normalizePlateInput = (value) => String(value || "").toUpperCase().replace(/[^A-Z0-9]/g, "");

    const applyPlateSearch = (exactPlate = "") => {{
      if (!plateSearchInput) return;
      const query = normalizePlateInput(plateSearchInput.value);
      const exact = normalizePlateInput(exactPlate);
      const activeQuery = exact || query;
      let visibleCards = 0;
      let visibleButtons = 0;

      searchCards.forEach((card) => {{
        const plate = card.dataset.searchPlate || "";
        const isVisible = Boolean(activeQuery) && (exact ? plate === exact : plate.includes(activeQuery));
        card.hidden = !isVisible;
        if (isVisible) visibleCards += 1;
      }});

      searchButtons.forEach((button) => {{
        const plate = button.dataset.searchPlate || "";
        const isVisible = Boolean(query) && plate.includes(query);
        button.hidden = !isVisible;
        button.classList.toggle("active", Boolean(activeQuery) && plate === activeQuery);
        if (isVisible) visibleButtons += 1;
      }});

      if (searchCount) {{
        if (!activeQuery) {{
          searchCount.textContent = "Wpisz pierwsza litere rejestracji";
        }} else {{
          searchCount.textContent = `${{visibleCards}} zdjec | ${{visibleButtons}} pasujacych tablic`;
        }}
      }}
      if (searchEmpty) {{
        searchEmpty.hidden = !activeQuery || visibleCards > 0;
      }}
    }};

    plateSearchInput?.addEventListener("input", () => {{
      plateSearchInput.value = normalizePlateInput(plateSearchInput.value);
      applyPlateSearch();
    }});

    searchButtons.forEach((button) => {{
      button.addEventListener("click", () => {{
        const plate = button.dataset.searchPlate || "";
        if (!plateSearchInput || !plate) return;
        plateSearchInput.value = plate;
        applyPlateSearch(plate);
        serialWrite(`szukam dokladnie: ${{plate}}`);
      }});
    }});

    applyPlateSearch();

    const serializeFilterModes = () => {{
      if (!currentFilterModes.size) {{
        return "normal";
      }}
      return Array.from(currentFilterModes).sort().join(",");
    }};

    const syncFilterUi = () => {{
      if (filterInput) {{
        filterInput.value = serializeFilterModes();
      }}
      filterButtons.forEach((button) => {{
        const mode = button.dataset.filterModeValue || "";
        const isActive = mode === "normal" ? currentFilterModes.size === 0 : currentFilterModes.has(mode);
        button.classList.toggle("active", isActive);
      }});
    }};

    syncFilterUi();

    filterButtons.forEach((button) => {{
      button.addEventListener("click", () => {{
        const mode = button.dataset.filterModeValue || "normal";
        if (mode === "normal") {{
          currentFilterModes.clear();
        }} else if (currentFilterModes.has(mode)) {{
          currentFilterModes.delete(mode);
        }} else {{
          currentFilterModes.add(mode);
        }}
        syncFilterUi();
      }});
    }});

    const stopTracks = () => {{
      if (!screenStream) return;
      for (const track of screenStream.getTracks()) {{
        track.stop();
      }}
      screenStream = null;
    }};

    const stopLiveLoop = () => {{
      if (frameLoop) {{
        clearInterval(frameLoop);
        frameLoop = null;
      }}
      if (previewVideo) {{
        previewVideo.pause();
        previewVideo.srcObject = null;
        previewVideo = null;
      }}
      if (rawPreview) {{
        rawPreview.pause?.();
        rawPreview.srcObject = null;
        rawPreview.classList.remove("ready");
      }}
      if (previewNode) {{
        previewNode.src = "";
        previewNode.classList.remove("ready");
      }}
      if (livePlaceholder) {{
        livePlaceholder.classList.remove("hidden");
      }}
    }};

    const sendLiveFrame = async () => {{
      if (!previewVideo || !frameCanvas || liveBusy) return;
      if (previewVideo.readyState < 2) return;
      const width = previewVideo.videoWidth;
      const height = previewVideo.videoHeight;
      if (!width || !height) return;
      if (!drawScaledFrame(previewVideo, frameCanvas, width, height)) return;
      const blob = await new Promise((resolve) => frameCanvas.toBlob(resolve, "image/jpeg", getJpegQuality()));
      if (!blob) return;
      liveBusy = true;
      try {{
        const data = new FormData();
        data.append("frame", new File([blob], "frame.jpg", {{ type: "image/jpeg" }}));
        data.append("source_name", "Ekran");
        data.append("seconds", String((Date.now() - liveStartedAt) / 1000));
        data.append("session_id", liveSessionId);
        data.append("filter_mode", serializeFilterModes());
        data.append("ai_profile", currentAiProfile);
        const response = await fetch("/analyze-live-frame", {{ method: "POST", body: data }});
        if (!response.ok) return;
        const payload = await response.json();
        if (previewNode && payload.image) {{
          previewNode.src = payload.image;
          previewNode.classList.add("ready");
        }}
        if (livePlaceholder) {{
          livePlaceholder.classList.add("hidden");
        }}
      }} catch (_error) {{
      }} finally {{
        liveBusy = false;
      }}
    }};

    fileButton?.addEventListener("click", () => {{
      fileInput?.click();
    }});

    fileInput?.addEventListener("change", async () => {{
      const file = fileInput?.files?.[0];
      const form = fileInput?.form;
      if (!file || !form) return;

      stopLiveLoop();
      stopTracks();
      serialSequence([
        `wybrano plik: ${{file.name}}`,
        "pokazuje lokalny podglad pliku",
        `profil: ${{getCurrentProfileConfig().label || currentAiProfile}}`,
        "wysylam lekkie klatki do analizy live",
      ]);
      setLocalFilePreview(file);

      if (!shouldRunFullFileAnalysis() && file.type.startsWith("video/")) {{
        serialWrite("CPU Lite: pelny render filmu wylaczony, zeby nie zacinac odtwarzania");
        fileInput.value = "";
        return;
      }}

      const data = new FormData(form);
      data.set("filter_mode", serializeFilterModes());
      data.set("ai_profile", currentAiProfile);
      try {{
        const response = await fetch(form.action, {{
          method: "POST",
          body: data,
        }});
        if (!response.ok) {{
          serialWrite(`blad HTTP ${{response.status}} podczas analizy`);
          setPreviewMessage("Blad analizy pliku", `HTTP ${{response.status}}`);
          return;
        }}

        const html = await response.text();
        const doc = new DOMParser().parseFromString(html, "text/html");
        const nextResult = doc.querySelector(".result-panel");
        const currentResult = document.querySelector(".result-panel");
        if (nextResult && currentResult) {{
          stopFileFrameLoop();
          if (activePreviewUrl) {{
            URL.revokeObjectURL(activePreviewUrl);
            activePreviewUrl = null;
          }}
          currentResult.replaceWith(nextResult);
          serialWrite("gotowe: wynik analizy wyswietlony");
        }} else {{
          serialWrite("gotowe, ale nie znaleziono panelu wyniku");
          setPreviewMessage("Analiza zakonczona", "Nie znaleziono panelu wyniku w odpowiedzi");
        }}
      }} catch (_error) {{
        serialWrite("blad: nie udalo sie wyslac albo odebrac wyniku");
        setPreviewMessage("Blad polaczenia", "Nie udalo sie wyslac pliku do backendu");
      }} finally {{
        fileInput.value = "";
      }}
    }});

    screenButton?.addEventListener("click", async () => {{
      if (!navigator.mediaDevices?.getDisplayMedia) {{
        serialWrite("blad: przegladarka nie obsluguje przechwytywania ekranu");
        return;
      }}
      try {{
        serialSequence([
          "prosze system o dostep do ekranu",
          "czekam na wybor okna lub monitora",
          "uruchamiam strumien wideo",
          "wysylam klatki do /analyze-live-frame",
        ]);
        stopFileFrameLoop();
        stopLiveLoop();
        screenStream = await navigator.mediaDevices.getDisplayMedia({{ video: true, audio: false }});
        previewVideo = document.createElement("video");
        previewVideo.autoplay = true;
        previewVideo.muted = true;
        previewVideo.playsInline = true;
        previewVideo.srcObject = screenStream;
        if (rawPreview) {{
          rawPreview.srcObject = screenStream;
          rawPreview.classList.add("ready");
          await rawPreview.play();
        }}
        if (previewNode) {{
          previewNode.srcObject = null;
          previewNode.src = "";
          previewNode.classList.remove("ready");
        }}
        if (livePlaceholder) {{
          livePlaceholder.classList.add("hidden");
        }}
        frameCanvas = document.createElement("canvas");
        liveStartedAt = Date.now();
        screenStream.getVideoTracks()[0]?.addEventListener("ended", () => {{
          stopLiveLoop();
          stopTracks();
        }});
        await previewVideo.play();
        frameLoop = setInterval(sendLiveFrame, getLiveInterval());
        await sendLiveFrame();
        serialWrite("tryb ekranu aktywny: analizuje klatki");
      }} catch (_error) {{
        serialWrite("anulowano albo nie udalo sie przechwycic ekranu");
      }}
    }});
  </script>
</body>
</html>"""
    )


def _render_profile_buttons(active_profile: str) -> str:
    active = get_ai_profile(active_profile)
    chunks: list[str] = []
    for profile_id, profile in AI_PROFILES.items():
        class_name = "profile-tab active" if profile_id == active["id"] else "profile-tab"
        chunks.append(
            f'<button class="{class_name}" data-ai-profile-value="{escape(profile_id)}" type="button" title="{escape(str(profile["description"]))}">{escape(str(profile["label"]))}</button>'
        )
    return "".join(chunks)


def _media_page(
    result: str = "",
    active_filter: str = "normal",
    active_profile: str = DEFAULT_AI_PROFILE,
) -> HTMLResponse:
    content = f"""
<section class="grid">
  <section class="panel media-shell">
    <div class="media-actions">
      <button class="source-tab" id="fileButton" type="button">Pliki</button>
      <button class="source-tab" id="screenButton" type="button">Ekran</button>
      <div class="profile-switch">{_render_profile_buttons(active_profile)}</div>
    </div>
    <form class="upload-form" action="/analyze" method="post" enctype="multipart/form-data">
      <input id="filterModeInput" type="hidden" name="filter_mode" value="{escape(active_filter)}">
      <input id="aiProfileInput" type="hidden" name="ai_profile" value="{escape(get_ai_profile(active_profile)["id"])}">
      <input class="file-input" id="fileInput" type="file" name="media_file" accept="video/*,image/*">
    </form>
    {result}
  </section>
</section>
"""
    return _page(content, "media", active_filter, active_profile)


def _analysis_page() -> HTMLResponse:
    history = _read_history()
    analysis = _build_analysis(history)
    regions, dangerous = _build_ranking(history)
    content = f"""
<section class="hero">
  <div>
    <h1>Analiza</h1>
    <p>Podsumowanie wykrytych tablic, powtorzen, regionow i trafien z watchlisty.</p>
  </div>
</section>
<section class="analysis-layout">
  <div class="metric-grid">{_render_stat_cards(analysis)}</div>
  <section class="analysis-columns">
    <section class="panel">
      <div class="panel-head">
        <div>
          <p>Tablice</p>
          <h2>Najczesciej wykrywane</h2>
        </div>
      </div>
      <div class="plate-list">{_render_plate_frequency(analysis["plate_counts"], analysis["dangerous_plates"])}</div>
    </section>
    <section class="panel">
      <div class="panel-head">
        <div>
          <p>Watchlista</p>
          <h2>Podejrzane trafienia</h2>
        </div>
      </div>
      <div class="danger-list">{_render_dangerous_ranking(dangerous)}</div>
    </section>
  </section>
  <section class="analysis-columns">
    <section class="panel">
      <div class="panel-head">
        <div>
          <p>Historia</p>
          <h2>Ostatnie wykrycia</h2>
        </div>
        <input class="analysis-search" id="analysisSearch" type="search" placeholder="Szukaj tablicy">
      </div>
      <div class="analysis-events">{_render_analysis_history(history, analysis["dangerous_plates"])}</div>
    </section>
    <section class="panel">
      <div class="panel-head">
        <div>
          <p>Regiony</p>
          <h2>Kody rejestracji</h2>
        </div>
      </div>
      <div class="rank-list">{_render_region_ranking(regions)}</div>
    </section>
  </section>
</section>
"""
    return _page(content, "analysis", "normal")


def _watchlist_page(message: str = "", error: str = "") -> HTMLResponse:
    plates = _read_dangerous_plates()
    notice = ""
    if error:
        notice = f'<div class="watchlist-message error">{escape(error)}</div>'
    elif message:
        notice = f'<div class="watchlist-message">{escape(message)}</div>'

    content = f"""
<section class="hero">
  <div>
    <h1>Watchlista</h1>
    <p>Czarna lista rejestracji uzywana przy oznaczaniu trafien w analizie.</p>
  </div>
</section>
<section class="watchlist-layout">
  <section class="panel">
    <div class="panel-head">
      <div>
        <p>Dodawanie</p>
        <h2>Nowa rejestracja</h2>
      </div>
    </div>
    <form class="watchlist-form" action="/watchlist/add" method="post">
      <label>
        Numer tablicy
        <input class="watchlist-input" name="plate" type="text" placeholder="EBECA32" autocomplete="off" required>
      </label>
      <button class="watchlist-submit" type="submit">Dodaj do listy</button>
      {notice}
    </form>
  </section>
  <section class="panel">
    <div class="panel-head">
      <div>
        <p>{len(plates)} wpisow</p>
        <h2>Aktualna lista</h2>
      </div>
    </div>
    <div class="watchlist-list">{_render_watchlist_items(plates)}</div>
  </section>
</section>
"""
    return _page(content, "watchlist", "normal")


def _search_page() -> HTMLResponse:
    history = _read_history()
    dangerous_plates = _read_dangerous_plates()
    content = f"""
<section class="hero">
  <div>
    <h1>Szukaj</h1>
    <p>Wpisz rejestracje. Wyniki i zdjecia pojawiaja sie od pierwszej litery.</p>
  </div>
</section>
<section class="search-layout">
  <section class="panel search-box">
    <div class="panel-head">
      <div>
        <p>Rejestracja</p>
        <h2>Wyszukiwarka</h2>
      </div>
    </div>
    <input class="search-input" id="plateSearchInput" type="search" placeholder="EBECA32" autocomplete="off" autofocus>
    <div class="search-counter" id="searchCount">Wpisz pierwsza litere rejestracji</div>
    <div class="search-plate-list">{_render_search_plate_buttons(history, dangerous_plates)}</div>
  </section>
  <section class="panel">
    <div class="panel-head">
      <div>
        <p>Zdjecia</p>
        <h2>Historia trafien</h2>
      </div>
    </div>
    <div class="empty-state" id="searchEmpty" hidden>Brak zdjec dla tej rejestracji</div>
    <div class="search-results">{_render_search_results(history, dangerous_plates)}</div>
  </section>
</section>
"""
    return _page(content, "search", "normal")


def _history_page() -> HTMLResponse:
    return _analysis_page()


def _ranking_page() -> HTMLResponse:
    return _analysis_page()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return _media_page(
        '<section class="panel result-panel"><div class="preview-stack"><video class="preview-layer" id="rawPreview" autoplay muted playsinline></video><img class="preview-layer" id="livePreview" alt=""><div class="result-empty" id="livePlaceholder"><div class="preview-message"><strong>Wybierz plik albo ekran</strong><span>Monitor po prawej pokazuje aktualny etap pracy</span></div></div></div></section>'
    )


@app.get("/analysis", response_class=HTMLResponse)
def analysis_page() -> HTMLResponse:
    return _analysis_page()


@app.get("/watchlist", response_class=HTMLResponse)
def watchlist_page() -> HTMLResponse:
    return _watchlist_page()


@app.get("/search", response_class=HTMLResponse)
def search_page() -> HTMLResponse:
    return _search_page()


@app.get("/history", response_class=HTMLResponse)
def history_page() -> HTMLResponse:
    return _history_page()


@app.get("/ranking", response_class=HTMLResponse)
def ranking_page() -> HTMLResponse:
    return _ranking_page()


@app.get("/system-status")
def system_status():
    return JSONResponse(
        {
            "cpu": _cpu_percent(),
            "ram": _ram_percent(),
            "gpu": _gpu_status(),
            "process": _process_status(),
            "device": RUNTIME_DEVICE.upper(),
            "profiles": AI_PROFILES,
        }
    )


@app.post("/watchlist/add", response_class=HTMLResponse)
def add_watchlist_plate(plate: str = Form("")) -> HTMLResponse:
    cleaned = clean_plate_text(plate)
    if not cleaned:
        return _watchlist_page(error="Podaj poprawna polska rejestracje, np. EBECA32 albo WR-3804R.")

    plates = _read_dangerous_plates()
    if cleaned in plates:
        return _watchlist_page(message=f"{cleaned} jest juz na watchliscie.")

    plates.add(cleaned)
    try:
        _write_dangerous_plates(plates)
    except OSError:
        return _watchlist_page(error="Nie udalo sie zapisac listy do pliku.")
    return _watchlist_page(message=f"Dodano {cleaned} do watchlisty.")


@app.post("/watchlist/remove", response_class=HTMLResponse)
def remove_watchlist_plate(plate: str = Form("")) -> HTMLResponse:
    cleaned = clean_plate_text(plate)
    if not cleaned:
        return _watchlist_page(error="Nie rozpoznano tablicy do usuniecia.")

    plates = _read_dangerous_plates()
    if cleaned not in plates:
        return _watchlist_page(message=f"{cleaned} nie ma juz na watchliscie.")

    plates.remove(cleaned)
    try:
        _write_dangerous_plates(plates)
    except OSError:
        return _watchlist_page(error="Nie udalo sie zapisac listy do pliku.")
    return _watchlist_page(message=f"Usunieto {cleaned} z watchlisty.")


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    media_file: UploadFile = File(...),
    filter_mode: str = Form("normal"),
    ai_profile: str = Form(DEFAULT_AI_PROFILE),
) -> HTMLResponse:
    original_name = media_file.filename or "plik.bin"
    suffix = Path(original_name).suffix.lower() or ".bin"
    safe_name = f"{uuid4().hex}{suffix}"
    upload_path = UPLOADS_DIR / safe_name
    data = await media_file.read()
    upload_path.write_bytes(data)

    result = run_detection(
        upload_path,
        output_dir=RESULTS_DIR,
        preview=False,
        source_name=original_name,
        history_dir=HISTORY_THUMBS_DIR,
        filter_mode=filter_mode,
        ai_profile=ai_profile,
    )
    _append_history(result.events)
    return _media_page(
        _render_result(result, original_name),
        active_filter=filter_mode,
        active_profile=ai_profile,
    )


@app.post("/analyze-live-frame")
async def analyze_live_frame(
    frame: UploadFile = File(...),
    source_name: str = Form("Ekran"),
    seconds: float = Form(0.0),
    session_id: str = Form("default"),
    filter_mode: str = Form("normal"),
    ai_profile: str = Form(DEFAULT_AI_PROFILE),
):
    data = await frame.read()
    np_data = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return JSONResponse({"image": None}, status_code=400)

    annotated, events = analyze_frame(
        image,
        source_name=source_name,
        seconds=seconds,
        history_dir=HISTORY_THUMBS_DIR,
        filter_mode=filter_mode,
        ai_profile=ai_profile,
    )
    _append_live_history(session_id, events)
    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok:
        return JSONResponse({"image": None}, status_code=500)
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return JSONResponse({"image": "data:image/jpeg;base64," + payload})
