import base64
import json
from html import escape
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import APP_TITLE, DANGEROUS_PLATES_LIST
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


def _render_result(result: DetectionResult, original_name: str) -> str:
    if result.exit_code != 0 or result.output_path is None:
        return f"""
<section class="panel result-panel">
  <div class="result-empty"></div>
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


def _render_nav(active_view: str) -> str:
    items = [
        ("/", "Media", active_view == "media"),
        ("/history", "Historia", active_view == "history"),
        ("/ranking", "Ranking", active_view == "ranking"),
    ]
    links: list[str] = []
    for href, label, is_active in items:
        class_name = "nav-link active" if is_active else "nav-link"
        links.append(f'<a class="{class_name}" href="{href}">{label}</a>')
    return "".join(links)


def _page(content: str, active_view: str, active_filter: str) -> HTMLResponse:
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
      --sidebar-width: 250px;
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
    .menu-toggle {{
      position: fixed;
      top: 18px;
      left: 18px;
      z-index: 30;
      border: 0;
      border-radius: 999px;
      background: rgba(17,17,17,0.92);
      color: #fff;
      width: 48px;
      height: 48px;
      font-size: 1.2rem;
      cursor: pointer;
      box-shadow: var(--shadow);
    }}
    .sidebar {{
      position: fixed;
      inset: 0 auto 0 0;
      width: var(--sidebar-width);
      background: rgba(10,10,10,0.96);
      color: #f7f7f2;
      padding: 76px 18px 20px;
      overflow-y: auto;
      transform: translateX(calc(-100% + 70px));
      transition: transform 0.25s ease;
      z-index: 20;
      backdrop-filter: blur(10px);
    }}
    body.sidebar-open .sidebar {{
      transform: translateX(0);
    }}
    .sidebar h2 {{
      margin: 0 0 6px;
      font-size: 1.25rem;
    }}
    .sidebar p {{
      margin: 0 0 18px;
      color: rgba(255,255,255,0.62);
      font-size: 0.95rem;
    }}
    .nav {{
      display: grid;
      gap: 10px;
      margin-top: 24px;
    }}
    .nav-link {{
      display: block;
      padding: 14px 16px;
      border-radius: 16px;
      color: rgba(255,255,255,0.82);
      text-decoration: none;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.06);
      transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease;
    }}
    .nav-link:hover {{
      background: rgba(255,255,255,0.1);
      color: #fff;
      transform: translateX(2px);
    }}
    .nav-link.active {{
      background: #fff;
      color: #111;
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
    .page {{
      min-height: 100vh;
      padding: 34px 20px 34px 110px;
      transition: padding-left 0.25s ease;
    }}
    body.sidebar-open .page {{
      padding-left: calc(var(--sidebar-width) + 24px);
    }}
    .main {{
      max-width: 1040px;
      display: grid;
      gap: 18px;
    }}
    .hero {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: clamp(2.4rem, 6vw, 4.4rem);
      line-height: 0.96;
      letter-spacing: -0.04em;
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
    .upload-form {{
      display: none;
    }}
    .file-input {{
      display: none;
    }}
    .result-panel {{
      min-height: 420px;
      padding: 0;
      overflow: hidden;
      background: #000;
      border-color: rgba(255,255,255,0.04);
      box-shadow: 0 18px 40px rgba(0,0,0,0.18);
    }}
    .preview-media {{
      width: 100%;
      height: 100%;
      min-height: 420px;
      border-radius: 18px;
      display: block;
      background: #000;
      object-fit: contain;
    }}
    .preview-stack {{
      position: relative;
      width: 100%;
      min-height: 420px;
      background: #000;
    }}
    .preview-layer {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      min-height: 420px;
      border-radius: 18px;
      background: #000;
      object-fit: contain;
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
      min-height: 420px;
      background: #000;
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
      .page,
      body.sidebar-open .page {{
        padding: 92px 16px 24px;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
      .split-grid {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        width: min(88vw, 360px);
      }}
      body {{
        overflow-x: hidden;
      }}
    }}
  </style>
</head>
<body class="sidebar-open" data-filter-mode="{escape(active_filter)}">
  <button class="menu-toggle" id="menuToggle" type="button">≡</button>
  <aside class="sidebar" id="sidebar">
    <h2>{APP_TITLE}</h2>
    <p>Menu</p>
    <nav class="nav">{_render_nav(active_view)}</nav>
  </aside>
  <div class="page">
    <main class="main">
      {content}
    </main>
  </div>
  <script>
    const body = document.body;
    const menuToggle = document.getElementById("menuToggle");
    menuToggle?.addEventListener("click", () => {{
      body.classList.toggle("sidebar-open");
    }});

    const fileButton = document.getElementById("fileButton");
    const fileInput = document.getElementById("fileInput");
    const filterInput = document.getElementById("filterModeInput");
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
    let currentFilterModes = new Set(
      (body.dataset.filterMode || "normal")
        .split(",")
        .map((value) => value.trim().toLowerCase())
        .filter((value) => value && value !== "normal")
    );

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
      frameCanvas.width = width;
      frameCanvas.height = height;
      const context = frameCanvas.getContext("2d");
      if (!context) return;
      context.drawImage(previewVideo, 0, 0, width, height);
      const blob = await new Promise((resolve) => frameCanvas.toBlob(resolve, "image/jpeg", 0.72));
      if (!blob) return;
      liveBusy = true;
      try {{
        const data = new FormData();
        data.append("frame", new File([blob], "frame.jpg", {{ type: "image/jpeg" }}));
        data.append("source_name", "Ekran");
        data.append("seconds", String((Date.now() - liveStartedAt) / 1000));
        data.append("session_id", liveSessionId);
        data.append("filter_mode", serializeFilterModes());
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

    fileInput?.addEventListener("change", () => {{
      if (fileInput?.files?.length) {{
        fileInput.form?.requestSubmit();
      }}
    }});

    screenButton?.addEventListener("click", async () => {{
      if (!navigator.mediaDevices?.getDisplayMedia) {{
        return;
      }}
      try {{
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
          livePlaceholder.classList.remove("hidden");
        }}
        frameCanvas = document.createElement("canvas");
        liveStartedAt = Date.now();
        screenStream.getVideoTracks()[0]?.addEventListener("ended", () => {{
          stopLiveLoop();
          stopTracks();
        }});
        await previewVideo.play();
        frameLoop = setInterval(sendLiveFrame, 500);
        await sendLiveFrame();
      }} catch (_error) {{}}
    }});
  </script>
</body>
</html>"""
    )


def _media_page(result: str = "", active_filter: str = "normal") -> HTMLResponse:
    active_filters = _parse_active_filters(active_filter)
    normal_class = "filter-tab active" if not active_filters else "filter-tab"
    bw_class = "filter-tab active" if "bw" in active_filters else "filter-tab"
    contrast_class = "filter-tab active" if "contrast" in active_filters else "filter-tab"
    deskew_class = "filter-tab active" if "deskew" in active_filters else "filter-tab"
    content = f"""
<section class="hero">
  <div>
    <h1>Media</h1>
  </div>
</section>
<section class="grid">
  <section class="panel media-shell">
    <div class="media-actions">
      <button class="source-tab" id="fileButton" type="button">Pliki</button>
      <button class="source-tab" id="screenButton" type="button">Ekran</button>
    </div>
    <form class="upload-form" action="/analyze" method="post" enctype="multipart/form-data">
      <input id="filterModeInput" type="hidden" name="filter_mode" value="{escape(active_filter)}">
      <input class="file-input" id="fileInput" type="file" name="media_file" accept="video/*,image/*">
    </form>
    {result}
    <div class="preview-controls">
      <button class="{normal_class}" data-filter-mode-value="normal" type="button">Normalny</button>
      <button class="{bw_class}" data-filter-mode-value="bw" type="button">Czarno-bialy</button>
      <button class="{contrast_class}" data-filter-mode-value="contrast" type="button">Kontrast</button>
      <button class="{deskew_class}" data-filter-mode-value="deskew" type="button">Prostowanie exp</button>
    </div>
  </section>
</section>
"""
    return _page(content, "media", active_filter)


def _history_page() -> HTMLResponse:
    history = _read_history()
    content = f"""
<section class="hero">
  <div>
    <h1>Historia</h1>
    <p>Wykryte tablice i zrodla.</p>
  </div>
</section>
<section class="panel">
  <div class="panel-head">
    <p>Historia</p>
    <h2>Wykrycia</h2>
  </div>
  <div class="history-list" style="margin-top:18px;">{_render_history(history)}</div>
</section>
"""
    return _page(content, "history", "normal")


def _ranking_page() -> HTMLResponse:
    history = _read_history()
    regions, dangerous = _build_ranking(history)
    content = f"""
<section class="hero">
  <div>
    <h1>Ranking</h1>
  </div>
</section>
<section class="split-grid">
  <section class="panel">
    <div class="panel-head">
      <p>Kody</p>
      <h2>Najwiecej wykryc</h2>
    </div>
    <div class="rank-list" style="margin-top:18px;">{_render_region_ranking(regions)}</div>
  </section>
  <section class="panel">
    <div class="panel-head">
      <p>Czerwone</p>
      <h2>Niebezpieczne tablice</h2>
    </div>
    <div class="danger-list" style="margin-top:18px;">{_render_dangerous_ranking(dangerous)}</div>
  </section>
</section>
"""
    return _page(content, "ranking", "normal")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return _media_page(
        '<section class="panel result-panel"><div class="preview-stack"><video class="preview-layer" id="rawPreview" autoplay muted playsinline></video><img class="preview-layer" id="livePreview" alt=""><div class="result-empty" id="livePlaceholder"></div></div></section>'
    )


@app.get("/history", response_class=HTMLResponse)
def history_page() -> HTMLResponse:
    return _history_page()


@app.get("/ranking", response_class=HTMLResponse)
def ranking_page() -> HTMLResponse:
    return _ranking_page()


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    media_file: UploadFile = File(...),
    filter_mode: str = Form("normal"),
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
    )
    _append_history(result.events)
    return _media_page(_render_result(result, original_name), active_filter=filter_mode)


@app.post("/analyze-live-frame")
async def analyze_live_frame(
    frame: UploadFile = File(...),
    source_name: str = Form("Ekran"),
    seconds: float = Form(0.0),
    session_id: str = Form("default"),
    filter_mode: str = Form("normal"),
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
    )
    _append_live_history(session_id, events)
    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok:
        return JSONResponse({"image": None}, status_code=500)
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return JSONResponse({"image": "data:image/jpeg;base64," + payload})
