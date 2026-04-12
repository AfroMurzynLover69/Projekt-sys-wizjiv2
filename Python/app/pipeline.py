from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from uuid import uuid4

import cv2
import numpy as np

from .config import (
    AI_EVERY_N_FRAMES,
    APP_TITLE,
    DANGEROUS_PLATES_LIST,
    ENABLE_YOLO_VEHICLE_TRACKING,
    PLATE_MAX_AREA_RATIO,
    PLATE_MAX_ASPECT,
    PLATE_MIN_AREA_RATIO,
    PLATE_MIN_ASPECT,
    PLATE_MIN_CONFIDENCE,
    RUNTIME_DEVICE,
    YOLO_DEVICE,
    YOLO_VEHICLE_CONF,
    YOLO_VEHICLE_LABEL,
)
from .media import is_image_file
from .models_runtime import init_ocr_engine, init_vehicle_tracker
from .plates import clean_plate_text


@dataclass
class DetectionEvent:
    plate_text: str
    source_name: str
    time_label: str
    time_seconds: float
    thumbnail_path: Path | None = None


@dataclass
class DetectionResult:
    exit_code: int
    output_path: Path | None = None
    media_kind: str = "unknown"
    events: list[DetectionEvent] = field(default_factory=list)


@dataclass
class RuntimeState:
    ocr_engine: object
    vehicle_model: object | None
    vehicle_ids: list[int]
    tracker_config: str | None
    dangerous_plates: set[str]


_RUNTIME: RuntimeState | None = None
_RUNTIME_LOCK = Lock()


def _check_highgui_support() -> bool:
    try:
        cv2.namedWindow("__cv_test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__cv_test__")
        return True
    except cv2.error:
        return False


def _get_runtime() -> RuntimeState:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME

        ocr_engine = init_ocr_engine()
        vehicle_model = None
        vehicle_ids: list[int] = []
        tracker_config = None
        if ENABLE_YOLO_VEHICLE_TRACKING:
            vehicle_model, vehicle_ids, tracker_config = init_vehicle_tracker()

        _RUNTIME = RuntimeState(
            ocr_engine=ocr_engine,
            vehicle_model=vehicle_model,
            vehicle_ids=vehicle_ids,
            tracker_config=tracker_config,
            dangerous_plates=_load_dangerous_plates(DANGEROUS_PLATES_LIST),
        )
        return _RUNTIME


def _close_preview(can_preview: bool) -> None:
    if not can_preview:
        return
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def _mean_confidence(conf_raw) -> float:
    if isinstance(conf_raw, list):
        return float(sum(conf_raw) / max(1, len(conf_raw)))
    return float(conf_raw or 0.0)


def _read_image(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: Path, image) -> bool:
    suffix = path.suffix.lower() if path.suffix else ".png"
    ext = suffix if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
        return True
    except OSError:
        return False


def _parse_filter_modes(filter_mode: str | None) -> list[str]:
    if not filter_mode:
        return []
    modes: list[str] = []
    for raw in str(filter_mode).split(","):
        mode = raw.strip().lower()
        if not mode or mode == "normal" or mode in modes:
            continue
        modes.append(mode)
    return modes


def _apply_filter_mode(frame, filter_mode: str):
    filtered = frame.copy()
    modes = _parse_filter_modes(filter_mode)
    if "contrast" in modes:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        boosted_l = clahe.apply(l_channel)
        boosted = cv2.merge((boosted_l, a_channel, b_channel))
        boosted_bgr = cv2.cvtColor(boosted, cv2.COLOR_LAB2BGR)
        filtered = cv2.convertScaleAbs(boosted_bgr, alpha=1.18, beta=6)
    if "bw" in modes:
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        filtered = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return filtered


def _crop_region(image, box: tuple[int, int, int, int], pad: int = 10):
    x1, y1, x2, y2 = box
    image_h, image_w = image.shape[:2]
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(image_w, x2 + pad)
    cy2 = min(image_h, y2 + pad)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    return crop


def _estimate_plate_angle(image) -> float | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 180)
    ys, xs = np.where(edges > 0)
    if xs.size < 32:
        return None
    points = np.column_stack((xs, ys)).astype(np.float32)
    (_center, (width, height), angle) = cv2.minAreaRect(points)
    if width <= 1 or height <= 1:
        return None
    if width < height:
        angle += 90.0
    if angle < -45.0:
        angle += 90.0
    if angle > 45.0:
        angle -= 90.0
    if abs(angle) < 4.0 or abs(angle) > 30.0:
        return None
    return angle


def _rotate_image(image, angle: float):
    image_h, image_w = image.shape[:2]
    center = (image_w / 2.0, image_h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (image_w, image_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _refine_plate_prediction(
    frame,
    box: tuple[int, int, int, int],
    text: str,
    confidence: float,
    ocr_engine,
    filter_modes: list[str],
) -> tuple[str, float]:
    if "deskew" not in filter_modes:
        return text, confidence

    crop = _crop_region(frame, box, pad=10)
    if crop is None:
        return text, confidence

    angle = _estimate_plate_angle(crop)
    if angle is None:
        return text, confidence

    try:
        deskewed_crop = _rotate_image(crop, angle)
        refined_results = ocr_engine.predict(deskewed_crop)
    except Exception:
        return text, confidence

    best_text = text
    best_confidence = confidence

    for result in refined_results:
        ocr = getattr(result, "ocr", None)
        if ocr is None:
            continue
        candidate_text = clean_plate_text(str(getattr(ocr, "text", "")))
        if not candidate_text:
            continue
        candidate_confidence = _mean_confidence(getattr(ocr, "confidence", 0.0))
        if candidate_confidence > best_confidence:
            best_text = candidate_text
            best_confidence = candidate_confidence

    return best_text, best_confidence


def _build_output_path(source_path: Path, output_dir: Path | None) -> Path:
    target_dir = output_dir or source_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    if is_image_file(source_path):
        return target_dir / f"{source_path.stem}_analiza{source_path.suffix}"
    return target_dir / f"{source_path.stem}_analiza.mp4"


def _create_video_writer(source_path: Path, output_dir: Path | None, width: int, height: int, fps: float):
    out_path = _build_output_path(source_path, output_dir)
    safe_fps = fps if 1.0 <= fps <= 240.0 else 25.0
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        safe_fps,
        (width, height),
    )
    return writer, out_path, safe_fps


def _load_dangerous_plates(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()

    dangerous: set[str] = set()
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        cleaned = clean_plate_text(raw)
        if cleaned:
            dangerous.add(cleaned)
    return dangerous


def _draw_double_box(
    image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int = 2,
    gap: int = 4,
) -> None:
    img_h, img_w = image.shape[:2]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    ox1 = max(0, x1 - gap)
    oy1 = max(0, y1 - gap)
    ox2 = min(img_w - 1, x2 + gap)
    oy2 = min(img_h - 1, y2 + gap)
    cv2.rectangle(image, (ox1, oy1), (ox2, oy2), color, thickness)


def _extract_plate_predictions(frame, ocr_engine, filter_mode: str = "normal") -> list[tuple[int, int, int, int, str, float]]:
    try:
        results = ocr_engine.predict(frame)
    except Exception:
        return []

    frame_h, frame_w = frame.shape[:2]
    filter_modes = _parse_filter_modes(filter_mode)
    predictions: list[tuple[int, int, int, int, str, float]] = []

    for result in results:
        detection = getattr(result, "detection", None)
        bbox = getattr(detection, "bounding_box", None)
        ocr = getattr(result, "ocr", None)
        if bbox is None or ocr is None:
            continue

        raw_text = str(getattr(ocr, "text", ""))
        text = clean_plate_text(raw_text)
        confidence = _mean_confidence(getattr(ocr, "confidence", 0.0))

        x1 = max(0, min(int(getattr(bbox, "x1", 0)), frame_w - 1))
        y1 = max(0, min(int(getattr(bbox, "y1", 0)), frame_h - 1))
        x2 = max(x1 + 1, min(int(getattr(bbox, "x2", 0)), frame_w))
        y2 = max(y1 + 1, min(int(getattr(bbox, "y2", 0)), frame_h))
        w = x2 - x1
        h = y2 - y1
        aspect = w / max(1, h)
        area_ratio = (w * h) / max(1, frame_w * frame_h)

        if aspect < PLATE_MIN_ASPECT or aspect > PLATE_MAX_ASPECT:
            continue
        if area_ratio < PLATE_MIN_AREA_RATIO or area_ratio > PLATE_MAX_AREA_RATIO:
            continue

        text, confidence = _refine_plate_prediction(
            frame,
            (x1, y1, x2, y2),
            text,
            confidence,
            ocr_engine,
            filter_modes,
        )
        if not text:
            continue
        if confidence < PLATE_MIN_CONFIDENCE:
            continue
        predictions.append((x1, y1, x2, y2, text, confidence))

    return predictions


def _extract_vehicle_predictions(
    frame,
    vehicle_model,
    vehicle_ids: list[int],
    tracker_config: str | None,
) -> list[tuple[int, int, int, int, int | None]]:
    if vehicle_model is None or not vehicle_ids:
        return []
    try:
        result = vehicle_model.track(
            source=frame,
            classes=vehicle_ids,
            conf=YOLO_VEHICLE_CONF,
            device=YOLO_DEVICE,
            tracker=tracker_config or "bytetrack.yaml",
            persist=True,
            verbose=False,
        )[0]
    except Exception:
        try:
            result = vehicle_model.predict(
                source=frame,
                classes=vehicle_ids,
                conf=YOLO_VEHICLE_CONF,
                device=YOLO_DEVICE,
                verbose=False,
            )[0]
        except Exception:
            return []

    if result.boxes is None or len(result.boxes) == 0:
        return []

    has_track_ids = result.boxes.id is not None
    track_ids = result.boxes.id.cpu().tolist() if has_track_ids else [None] * len(result.boxes)
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    frame_h, frame_w = frame.shape[:2]
    predictions: list[tuple[int, int, int, int, int | None]] = []

    for track_id_raw, box in zip(track_ids, boxes):
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(x1 + 1, min(x2, frame_w))
        y2 = max(y1 + 1, min(y2, frame_h))
        track_id = int(track_id_raw) if track_id_raw is not None else None
        predictions.append((x1, y1, x2, y2, track_id))
    return predictions


def _match_plate_to_vehicle(
    plate_box: tuple[int, int, int, int],
    vehicle_predictions: list[tuple[int, int, int, int, int | None]],
) -> int | None:
    px1, py1, px2, py2 = plate_box
    cx = (px1 + px2) // 2
    cy = (py1 + py2) // 2
    best_idx = None
    best_overlap = 0

    for idx, (vx1, vy1, vx2, vy2, _) in enumerate(vehicle_predictions):
        if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
            return idx

        ix1 = max(px1, vx1)
        iy1 = max(py1, vy1)
        ix2 = min(px2, vx2)
        iy2 = min(py2, vy2)
        overlap_w = max(0, ix2 - ix1)
        overlap_h = max(0, iy2 - iy1)
        overlap = overlap_w * overlap_h
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    if best_overlap > 0:
        return best_idx
    return None


def _draw_vehicle_overlay(
    annotated,
    vehicle_predictions: list[tuple[int, int, int, int, int | None]],
    dangerous_vehicle_indexes: set[int],
) -> None:
    for idx, (x1, y1, x2, y2, track_id) in enumerate(vehicle_predictions):
        is_dangerous = idx in dangerous_vehicle_indexes
        color = (0, 0, 255) if is_dangerous else (0, 220, 120)
        if is_dangerous:
            _draw_double_box(annotated, x1, y1, x2, y2, color, thickness=2, gap=5)
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = YOLO_VEHICLE_LABEL if track_id is None else f"{YOLO_VEHICLE_LABEL} ID {track_id}"
        if is_dangerous:
            label = f"NIEBEZPIECZNY {label}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_overlay(
    frame,
    ocr_engine,
    vehicle_model=None,
    vehicle_ids=None,
    tracker_config=None,
    dangerous_plates: set[str] | None = None,
    filter_mode: str = "normal",
):
    annotated = frame.copy()
    vehicle_predictions = _extract_vehicle_predictions(
        frame,
        vehicle_model,
        vehicle_ids or [],
        tracker_config,
    )
    predictions = _extract_plate_predictions(frame, ocr_engine, filter_mode=filter_mode)
    dangerous_plates = dangerous_plates or set()
    dangerous_plate_indexes: set[int] = set()
    dangerous_vehicle_indexes: set[int] = set()

    for idx, (x1, y1, x2, y2, text, _confidence) in enumerate(predictions):
        if text not in dangerous_plates:
            continue
        dangerous_plate_indexes.add(idx)
        match_idx = _match_plate_to_vehicle((x1, y1, x2, y2), vehicle_predictions)
        if match_idx is not None:
            dangerous_vehicle_indexes.add(match_idx)

    _draw_vehicle_overlay(annotated, vehicle_predictions, dangerous_vehicle_indexes)

    for idx, (x1, y1, x2, y2, text, confidence) in enumerate(predictions):
        is_dangerous = idx in dangerous_plate_indexes
        if is_dangerous:
            _draw_double_box(annotated, x1, y1, x2, y2, (0, 0, 255), thickness=2, gap=4)
            plate_label = f"NIEBEZPIECZNY {text} ({confidence:.2f})"
            plate_color = (0, 0, 255)
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 170, 255), 2)
            plate_label = f"{text} ({confidence:.2f})"
            plate_color = (0, 255, 0)
        cv2.putText(
            annotated,
            plate_label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            plate_color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"{APP_TITLE} {RUNTIME_DEVICE.upper()}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 220, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated, predictions


def _format_time_label(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _crop_thumbnail(image, box: tuple[int, int, int, int]):
    x1, y1, x2, y2 = box
    pad = 18
    frame_h, frame_w = image.shape[:2]
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(frame_w, x2 + pad)
    cy2 = min(frame_h, y2 + pad)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    return crop


def _build_event(
    image,
    box: tuple[int, int, int, int],
    text: str,
    source_name: str,
    seconds: float,
    history_dir: Path | None,
) -> DetectionEvent:
    thumbnail_path = None
    crop = _crop_thumbnail(image, box)
    if crop is not None and history_dir is not None:
        history_dir.mkdir(parents=True, exist_ok=True)
        candidate = history_dir / f"{uuid4().hex}.png"
        if _write_image(candidate, crop):
            thumbnail_path = candidate

    return DetectionEvent(
        plate_text=text,
        source_name=source_name,
        time_label=_format_time_label(seconds),
        time_seconds=seconds,
        thumbnail_path=thumbnail_path,
    )


def _build_events_from_predictions(
    annotated,
    predictions: list[tuple[int, int, int, int, str, float]],
    source_name: str,
    seconds: float,
    history_dir: Path | None,
) -> list[DetectionEvent]:
    seen_texts: set[str] = set()
    events: list[DetectionEvent] = []
    for x1, y1, x2, y2, text, _confidence in predictions:
        if text in seen_texts:
            continue
        seen_texts.add(text)
        events.append(
            _build_event(
                annotated,
                (x1, y1, x2, y2),
                text,
                source_name,
                seconds,
                history_dir,
            )
        )
    return events


def analyze_frame(
    frame,
    source_name: str,
    seconds: float = 0.0,
    history_dir: Path | None = None,
    filter_mode: str = "normal",
) -> tuple[object, list[DetectionEvent]]:
    runtime = _get_runtime()
    processed_frame = _apply_filter_mode(frame, filter_mode)
    annotated, predictions = _draw_overlay(
        processed_frame,
        runtime.ocr_engine,
        runtime.vehicle_model,
        runtime.vehicle_ids,
        runtime.tracker_config,
        runtime.dangerous_plates,
        filter_mode=filter_mode,
    )
    events = _build_events_from_predictions(
        annotated,
        predictions,
        source_name,
        seconds,
        history_dir,
    )
    return annotated, events


def run_image_detection(
    image_path: Path,
    output_dir: Path | None = None,
    preview: bool = True,
    source_name: str | None = None,
    history_dir: Path | None = None,
    filter_mode: str = "normal",
) -> DetectionResult:
    if not image_path.exists():
        print(f"Missing: {image_path}")
        return DetectionResult(1, None, "image")

    frame = _read_image(image_path)
    if frame is None or frame.size == 0:
        print("Read error.")
        return DetectionResult(1, None, "image")

    try:
        runtime = _get_runtime()
    except RuntimeError as error:
        print(str(error))
        return DetectionResult(1, None, "image")

    can_preview = preview and _check_highgui_support()
    source_label = source_name or image_path.name
    out_path = _build_output_path(image_path, output_dir)
    processed_frame = _apply_filter_mode(frame, filter_mode)
    annotated, predictions = _draw_overlay(
        processed_frame,
        runtime.ocr_engine,
        runtime.vehicle_model,
        runtime.vehicle_ids,
        runtime.tracker_config,
        runtime.dangerous_plates,
        filter_mode=filter_mode,
    )
    saved = _write_image(out_path, annotated)
    if not saved:
        print(f"Save error: {out_path}")

    events = _build_events_from_predictions(
        annotated,
        predictions,
        source_label,
        0.0,
        history_dir,
    )

    try:
        while can_preview:
            cv2.imshow(APP_TITLE, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        _close_preview(can_preview)

    return DetectionResult(0 if saved else 1, out_path if saved else None, "image", events)


def run_detection(
    video_path: Path,
    output_dir: Path | None = None,
    preview: bool = True,
    source_name: str | None = None,
    history_dir: Path | None = None,
    filter_mode: str = "normal",
) -> DetectionResult:
    if not video_path.exists():
        print(f"Missing: {video_path}")
        return DetectionResult(1, None, "video")

    if is_image_file(video_path):
        return run_image_detection(
            video_path,
            output_dir=output_dir,
            preview=preview,
            source_name=source_name,
            history_dir=history_dir,
            filter_mode=filter_mode,
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Open error.")
        return DetectionResult(1, None, "video")

    try:
        runtime = _get_runtime()
    except RuntimeError as error:
        print(str(error))
        cap.release()
        return DetectionResult(1, None, "video")

    can_preview = preview and _check_highgui_support()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if width <= 0 or height <= 0:
        cap.release()
        return DetectionResult(1, None, "video")

    writer, out_path, effective_fps = _create_video_writer(video_path, output_dir, width, height, fps)
    if writer is None or not writer.isOpened():
        cap.release()
        _close_preview(can_preview)
        print("Save error.")
        return DetectionResult(1, None, "video")

    source_label = source_name or video_path.name
    frame_idx = 0
    last_out = None
    processed_predictions: list[tuple[int, int, int, int, str, float]] = []
    history_keys: set[tuple[str, int]] = set()
    events: list[DetectionEvent] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            seconds = (frame_idx - 1) / effective_fps if effective_fps > 0 else 0.0
            should_process = last_out is None or (frame_idx % AI_EVERY_N_FRAMES) == 0

            if should_process:
                processed_frame = _apply_filter_mode(frame, filter_mode)
                last_out, processed_predictions = _draw_overlay(
                    processed_frame,
                    runtime.ocr_engine,
                    runtime.vehicle_model,
                    runtime.vehicle_ids,
                    runtime.tracker_config,
                    runtime.dangerous_plates,
                    filter_mode=filter_mode,
                )
                minute_idx = int(seconds // 60)
                for x1, y1, x2, y2, text, _confidence in processed_predictions:
                    key = (text, minute_idx)
                    if key in history_keys:
                        continue
                    history_keys.add(key)
                    events.append(
                        _build_event(
                            last_out,
                            (x1, y1, x2, y2),
                            text,
                            source_label,
                            seconds,
                            history_dir,
                        )
                    )

            if last_out is not None:
                writer.write(last_out)

            if can_preview and last_out is not None:
                cv2.imshow(APP_TITLE, last_out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        _close_preview(can_preview)

    return DetectionResult(0, out_path, "video", events)
