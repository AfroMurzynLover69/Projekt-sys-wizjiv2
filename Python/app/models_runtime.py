from pathlib import Path

from .config import (
    BASE_DIR,
    FAST_ALPR_DETECTOR_CONF,
    FAST_ALPR_DETECTOR_MODEL,
    FAST_ALPR_OCR_MODEL,
    ONNX_PROVIDERS,
    RUNTIME_DEVICE,
    YOLO_DEVICE,
    YOLO_TRACKER_CONFIG_CANDIDATES,
    YOLO_VEHICLE_MODEL_CANDIDATES,
    YOLO_VEHICLE_NAMES,
)

try:
    from fast_alpr import ALPR as FastALPR
except Exception:
    FastALPR = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def init_ocr_engine(profile: dict | None = None):
    if FastALPR is None:
        raise RuntimeError("Missing fast-alpr.")

    runtime_device = str((profile or {}).get("runtime_device") or RUNTIME_DEVICE)
    onnx_providers = list((profile or {}).get("onnx_providers") or ONNX_PROVIDERS)
    engine = FastALPR(
        detector_model=FAST_ALPR_DETECTOR_MODEL,
        detector_conf_thresh=FAST_ALPR_DETECTOR_CONF,
        detector_providers=onnx_providers,
        ocr_model=FAST_ALPR_OCR_MODEL,
        ocr_device=runtime_device,
        ocr_providers=onnx_providers,
    )
    return engine


def _choose_tracker_config() -> str:
    for candidate in YOLO_TRACKER_CONFIG_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return "bytetrack.yaml"


def init_vehicle_tracker(profile: dict | None = None):
    if YOLO is None:
        return None, [], None

    yolo_device = str((profile or {}).get("yolo_device") or YOLO_DEVICE)
    profile_model = str((profile or {}).get("yolo_vehicle_model") or "").strip()
    candidates: list[Path | str] = []
    if profile_model:
        candidates.extend(
            [
                BASE_DIR / profile_model,
                BASE_DIR / "models" / profile_model,
                profile_model,
            ]
        )
    candidates.extend(YOLO_VEHICLE_MODEL_CANDIDATES)

    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if isinstance(candidate, Path) and not candidate.exists():
            continue
        model = YOLO(candidate_key)
        try:
            model.to(yolo_device)
        except Exception:
            pass
        vehicle_ids = [
            int(class_id)
            for class_id, class_name in model.names.items()
            if str(class_name).lower() in YOLO_VEHICLE_NAMES
        ]
        if not vehicle_ids:
            return None, [], None
        return model, vehicle_ids, _choose_tracker_config()

    return None, [], None
