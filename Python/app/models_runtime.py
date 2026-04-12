from .config import (
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


def init_ocr_engine():
    if FastALPR is None:
        raise RuntimeError("Missing fast-alpr.")

    engine = FastALPR(
        detector_model=FAST_ALPR_DETECTOR_MODEL,
        detector_conf_thresh=FAST_ALPR_DETECTOR_CONF,
        detector_providers=ONNX_PROVIDERS,
        ocr_model=FAST_ALPR_OCR_MODEL,
        ocr_device=RUNTIME_DEVICE,
        ocr_providers=ONNX_PROVIDERS,
    )
    return engine


def _choose_tracker_config() -> str:
    for candidate in YOLO_TRACKER_CONFIG_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return "bytetrack.yaml"


def init_vehicle_tracker():
    if YOLO is None:
        return None, [], None

    for candidate in YOLO_VEHICLE_MODEL_CANDIDATES:
        if not candidate.exists():
            continue
        model = YOLO(str(candidate))
        try:
            model.to(YOLO_DEVICE)
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
