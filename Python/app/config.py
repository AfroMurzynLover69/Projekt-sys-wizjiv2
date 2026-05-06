from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
AI_EVERY_N_FRAMES = 5
DANGEROUS_PLATES_LIST = BASE_DIR / "list.txt"
APP_TITLE = "ALPR"
RUNTIME_DEVICE = "cpu"
FAST_ALPR_DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
FAST_ALPR_OCR_MODEL = "european-plates-mobile-vit-v2-model"
FAST_ALPR_DETECTOR_CONF = 0.3
ONNX_PROVIDERS = ["CPUExecutionProvider"]
DEFAULT_AI_PROFILE = "cpu_lite"
AI_PROFILES = {
    "cpu_lite": {
        "label": "CPU Lite",
        "runtime_device": "cpu",
        "yolo_device": "cpu",
        "onnx_providers": ["CPUExecutionProvider"],
        "ai_every_n_frames": 12,
        "enable_yolo_vehicle_tracking": False,
        "live_interval_ms": 2600,
        "live_jpeg_quality": 0.45,
        "live_max_width": 640,
        "full_file_analysis": False,
        "description": "mniejsze obciazenie CPU",
    },
    "gpu_heavy": {
        "label": "GPU Heavy",
        "runtime_device": "cuda",
        "yolo_device": "0",
        "onnx_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "ai_every_n_frames": 3,
        "enable_yolo_vehicle_tracking": True,
        "live_interval_ms": 500,
        "live_jpeg_quality": 0.82,
        "live_max_width": 1280,
        "full_file_analysis": True,
        "description": "czestsza analiza i sledzenie pojazdow",
    },
}
PLATE_TEXT_MIN_LEN = 5
PLATE_TEXT_MAX_LEN = 10
PLATE_MIN_CONFIDENCE = 0.2
PLATE_MIN_ASPECT = 1.6
PLATE_MAX_ASPECT = 8.0
PLATE_MIN_AREA_RATIO = 0.0002
PLATE_MAX_AREA_RATIO = 0.12

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ENABLE_YOLO_VEHICLE_TRACKING = True
YOLO_VEHICLE_CONF = 0.35
YOLO_VEHICLE_LABEL = "SAMOCHOD"
YOLO_DEVICE = "cpu"
YOLO_VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle"}
YOLO_VEHICLE_MODEL_CANDIDATES = [
    BASE_DIR / "yolov8n.pt",
    BASE_DIR / "models" / "yolov8n.pt",
    BASE_DIR / "yolo11x.pt",
    BASE_DIR / "models" / "yolo11x.pt",
]
YOLO_TRACKER_CONFIG_CANDIDATES = [
    BASE_DIR / "bytetrack_plates.yaml",
    BASE_DIR / "models" / "bytetrack_plates.yaml",
    BASE_DIR / "bytetrack.yaml",
    BASE_DIR / "models" / "bytetrack.yaml",
]


def get_ai_profile(profile_id: str | None = None) -> dict:
    key = str(profile_id or DEFAULT_AI_PROFILE).strip().lower()
    if key not in AI_PROFILES:
        key = DEFAULT_AI_PROFILE
    return {
        "id": key,
        **AI_PROFILES[key],
    }
