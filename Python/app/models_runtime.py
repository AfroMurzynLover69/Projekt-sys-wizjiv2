from pathlib import Path
import shutil
import warnings

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
except Exception as error:
    warnings.warn(f"Ultralytics YOLO is not available; vehicle boxes are disabled: {error}")
    YOLO = None


def _cuda_device_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _normalize_runtime_device(device: str) -> str:
    normalized = str(device or RUNTIME_DEVICE).strip().lower()
    if normalized in {"cpu", ""}:
        return "cpu"
    if normalized.startswith("cuda") and not _cuda_device_available():
        warnings.warn("CUDA is not available for OCR; using CPU.")
        return "cpu"
    return device


def _normalize_yolo_device(device: str) -> str:
    normalized = str(device or YOLO_DEVICE).strip().lower()
    if normalized in {"cpu", ""}:
        return "cpu"

    if _cuda_device_available():
        return device

    warnings.warn("CUDA is not available for Ultralytics YOLO; using CPU for vehicle tracking.")
    return "cpu"


def _available_onnx_providers() -> set[str] | None:
    try:
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        return set(ort.get_available_providers())
    except Exception:
        return None


def _normalize_onnx_providers(requested_providers: list[str]) -> list[str]:
    available = _available_onnx_providers()
    if "CUDAExecutionProvider" in requested_providers and not _cuda_device_available():
        warnings.warn("CUDA device is not available for ONNX Runtime; using CPUExecutionProvider.")
        requested_providers = [
            provider for provider in requested_providers if provider != "CUDAExecutionProvider"
        ]

    if not available:
        return requested_providers

    providers = [provider for provider in requested_providers if provider in available]
    if providers:
        if "CUDAExecutionProvider" in requested_providers and "CUDAExecutionProvider" not in providers:
            warnings.warn("ONNX Runtime CUDA provider is not available; using CPUExecutionProvider.")
        return providers

    if "CPUExecutionProvider" in available:
        warnings.warn("Requested ONNX providers are not available; using CPUExecutionProvider.")
        return ["CPUExecutionProvider"]
    return requested_providers


def _resolve_yolo_weight(model_name: str) -> Path | str:
    model_path = Path(model_name)
    if model_path.is_absolute():
        return model_path

    if len(model_path.parts) == 1 and model_path.suffix == ".pt":
        target = BASE_DIR / "models" / model_path.name
        legacy = BASE_DIR / model_path.name
        if target.exists():
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        if legacy.exists():
            try:
                legacy.replace(target)
            except OSError:
                shutil.copy2(legacy, target)
            return target

        try:
            from ultralytics.utils.downloads import attempt_download_asset

            return Path(attempt_download_asset(target))
        except Exception:
            return target

    return BASE_DIR / model_path


def init_ocr_engine(profile: dict | None = None):
    if FastALPR is None:
        raise RuntimeError("Missing fast-alpr.")

    runtime_device = _normalize_runtime_device(str((profile or {}).get("runtime_device") or RUNTIME_DEVICE))
    onnx_providers = _normalize_onnx_providers(
        list((profile or {}).get("onnx_providers") or ONNX_PROVIDERS)
    )
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

    yolo_device = _normalize_yolo_device(str((profile or {}).get("yolo_device") or YOLO_DEVICE))
    profile_model = str((profile or {}).get("yolo_vehicle_model") or "").strip()
    candidates: list[Path | str] = []
    if profile_model:
        candidates.append(_resolve_yolo_weight(profile_model))
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
