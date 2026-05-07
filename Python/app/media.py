from pathlib import Path

from .config import IMAGE_EXTENSIONS


def _normalize_path(raw_path: str) -> Path:
    cleaned = raw_path.strip().strip('"').strip("'")
    return Path(cleaned).expanduser()


def choose_media_file(cli_path: str | None = None) -> Path | None:
    if cli_path:
        return _normalize_path(cli_path)

    try:
        file_path = input("Plik: ").strip()
    except EOFError:
        return None

    if not file_path:
        return None
    return _normalize_path(file_path)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower().strip() in IMAGE_EXTENSIONS
