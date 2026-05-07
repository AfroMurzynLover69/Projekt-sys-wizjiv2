import argparse
import os

# Limitujemy wątki do 20 dla procesorów wielordzeniowych zamiast sztywnej 4
MAX_THREADS = "20"

os.environ["OMP_NUM_THREADS"] = MAX_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = MAX_THREADS
os.environ["MKL_NUM_THREADS"] = MAX_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = MAX_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = MAX_THREADS

import cv2
cv2.setNumThreads(int(MAX_THREADS))

from .media import choose_media_file
from .pipeline import run_detection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "media_path",
        nargs="?",
        help="Path to file.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    if not args.media_path:
        import uvicorn

        print(f"http://{args.host}:{args.port}")
        uvicorn.run("app.web:app", host=args.host, port=args.port)
        return 0

    media_path = choose_media_file(args.media_path)
    if media_path is None:
        return 0

    return run_detection(media_path).exit_code
