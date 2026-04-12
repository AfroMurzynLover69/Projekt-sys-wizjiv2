import argparse
import sys

from .media import choose_media_file
from .pipeline import run_detection


def main() -> int:
    if not sys.platform.startswith("linux"):
        print("Linux only.")
        return 1

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "media_path",
        nargs="?",
        help="Path to file.",
    )
    args = parser.parse_args()

    media_path = choose_media_file(args.media_path)
    if media_path is None:
        return 0

    return run_detection(media_path).exit_code
