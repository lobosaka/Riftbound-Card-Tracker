from pathlib import Path
import logging
import sys
from time import perf_counter

import numpy as np

if __package__ in (None, ""):
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from backend.card_code_parser import CardCodeParser
    from backend.card_image_processor import CardImageProcessor
    from backend.card_repository import CardRepository
    from backend.card_scanner import CardScanner
    from backend.ocr_backend import describe_ocr_backend
else:
    from .backend.card_code_parser import CardCodeParser
    from .backend.card_image_processor import CardImageProcessor
    from .backend.card_repository import CardRepository
    from .backend.card_scanner import CardScanner
    from .backend.ocr_backend import describe_ocr_backend


logger = logging.getLogger(__name__)
OCR_IMAGES_DIR = Path(__file__).resolve().parent / "images"
OCR_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_scanner():
    image_processor = CardImageProcessor()
    repository = CardRepository()
    parser = CardCodeParser(repository)
    return CardScanner(repository, parser, image_processor)


def resolve_image_file(image_path):
    image_file = Path(image_path).expanduser().resolve()
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")
    if not image_file.is_file():
        raise ValueError(f"Expected an image file path, got: {image_file}")
    return image_file


def get_latest_ocr_image_path():
    image_paths = [
        path
        for path in OCR_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in OCR_IMAGE_EXTENSIONS
    ]

    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {OCR_IMAGES_DIR}")

    return max(image_paths, key=lambda path: path.stat().st_mtime)


def resolve_ocr_image_path(image_path=None):
    if image_path is None:
        return get_latest_ocr_image_path()
    return resolve_image_file(image_path)


def serialize_candidates(candidates):
    return [
        {"source": candidate.source, "code": candidate.code}
        for candidate in candidates
    ]


def process_image(image_path, emit_result=True):
    image_file = resolve_image_file(image_path)

    scanner = build_scanner()
    code, candidates = scanner.extract_code_from_photo(str(image_file))
    if emit_result:
        scanner.print_result(image_file, code, candidates)
    return code, candidates


def process_image_with_debug(image_path):
    image_file = resolve_image_file(image_path)

    scanner = build_scanner()
    started_at = perf_counter()
    code, candidates = scanner.extract_code_from_photo(str(image_file))
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)

    return {
        "image_path": str(image_file),
        "code": code,
        "candidates": serialize_candidates(candidates),
        "debug": {
            "elapsed_ms": elapsed_ms,
            "candidate_count": len(candidates),
            "selected_code_found": bool(code),
            "file_size_bytes": image_file.stat().st_size,
            "ocr_backend": describe_ocr_backend(),
        },
    }


def process_uploaded_image(image, filename=None, file_size_bytes=None, debug=False):
    photo_rgb = np.array(image.convert("RGB"))

    scanner = build_scanner()
    started_at = perf_counter()
    candidates = scanner.collect_ocr_candidates(photo_rgb)
    code = scanner.choose_best_candidate(candidates)
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)

    result = {
        "image_path": filename,
        "code": code,
        "candidates": serialize_candidates(candidates),
    }

    if debug:
        result["debug"] = {
            "elapsed_ms": elapsed_ms,
            "candidate_count": len(candidates),
            "selected_code_found": bool(code),
            "file_size_bytes": file_size_bytes,
            "ocr_backend": describe_ocr_backend(),
        }

    return result


def run_ocr_for_image(image_path=None, debug=False):
    resolved_path = resolve_ocr_image_path(image_path)
    logger.info("Starting OCR for image: %s", resolved_path)

    if debug:
        result = process_image_with_debug(str(resolved_path))
    else:
        code, candidates = process_image(str(resolved_path), emit_result=False)
        result = {
            "image_path": str(resolved_path),
            "code": code,
            "candidates": serialize_candidates(candidates),
        }

    logger.info(
        "Finished OCR for image: %s with %s candidates",
        resolved_path,
        len(result["candidates"]),
    )
    return result


def main(image_path=None):
    if image_path is None:
        if len(sys.argv) < 2:
            image_path = str(get_latest_ocr_image_path())
        else:
            image_path = sys.argv[1]

    return process_image(image_path)


if __name__ == "__main__":
    main()
