from pathlib import Path
import sys
from time import perf_counter


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


def main(image_path=None):
    if image_path is None:
        if len(sys.argv) < 2:
            image_path = str(
                Path(__file__).resolve().parent / "images/1.jpg"
            )
        else:
            image_path = sys.argv[1]

    return process_image(image_path)


if __name__ == "__main__":
    main()
