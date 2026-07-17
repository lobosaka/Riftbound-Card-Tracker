from pathlib import Path
import sys
from backend.card_code_parser import CardCodeParser
from backend.card_image_processor import CardImageProcessor
from backend.card_repository import CardRepository
from backend.card_scanner import CardScanner



def build_scanner():
    image_processor = CardImageProcessor()
    repository = CardRepository(Path(__file__).resolve().parents[2] / "riftbound.db")
    parser = CardCodeParser(repository)
    return CardScanner(repository, parser, image_processor)


def process_image(image_path):
    image_file = Path(image_path).expanduser().resolve()
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")
    if not image_file.is_file():
        raise ValueError(f"Expected an image file path, got: {image_file}")

    scanner = build_scanner()
    code, candidates = scanner.extract_code_from_photo(str(image_file))
    scanner.print_result(image_file, code, candidates)
    return code, candidates


def main(image_path=None):
    image_path = "OCR/images/1.jpg"  # Default image path for testing
    if image_path is None:
        if len(sys.argv) < 2:
            raise ValueError("Usage: python pilot/main.py <image_path>")
        image_path = sys.argv[1]

    return process_image(image_path)


if __name__ == "__main__":
    main()
