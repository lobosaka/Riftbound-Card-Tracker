import numpy as np
from .backend.card_codes import CardCodeParser, CardRepository
from .backend.ocr_backend import run_ocr


class CardScanner:
    def __init__(self, repository, parser):
        self.repository = repository
        self.parser = parser

    def collect_ocr_candidates(self, photo_rgb):
        candidates = []
        for code in self.run_code_ocr(photo_rgb):
            candidates.append(code)
            if len(self.repository.verify_code(code)) == 1:
                return candidates
        return candidates

    def run_code_ocr(self, image):
        observations = run_ocr(image)
        lines = [str(item.get("text", "")) for item in observations if item.get("text")]
        return self.parser.extract_candidates(lines)

    def choose_best_candidate(self, candidates):
        if not candidates:
            return ""
        return max(
            candidates,
            key=self.parser.score_candidate,
        )

# Entrypoint from api
def process_uploaded_image(image, filename=None):
    photo_rgb = np.array(image.convert("RGB"))
    # Build scanner
    repository = CardRepository()
    parser = CardCodeParser(repository)
    scanner = CardScanner(repository, parser)
    # Collect candidates for the uploaded image
    candidates = scanner.collect_ocr_candidates(photo_rgb)
    # Choose the best candidate based on scoring
    code = scanner.choose_best_candidate(candidates)

    return {
        "image_path": filename,
        "code": code,
        "candidates": [
        {"source": "photo/native/full/raw", "code": code}
        for code in candidates
    ],
    }
