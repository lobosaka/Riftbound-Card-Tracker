"""High-level scan orchestration for the card OCR pipeline.

The current scanner runs a single production path: load the full photo, send it
through the OCR backend, normalize the detected text into code candidates, and
select the strongest match against the card repository.
"""

from dataclasses import dataclass

from .ocr_backend import run_ocr


@dataclass(frozen=True)
class ScanCandidate:
    source: str
    code: str


class CardScanner:
    def __init__(
        self,
        repository,
        parser,
        image_processor,
    ):
        self.repository = repository
        self.parser = parser
        self.image_processor = image_processor

    def extract_code_from_photo(self, image_path):
        photo = self.image_processor.load_rgb_image(image_path)
        candidates = self.collect_ocr_candidates(photo)
        return self.choose_best_candidate(candidates), candidates

    def collect_ocr_candidates(self, photo_rgb):
        candidates = []
        for code in self.run_code_ocr(photo_rgb):
            candidate = ScanCandidate(source="photo/native/full/raw", code=code)
            candidates.append(candidate)
            # Stop as soon as one OCR result maps to exactly one known card.
            if self.is_verified_code(code):
                return candidates
        return candidates

    def is_verified_code(self, code):
        return len(self.repository.verify_code(code)) == 1

    def run_code_ocr(self, image):
        observations = run_ocr(image)
        lines = [str(item.get("text", "")) for item in observations if item.get("text")]
        # Convert raw OCR strings into normalized card-code candidates.
        return self.parser.extract_candidates(lines)

    def choose_best_candidate(self, candidates):
        if not candidates:
            return ""
        return max(candidates, key=lambda candidate: self.parser.score_candidate(candidate.code)).code

    def print_result(self, image_file, code, candidates):
        matches = self.repository.verify_code(code)
        print(f"{image_file.name}: {code} [{self.repository.format_match_status(matches)}]")

        for candidate in candidates:
            variant_matches = self.repository.verify_code(candidate.code)
            marker = "ok" if variant_matches else "missing"
            print(f"  {candidate.source}: {candidate.code} ({marker})")
