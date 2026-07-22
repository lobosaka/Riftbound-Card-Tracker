"""High-level scan orchestration for the card OCR pipeline.

This module coordinates the full OCR search strategy after an image path is
provided.  It asks ``CardImageProcessor`` to generate many plausible views of the same
card and then runs OCR on each of them until a reliable code is found.

The workflow is:

1. Load the photo as a normalized RGB NumPy array.
2. Ask ``CardImageProcessor`` for card variants.
   - If card extraction succeeds, this includes a rectified card image.
   - The original photo is also kept as a fallback.
3. Try both ``native`` and ``rot180`` orientations to handle upside-down cards.
4. Crop several likely footer regions from the lower-left area where the card
   ID is expected to appear.
5. For each cropped region, generate OCR-friendly preprocessing variants such
   as grayscale, contrast-enhanced, sharpened, and thresholded images.
6. Send each of those variants through ``ocr_backend``.
7. Pass the OCR text through ``CardCodeParser``, which applies heuristics for
   common OCR mistakes and extracts possible card-ID candidates.
8. Check the candidates against ``CardRepository``, which holds the known
   card-ID index built from the card data.

The scanner keeps every candidate together with a source string that records
which variant produced it. If one candidate uniquely matches a known card, the
scan stops early. Otherwise the scanner ranks all collected candidates and
returns the strongest remaining match.
"""

from pathlib import Path

import numpy as np
from .card_code_parser import CardCodeParser
from .card_image_processor import CardImageProcessor
from .card_repository import CardRepository
from .ocr_backend import run_ocr



class ScanCandidate:
    def __init__(self, source, code):
        self.source = source
        self.code = code


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

        for source, image in self.iter_candidate_images(photo_rgb):
            for code in self.run_code_ocr(image):
                candidate = ScanCandidate(source=source, code=code)
                candidates.append(candidate)
                # Stop as soon as one OCR result maps to exactly one known card.
                if self.is_verified_code(code):
                    return candidates

        return candidates

    def iter_candidate_images(self, photo_rgb):
        images = []

        # Start with the rectified card if extraction worked, but keep the
        # original photo as a fallback when the geometric correction is wrong or
        # unavailable.
        for card_variant in self.image_processor.try_extract_card_variants(photo_rgb):
            # Try both upright and upside-down orientations before cropping the
            # likely footer regions that contain the printed card code.
            for oriented_card in self.image_processor.iter_oriented_cards(card_variant):
                for code_region in self.image_processor.crop_code_regions(oriented_card.image):
                    # Each crop is expanded into several OCR-friendly renderings
                    # so the backend can try different contrast and threshold
                    # conditions on the same physical code area.
                    for preprocess_variant in self.image_processor.preprocess_variants(code_region.image):
                        source = self.build_source_name(
                            card_variant.name,
                            oriented_card.name,
                            code_region.name,
                            preprocess_variant.name,
                        )
                        images.append((source, preprocess_variant.image))

        return images

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

    @staticmethod
    def build_source_name(
        card_variant,
        orientation_variant,
        region_variant,
        preprocess_variant,
    ):
        return (
            f"{card_variant}/"
            f"{orientation_variant}/"
            f"{region_variant}/"
            f"{preprocess_variant}"
        )
