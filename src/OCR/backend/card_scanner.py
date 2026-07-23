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
import re

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

    def extract_code_from_photo(self, image_path, debug_variant_dir=None):
        photo = self.image_processor.load_rgb_image(image_path)
        candidates = self.collect_ocr_candidates(photo, debug_variant_dir=debug_variant_dir)
        return self.choose_best_candidate(candidates), candidates

    def collect_ocr_candidates(self, photo_rgb, debug_variant_dir=None):
        candidates = []
        attempted_images = []

        for source, image in self.iter_candidate_images(photo_rgb):
            attempted_images.append((source, image))
            for code in self.run_code_ocr(image):
                candidate = ScanCandidate(source=source, code=code)
                candidates.append(candidate)
                # Stop as soon as one OCR result maps to exactly one known card.
                if self.is_verified_code(code):
                    self.save_attempted_variants(debug_variant_dir, attempted_images)
                    return candidates

        self.save_attempted_variants(debug_variant_dir, attempted_images)
        return candidates

    def iter_candidate_images(self, photo_rgb):
        # Debug mode: send the full photo directly into the OCR model once,
        # without rectification, orientation variants, footer crops, or extra
        # preprocessing.
        return [("photo/native/full/raw", photo_rgb)]

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

    def save_attempted_variants(self, debug_variant_dir, attempted_images):
        if debug_variant_dir is None or len(attempted_images) <= 1:
            return

        output_dir = Path(debug_variant_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, (source, image) in enumerate(attempted_images, start=1):
            filename = f"{index:03d}_{self.slugify_source(source)}.png"
            self.image_processor.save_debug_image(output_dir / filename, image)

    @staticmethod
    def slugify_source(source):
        slug = source.replace("/", "__")
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)
        return slug.strip("_")
