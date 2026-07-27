import logging

import numpy as np
from .processing.code_parser import CardCodeParser, CardRepository
from ocr.processing.ocr_engine import run_ocr

# Entry point from api
def process_uploaded_image(image, filename=None, logger=None):
    active_logger = logger or logging.getLogger(__name__)
    active_logger.info(
        "Starting OCR pipeline for image '%s'.",
        filename or "<unknown>",
    )

    # Convert the uploaded image to RGB format for processing
    photo_rgb = np.array(image.convert("RGB"))
    active_logger.debug(
        "Converted image '%s' to RGB numpy array with shape %s and dtype %s.",
        filename or "<unknown>",
        photo_rgb.shape,
        photo_rgb.dtype,
    )
    # Build scanner with CardRepository and CardCodeParser
    repository = CardRepository(logger=active_logger)
    parser = CardCodeParser(repository, logger=active_logger)
    # Run OCR on the provided image and orchestrate candidate extraction and ranking
    observations = run_ocr(photo_rgb, logger=active_logger)
    lines = [str(item.get("text", "")) for item in observations if item.get("text")]
    # Run the parser to extract candidates and score them
    result = parser.main(lines)
    candidates = result["candidates"]
    code = result["best_candidate"]
    active_logger.debug(
        "OCR candidate list for '%s': %s",
        filename or "<unknown>",
        candidates,
    )
    active_logger.info(
        "Finished OCR pipeline for image '%s' with %d observations, %d candidates, best code '%s'.",
        filename or "<unknown>",
        len(observations),
        len(candidates),
        code or "",
    )

    return {
        "image_path": filename,
        "code": code,
        "candidates": [
            {"source": "photo/native/full/raw", "code": code}
            for code in candidates
        ],
    }
