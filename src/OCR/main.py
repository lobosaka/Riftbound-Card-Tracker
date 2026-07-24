import numpy as np
from .backend.card_codes import CardCodeParser, CardRepository
from OCR.backend.ocr_backend import run_ocr

# Entry point from api
def process_uploaded_image(image, filename=None):
    # Convert the uploaded image to RGB format for processing
    photo_rgb = np.array(image.convert("RGB"))
    # Build scanner with CardRepository and CardCodeParser
    repository = CardRepository()
    parser = CardCodeParser(repository)
    # Run OCR on the provided image and orchestrate candidate extraction and ranking
    observations = run_ocr(photo_rgb)
    lines = [str(item.get("text", "")) for item in observations if item.get("text")]
    # Run the parser to extract candidates and score them
    result = parser.main(lines)
    scan_result = {
        "observations": observations,
        "lines": lines,
        "candidates": result["candidates"],
        "scored_candidates": result["scored_candidates"],
        "best_candidate": result["best_candidate"],
    }

    candidates = scan_result["candidates"]
    code = scan_result["best_candidate"]

    return {
        "image_path": filename,
        "code": code,
        "candidates": [
        {"source": "photo/native/full/raw", "code": code}
        for code in candidates
    ],
    }
