import os
from pathlib import Path
from functools import lru_cache

import numpy as np

WORKSPACE_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache"
PADDLEX_CACHE_DIR = WORKSPACE_CACHE_DIR / "paddlex"
MPL_CACHE_DIR = WORKSPACE_CACHE_DIR / "matplotlib"
FONTCONFIG_CACHE_DIR = WORKSPACE_CACHE_DIR / "fontconfig"

for cache_dir in (
    PADDLEX_CACHE_DIR,
    MPL_CACHE_DIR,
    FONTCONFIG_CACHE_DIR,
):
    cache_dir.mkdir(parents=True, exist_ok=True)

# PaddleX and matplotlib default to locations outside the writable workspace in
# this environment, so redirect their caches into the repository.
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLEX_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(WORKSPACE_CACHE_DIR))

try:
    from rapidocr import RapidOCR
except ImportError:  # pragma: no cover - optional dependency
    RapidOCR = None


def _ensure_rgb_array(image):
    image_array = np.array(image)
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    return image_array


@lru_cache(maxsize=1)
def get_rapidocr():
    if RapidOCR is None:
        return None
    return RapidOCR()


def run_ocr(image):
    engine = get_rapidocr()
    if engine is None:
        return []

    image_array = _ensure_rgb_array(image)
    result = engine(image_array)

    observations = []
    txts = list(getattr(result, "txts", ()) or ())
    scores = list(getattr(result, "scores", ()) or ())
    boxes = getattr(result, "boxes", None)

    for index, text in enumerate(txts):
        if not text:
            continue
        confidence = float(scores[index]) if index < len(scores) else 0.0
        box = boxes[index].tolist() if boxes is not None and index < len(boxes) else None
        observations.append(
            {
                "text": str(text),
                "confidence": confidence,
                "box": box,
            }
        )

    return observations


def describe_ocr_backend():
    return {
        "paddle_available": False,
        "easyocr_available": False,
        "tesseract_available": False,
        "rapidocr_available": RapidOCR is not None,
        "backend_order": ["rapidocr"],
    }
