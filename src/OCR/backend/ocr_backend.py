import os
import shutil
from pathlib import Path
from functools import lru_cache

import numpy as np

WORKSPACE_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache"
PADDLEX_CACHE_DIR = WORKSPACE_CACHE_DIR / "paddlex"
EASYOCR_CACHE_DIR = WORKSPACE_CACHE_DIR / "easyocr"
EASYOCR_USER_NETWORK_DIR = EASYOCR_CACHE_DIR / "user_network"
MPL_CACHE_DIR = WORKSPACE_CACHE_DIR / "matplotlib"
FONTCONFIG_CACHE_DIR = WORKSPACE_CACHE_DIR / "fontconfig"

for cache_dir in (
    PADDLEX_CACHE_DIR,
    EASYOCR_CACHE_DIR,
    EASYOCR_USER_NETWORK_DIR,
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
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None

try:
    import easyocr
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None

EASYOCR_DEFAULT_MODELS = {
    "detector": {
        "filename": "craft_mlt_25k.pth",
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
    },
    "recognizer": {
        "filename": "english_g2.pth",
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
    },
}


@lru_cache(maxsize=1)
def get_paddle_ocr():
    if PaddleOCR is None:
        return None

    init_variants = (
        {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "lang": "en",
        },
        {
            "use_angle_cls": False,
            "lang": "en",
        },
        {
            "lang": "en",
        },
    )

    last_error = None
    for kwargs in init_variants:
        try:
            return PaddleOCR(**kwargs)
        except TypeError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    return None


def _extract_text_conf_box(item):
    if isinstance(item, dict):
        text = item.get("text")
        if not text:
            return None
        confidence = float(item.get("confidence", 0.0))
        box = item.get("box")
        return str(text), confidence, box

    if isinstance(item, (list, tuple)):
        if len(item) >= 2 and isinstance(item[1], (list, tuple)) and item[1]:
            maybe_text = item[1][0]
            if maybe_text:
                confidence = float(item[1][1]) if len(item[1]) > 1 else 0.0
                box = item[0]
                return str(maybe_text), confidence, box
        if item and isinstance(item[0], str) and (
            len(item) == 1 or isinstance(item[1], (int, float))
        ):
            confidence = float(item[1]) if len(item) > 1 else 0.0
            box = item[2] if len(item) > 2 else None
            return str(item[0]), confidence, box

    return None


def _ensure_rgb_array(image):
    image_array = np.array(image)
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    return image_array


def parse_paddle_result(result):
    observations = []

    def visit(node):
        parsed = _extract_text_conf_box(node)
        if parsed is not None:
            text, confidence, box = parsed
            observations.append({"text": text, "confidence": confidence, "box": box})
            return

        if isinstance(node, dict):
            for value in node.values():
                visit(value)
            return

        if isinstance(node, (list, tuple)):
            for value in node:
                visit(value)

    visit(result)
    return observations


def run_paddle_ocr(image):
    engine = get_paddle_ocr()
    if engine is None:
        return []

    image_array = _ensure_rgb_array(image)

    # PaddleOCR 3.x routes `ocr()` through `predict()` and no longer accepts
    # the legacy `cls` keyword. Try the current API first, then fall back for
    # older installs that still expect the 2.x call shape.
    try:
        result = engine.predict(
            image_array,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        result = engine.ocr(image_array, cls=False)

    return parse_paddle_result(result)


@lru_cache(maxsize=1)
def get_easyocr_reader():
    if easyocr is None:
        return None

    missing = []
    for model in EASYOCR_DEFAULT_MODELS.values():
        model_path = EASYOCR_CACHE_DIR / model["filename"]
        if not model_path.exists():
            missing.append(f"{model['filename']} <- {model['url']}")

    if missing:
        missing_text = "; ".join(missing)
        raise FileNotFoundError(
            "EasyOCR model files are missing. Download these files and place "
            f"them in {EASYOCR_CACHE_DIR}: {missing_text}"
        )

    return easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(EASYOCR_CACHE_DIR),
        user_network_directory=str(EASYOCR_USER_NETWORK_DIR),
        download_enabled=False,
    )


def run_easyocr(image):
    reader = get_easyocr_reader()
    if reader is None:
        return []

    image_array = _ensure_rgb_array(image)
    results = reader.readtext(image_array, detail=1, paragraph=False)

    observations = []
    for item in results:
        if len(item) < 3:
            continue
        box, text, confidence = item[:3]
        if not text:
            continue
        observations.append(
            {
                "text": str(text),
                "confidence": float(confidence),
                "box": box,
            }
        )
    return observations


def _tesseract_binary_available():
    return shutil.which("tesseract") is not None


def run_tesseract_ocr(image):
    if pytesseract is None or not _tesseract_binary_available():
        return []

    image_array = _ensure_rgb_array(image)
    data = pytesseract.image_to_data(
        image_array,
        output_type=pytesseract.Output.DICT,
        config="--psm 6",
    )

    observations = []
    for index, text in enumerate(data.get("text", [])):
        if not text or not text.strip():
            continue

        confidence_raw = data.get("conf", [])[index]
        try:
            confidence = max(float(confidence_raw), 0.0) / 100.0
        except (TypeError, ValueError):
            confidence = 0.0

        left = data.get("left", [None])[index]
        top = data.get("top", [None])[index]
        width = data.get("width", [None])[index]
        height = data.get("height", [None])[index]

        box = None
        if None not in (left, top, width, height):
            box = [
                [left, top],
                [left + width, top],
                [left + width, top + height],
                [left, top + height],
            ]

        observations.append(
            {
                "text": text.strip(),
                "confidence": confidence,
                "box": box,
            }
        )
    return observations


def run_ocr(image):
    backends = (
        ("easyocr", run_easyocr),
        ("tesseract", run_tesseract_ocr),
    )

    last_error = None
    for _, runner in backends:
        try:
            results = runner(image)
        except Exception as error:
            last_error = error
            continue

        if results:
            return results

    if last_error is not None:
        raise last_error
    return []


def describe_ocr_backend():
    return {
        "paddle_available": False,
        "easyocr_available": easyocr is not None,
        "tesseract_available": pytesseract is not None and _tesseract_binary_available(),
        "backend_order": ["easyocr", "tesseract"],
        "easyocr_model_dir": str(EASYOCR_CACHE_DIR),
        "easyocr_required_models": EASYOCR_DEFAULT_MODELS,
    }
