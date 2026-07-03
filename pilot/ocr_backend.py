from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

try:
    from paddleocr import PaddleOCR  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None  # type: ignore[assignment]

try:  # pragma: no cover - macOS-only fallback
    from Foundation import NSData
    from Vision import (
        VNImageRequestHandler,
        VNRecognizeTextRequest,
        VNRequestTextRecognitionLevelAccurate,
    )
except ImportError:  # pragma: no cover - optional dependency
    NSData = None  # type: ignore[assignment]
    VNImageRequestHandler = None  # type: ignore[assignment]
    VNRecognizeTextRequest = None  # type: ignore[assignment]
    VNRequestTextRecognitionLevelAccurate = None  # type: ignore[assignment]


def pil_from_array(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(image.astype(np.uint8), mode="RGB")


@lru_cache(maxsize=1)
def get_paddle_ocr() -> Any:
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

    last_error: Exception | None = None
    for kwargs in init_variants:
        try:
            return PaddleOCR(**kwargs)
        except TypeError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    return None


def _extract_text_conf_box(item: Any) -> tuple[str, float, Any] | None:
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
        if item and isinstance(item[0], str):
            confidence = float(item[1]) if len(item) > 1 else 0.0
            box = item[2] if len(item) > 2 else None
            return str(item[0]), confidence, box

    return None


def parse_paddle_result(result: Any) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
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


def run_paddle_ocr(image: np.ndarray) -> list[dict[str, Any]]:
    engine = get_paddle_ocr()
    if engine is None:
        return []

    result = engine.ocr(np.array(image), cls=False)
    return parse_paddle_result(result)


def run_vision_ocr(image: np.ndarray) -> list[dict[str, Any]]:
    if VNRecognizeTextRequest is None or NSData is None or VNImageRequestHandler is None:
        return []

    pil_image = pil_from_array(image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    image_data = NSData.dataWithBytes_length_(payload, len(payload))

    request = VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)
    request.setRecognitionLanguages_(["en-US"])

    handler = VNImageRequestHandler.alloc().initWithData_options_(image_data, {})
    success, error = handler.performRequests_error_([request], None)
    if not success:
        raise RuntimeError(f"OCR failed: {error}")

    results: list[dict[str, Any]] = []
    for observation in request.results():
        for candidate in observation.topCandidates_(5):
            results.append(
                {
                    "text": str(candidate.string()),
                    "confidence": float(candidate.confidence()),
                    "box": None,
                }
            )
    return results


def run_ocr(image: np.ndarray) -> list[dict[str, Any]]:
    results = run_paddle_ocr(image)
    if results:
        return results
    return run_vision_ocr(image)
