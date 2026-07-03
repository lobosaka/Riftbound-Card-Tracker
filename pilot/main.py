from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from paddleocr import run_ocr


_CANDIDATE_DIRS = [
    Path(__file__).parent / "debug_steps" / "images",
    Path(__file__).parent / "images",
    Path(__file__).parent.parent / "debug_steps" / "images",
]

for _d in _CANDIDATE_DIRS:
    if _d.exists():
        IMAGE_DIR = _d
        break
else:
    IMAGE_DIR = _CANDIDATE_DIRS[0]


SET_CODE_PATTERN = re.compile(r"^[A-Z]{3}$")
FULL_CODE_PATTERN = re.compile(
    r"(?P<set>[A-Z]{3})[-\s•.]*?(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)[/\|](?P<total>\d{3})"
)
SUFFIXLESS_CODE_PATTERN = re.compile(r"(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)[/\|](?P<total>\d{3})")
SHORT_CODE_PATTERN = re.compile(r"[A-Z]\d{2,3}")
CANONICAL_CODE_PATTERN = re.compile(
    r"^(?:(?P<set>[A-Z]{3})-?)?(?P<number>\d{1,3})(?P<suffix>[A-Z*]?)/(?P<total>\d{3})$"
)
DB_PATH = Path(__file__).resolve().parents[1] / "riftbound.db"


def canonicalize_code(code: str) -> str:
    cleaned = code.upper().strip().replace(" ", "")
    cleaned = cleaned.replace("•", "")
    cleaned = cleaned.replace("|", "/")

    match = CANONICAL_CODE_PATTERN.fullmatch(cleaned)
    if not match:
        return cleaned

    set_code = match.group("set")
    number = int(match.group("number"))
    suffix = match.group("suffix")
    total = match.group("total")
    if set_code:
        return f"{set_code}-{number:03d}{suffix}/{total}"
    return f"{number:03d}{suffix}/{total}"


def load_rgb_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.array(image.convert("RGB"))


def fetch_card_code_rows() -> list[tuple[str, str | None, str | None, str | None]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, publicCode, collectorNumber FROM cards")
    rows = cursor.fetchall()
    conn.close()
    return rows


def build_card_code_index() -> dict[str, list[dict[str, str | None]]]:
    index: dict[str, list[dict[str, str | None]]] = {}
    for card_id, name, public_code, collector_number in fetch_card_code_rows():
        if not public_code:
            continue

        canonical_public_code = canonicalize_code(public_code)
        variants = {canonical_public_code}
        suffix = public_code.split("-", 1)[1] if "-" in public_code else public_code
        variants.add(canonicalize_code(suffix))
        if collector_number and "/" in suffix:
            variants.add(canonicalize_code(f"{collector_number}/{suffix.split('/', 1)[1]}"))
        if "-" in canonical_public_code:
            set_code = canonical_public_code.split("-", 1)[0]
            variants.add(set_code)

        card = {
            "id": card_id,
            "name": name,
            "public_code": public_code,
        }
        for variant in variants:
            index.setdefault(variant, []).append(card)
    return index


CARD_CODE_INDEX = build_card_code_index()
KNOWN_SET_CODES = {code for code in CARD_CODE_INDEX if SET_CODE_PATTERN.fullmatch(code)}


def order_quad(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    summed = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(summed)]
    ordered[2] = points[np.argmax(summed)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def extract_card(photo_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(photo_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Your sample photos have dark/saturated cards on a light wall.
    mask = ((value < 150) | (saturation > 60)).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Could not detect card contour")

    contour = max(contours, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    source = order_quad(box)

    width = int(max(np.linalg.norm(source[1] - source[0]), np.linalg.norm(source[2] - source[3])))
    height = int(max(np.linalg.norm(source[3] - source[0]), np.linalg.norm(source[2] - source[1])))
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(photo_rgb, matrix, (width, height))


def crop_bottom_left_code_region(card_rgb: np.ndarray) -> np.ndarray:
    height, width = card_rgb.shape[:2]
    x0 = int(width * 0.01)
    x1 = int(width * 0.42)
    y0 = int(height * 0.86)
    y1 = height
    return card_rgb[y0:y1, x0:x1]


def crop_code_regions(card_rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    height, width = card_rgb.shape[:2]
    regions = [
        (
            "footer_left_tight",
            card_rgb[int(height * 0.90) : height, int(width * 0.01) : int(width * 0.28)],
        ),
        (
            "footer_left",
            card_rgb[int(height * 0.88) : height, int(width * 0.01) : int(width * 0.38)],
        ),
        (
            "footer_band",
            card_rgb[int(height * 0.85) : height, int(width * 0.00) : int(width * 0.55)],
        ),
        ("legacy_bottom_left", crop_bottom_left_code_region(card_rgb)),
    ]
    return [(name, region) for name, region in regions if region.size]


def preprocess_variants(strip_rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2GRAY)
    enlarged = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(enlarged, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    sharpened = cv2.addWeighted(clahe, 1.6, cv2.GaussianBlur(clahe, (0, 0), 2), -0.6, 0)

    variants: list[tuple[str, np.ndarray]] = [
        ("gray", denoised),
        ("clahe", clahe),
        ("sharpened", sharpened),
    ]

    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))
    variants.append(("otsu_inv", 255 - otsu))

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    variants.append(("adaptive", adaptive))
    variants.append(("adaptive_inv", 255 - adaptive))
    return variants


def normalize_text(text: str) -> str:
    cleaned = text.upper().strip()
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("\n", "")
    cleaned = cleaned.replace("|", "/")
    cleaned = cleaned.replace("\\", "/")
    cleaned = cleaned.replace("•", "")
    cleaned = cleaned.replace("·", "")
    cleaned = cleaned.replace(".", "")
    cleaned = cleaned.replace(":", "")
    cleaned = cleaned.replace("I", "1")
    cleaned = cleaned.replace("L", "1")
    cleaned = cleaned.replace("O", "0")
    cleaned = cleaned.replace("Q", "0")
    cleaned = cleaned.replace("X", "*")
    return cleaned


def normalize_candidate(text: str) -> str:
    cleaned = normalize_text(text)

    full_match = FULL_CODE_PATTERN.search(cleaned)
    if full_match:
        return canonicalize_code(full_match.group(0))

    suffixless_match = SUFFIXLESS_CODE_PATTERN.search(cleaned)
    if suffixless_match:
        return canonicalize_code(suffixless_match.group(0))

    short_match = SHORT_CODE_PATTERN.search(cleaned)
    return short_match.group(0) if short_match else cleaned


def score_candidate(candidate: str) -> tuple[int, int, int]:
    db_matches = len(verify_code_against_db(candidate))
    if db_matches == 1:
        db_score = 3
    elif db_matches > 1:
        db_score = 1
    else:
        db_score = 0

    if CANONICAL_CODE_PATTERN.fullmatch(candidate) and "-" in candidate:
        match_score = 4
    elif CANONICAL_CODE_PATTERN.fullmatch(candidate):
        match_score = 3
    elif re.fullmatch(r"[A-Z]\d{2}", candidate):
        match_score = 2
    elif SHORT_CODE_PATTERN.fullmatch(candidate):
        match_score = 1
    else:
        match_score = 0
    digits = sum(char.isdigit() for char in candidate)
    star_bonus = 1 if "*" in candidate else 0
    penalty = sum(char not in "0123456789/*ABCDEFGHIJKLMNOPQRSTUVWXYZ" for char in candidate)
    return (db_score, match_score, star_bonus, digits, -penalty)


def extract_candidates_from_ocr_lines(lines: list[str]) -> list[str]:
    normalized_lines = [normalize_text(line) for line in lines if line]
    joined = " ".join(normalized_lines)
    candidates: list[str] = []

    for text in [*normalized_lines, joined]:
        for match in FULL_CODE_PATTERN.finditer(text):
            set_code = match.group("set")
            if set_code in KNOWN_SET_CODES:
                candidates.append(canonicalize_code(match.group(0)))
        for match in SUFFIXLESS_CODE_PATTERN.finditer(text):
            candidates.append(canonicalize_code(match.group(0)))
        if text in KNOWN_SET_CODES:
            candidates.append(text)

    if not candidates:
        candidates.extend(normalize_candidate(line) for line in normalized_lines)

    seen: set[str] = set()
    unique_candidates: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def run_code_ocr(image: np.ndarray) -> list[str]:
    observations = run_ocr(image)
    lines = [str(item.get("text", "")) for item in observations if item.get("text")]
    return extract_candidates_from_ocr_lines(lines)


def verify_code_against_db(code: str) -> list[dict[str, str | None]]:
    canonical = canonicalize_code(code)
    matches = CARD_CODE_INDEX.get(canonical, [])
    if matches:
        return matches

    if "-" in canonical:
        return CARD_CODE_INDEX.get(canonical.split("-", 1)[1], [])

    return matches


def try_extract_card(photo_rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants = [("photo", photo_rgb)]
    try:
        variants.insert(0, ("rectified", extract_card(photo_rgb)))
    except RuntimeError:
        pass
    return variants


def is_unique_db_match(candidate: str) -> bool:
    return len(verify_code_against_db(candidate)) == 1


def extract_code_from_photo(image_path: str) -> tuple[str, list[tuple[str, str]]]:
    photo = load_rgb_image(image_path)
    candidates: list[tuple[str, str]] = []
    for card_name, card_image in try_extract_card(photo):
        for orientation_name, oriented_card in (
            ("native", card_image),
            ("rot180", cv2.rotate(card_image, cv2.ROTATE_180)),
        ):
            for region_name, region_image in crop_code_regions(oriented_card):
                for variant_name, variant_image in preprocess_variants(region_image):
                    for candidate in run_code_ocr(variant_image):
                        source_name = f"{card_name}/{orientation_name}/{region_name}/{variant_name}"
                        candidates.append((source_name, candidate))
                        if is_unique_db_match(candidate):
                            return candidate, candidates

    if not candidates:
        return "", []

    best_variant, best_candidate = max(candidates, key=lambda item: score_candidate(item[1]))
    return best_candidate, candidates


def main():
    image_dir = IMAGE_DIR
    if not image_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")

    for image_file in sorted(image_dir.iterdir()):
        if not image_file.is_file():
            continue
        if image_file.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
            continue
        if image_file.stem.lower() == "original":
            continue

        code, candidates = extract_code_from_photo(str(image_file))
        matches = verify_code_against_db(code)
        if len(matches) == 1:
            match = matches[0]
            status = f"VERIFIED -> {match['public_code']} ({match['name']})"
        elif len(matches) > 1:
            status = "AMBIGUOUS -> " + ", ".join(
                f"{match['public_code']} ({match['name']})" for match in matches
            )
        else:
            status = "NOT FOUND IN DB"

        print(f"{image_file.name}: {code} [{status}]")
        for variant_name, candidate in candidates:
            variant_matches = verify_code_against_db(candidate)
            marker = "ok" if variant_matches else "missing"
            print(f"  {variant_name}: {candidate} ({marker})")


if __name__ == "__main__":
    main()
