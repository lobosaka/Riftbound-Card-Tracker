"""Image preparation utilities for the card OCR pipeline.

This module prepares raw card photos for OCR. It first loads images into a
consistent RGB NumPy array so the rest of the pipeline always receives the same
input format. It then tries to detect the card inside the full photo and remove
unnecessary background by rectifying the card into a flat, front-facing view.

Card detection works by converting the image from RGB to HSV, because darkness
and saturation are easier to threshold there than in RGB. Pixels that are dark
enough or saturated enough are kept in a binary mask, which is used as a rough
foreground estimate for the card. Morphological closing fills small holes and
connects gaps in that mask so the card region becomes a more solid shape.

After that, the processor finds contours in the mask, assumes the largest
contour is the card, fits a rotated rectangle around it, and extracts four
corner points. Those corners are passed into OpenCV's perspective transform so
the card can be warped into a normalized view. Later OCR steps can then crop
stable footer regions, try multiple preprocessing variants, and read the card
code more reliably.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class NamedImage:
    def __init__(self, name, image):
        self.name = name
        self.image = image


class CardImageProcessor:
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    def choose_image_dir(self, current_file):
        candidates = [
            current_file.parent / "debug_steps" / "images",
            current_file.parent / "images",
            current_file.parent.parent / "debug_steps" / "images",
        ]

        for directory in candidates:
            if directory.exists():
                return directory
        return candidates[0]

    def load_rgb_image(self, image_path):
        """Load the input file into a normalized RGB NumPy array."""
        with Image.open(image_path) as image:
            return np.array(image.convert("RGB"))

    def export_card_debug(self, image_path, output_dir):
        """Save card-extraction debug artifacts for one source image."""
        image_file = Path(image_path).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        photo_rgb = self.load_rgb_image(image_file)
        self.save_debug_image(output_path / "original.png", photo_rgb)

        try:
            debug = self.extract_card_debug(photo_rgb)
        except RuntimeError as error:
            (output_path / "error.txt").write_text(f"{error}\n", encoding="utf-8")
            return {"status": "error", "message": str(error), "output_dir": str(output_path)}

        self.save_debug_image(output_path / "mask.png", debug["mask"])
        self.save_debug_image(output_path / "contour_overlay.png", debug["overlay"])
        self.save_debug_image(output_path / "rectified.png", debug["rectified"])

        return {"status": "ok", "output_dir": str(output_path)}

    def iter_image_files(self, image_dir):
        return [
            image_file
            for image_file in sorted(image_dir.iterdir())
            if image_file.is_file()
            and image_file.suffix.lower() in self.IMAGE_EXTENSIONS
            and image_file.stem.lower() != "original"
        ]

    def try_extract_card_variants(self, photo_rgb):
        variants = [NamedImage(name="photo", image=photo_rgb)]
        try:
            variants.insert(0, NamedImage(name="rectified", image=self.extract_card(photo_rgb)))
        except RuntimeError:
            pass
        return variants

    def iter_oriented_cards(self, card_variant):
        return [
            NamedImage(name="native", image=card_variant.image),
            NamedImage(name="rot180", image=cv2.rotate(card_variant.image, cv2.ROTATE_180)),
        ]

    def crop_code_regions(self, card_rgb):
        height, width = card_rgb.shape[:2]
        regions = [
            NamedImage(
                name="footer_left_tight",
                image=card_rgb[int(height * 0.90) : height, int(width * 0.01) : int(width * 0.28)],
            ),
            NamedImage(
                name="footer_left",
                image=card_rgb[int(height * 0.88) : height, int(width * 0.01) : int(width * 0.38)],
            ),
            NamedImage(
                name="footer_band",
                image=card_rgb[int(height * 0.85) : height, int(width * 0.00) : int(width * 0.55)],
            ),
            NamedImage(name="legacy_bottom_left", image=self.crop_bottom_left_code_region(card_rgb)),
        ]
        return [region for region in regions if region.image.size]

    def preprocess_variants(self, strip_rgb):
        gray = cv2.cvtColor(strip_rgb, cv2.COLOR_RGB2GRAY)
        enlarged = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.bilateralFilter(enlarged, 7, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
        sharpened = cv2.addWeighted(clahe, 1.6, cv2.GaussianBlur(clahe, (0, 0), 2), -0.6, 0)

        variants = [
            NamedImage(name="gray", image=denoised),
            NamedImage(name="clahe", image=clahe),
            NamedImage(name="sharpened", image=sharpened),
        ]

        _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(NamedImage(name="otsu", image=otsu))
        variants.append(NamedImage(name="otsu_inv", image=255 - otsu))

        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        variants.append(NamedImage(name="adaptive", image=adaptive))
        variants.append(NamedImage(name="adaptive_inv", image=255 - adaptive))
        return variants

    @staticmethod
    def crop_bottom_left_code_region(card_rgb):
        height, width = card_rgb.shape[:2]
        x0 = int(width * 0.01)
        x1 = int(width * 0.42)
        y0 = int(height * 0.86)
        return card_rgb[y0:height, x0:x1]

    @staticmethod
    def order_quad(points):
        points = points.astype(np.float32)
        summed = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)

        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = points[np.argmin(summed)]
        ordered[2] = points[np.argmax(summed)]
        ordered[1] = points[np.argmin(diffs)]
        ordered[3] = points[np.argmax(diffs)]
        return ordered

    def extract_card(self, photo_rgb):
        """Detect the card in the photo and warp it into a flat front-facing view.

        The input is converted to HSV so we can threshold dark or saturated
        pixels more reliably than in RGB. That threshold becomes a binary mask
        where likely card pixels stay white and likely background pixels turn
        black. Morphological closing then fills small holes and reconnects gaps
        before contour detection.
        """
        debug = self.extract_card_debug(photo_rgb)
        return debug["rectified"]

    def extract_card_debug(self, photo_rgb):
        """Return intermediate artifacts for the card-extraction stage."""
        hsv = cv2.cvtColor(photo_rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Sample photos show dark or saturated cards against a brighter background.
        mask = ((value < 150) | (saturation > 60)).astype(np.uint8) * 255
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("Could not detect card contour")

        # Fit a rotated rectangle around the largest contour and use its four
        # corners as the source quad for perspective correction.
        contour = max(contours, key=cv2.contourArea)
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        source = self.order_quad(box)
        overlay = self.draw_detected_card(photo_rgb, contour, source)

        width = int(max(np.linalg.norm(source[1] - source[0]), np.linalg.norm(source[2] - source[3])))
        height = int(max(np.linalg.norm(source[3] - source[0]), np.linalg.norm(source[2] - source[1])))
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        rectified = cv2.warpPerspective(photo_rgb, matrix, (width, height))
        return {
            "mask": mask,
            "overlay": overlay,
            "rectified": rectified,
        }

    @staticmethod
    def draw_detected_card(photo_rgb, contour, source):
        overlay = photo_rgb.copy()
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 6)
        cv2.polylines(overlay, [source.astype(np.int32)], True, (255, 0, 0), 6)
        for x_coord, y_coord in source.astype(np.int32):
            cv2.circle(overlay, (x_coord, y_coord), 10, (255, 255, 0), -1)
        return overlay

    @staticmethod
    def save_debug_image(image_path, image):
        image_file = Path(image_path)
        if image.ndim == 2:
            Image.fromarray(image).save(image_file)
            return
        Image.fromarray(image.astype(np.uint8), mode="RGB").save(image_file)
