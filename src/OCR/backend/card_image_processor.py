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
        with Image.open(image_path) as image:
            return np.array(image.convert("RGB"))

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

        contour = max(contours, key=cv2.contourArea)
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        source = self.order_quad(box)

        width = int(max(np.linalg.norm(source[1] - source[0]), np.linalg.norm(source[2] - source[3])))
        height = int(max(np.linalg.norm(source[3] - source[0]), np.linalg.norm(source[2] - source[1])))
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        return cv2.warpPerspective(photo_rgb, matrix, (width, height))
