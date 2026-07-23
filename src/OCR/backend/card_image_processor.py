"""Image preparation utilities for the card OCR pipeline.

This module prepares raw card photos for OCR. The current production pipeline
only needs one responsibility here: load input files into a consistent RGB
NumPy array so the downstream OCR backend always receives the same format.
"""

import numpy as np
from PIL import Image


class CardImageProcessor:
    def load_rgb_image(self, image_path):
        """Load the input file into a normalized RGB NumPy array."""
        with Image.open(image_path) as image:
            return np.array(image.convert("RGB"))
