from PIL import Image
import pytesseract  # type: ignore[import-not-found]
import cv2
from pathlib import Path


 # Resolve images directory: prefer pilot/debug_steps/images, then pilot/images, then ../debug_steps/images
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
    # default to the first candidate (keeps original behavior and error message)
    IMAGE_DIR = _CANDIDATE_DIRS[0]


def preprocess_image(image_path: str):
    # Load image with OpenCV
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    # Pixels above 150 become white; below become black
    _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresholded


def run_ocr(image):
    # psm 6 means: assume a single uniform block of text
    # This often works well for simple demo images
    config = "--psm 6"

    # Convert OpenCV (numpy) image to PIL Image for pytesseract
    pil_image = Image.fromarray(image)

    text = pytesseract.image_to_string(pil_image, config=config)
    return text


def main():
    image_dir = IMAGE_DIR

    if not image_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")

    for image_file in sorted(image_dir.iterdir()):
        if not image_file.is_file():
            continue
        if image_file.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):  # skip non-images
            continue

        image_path = str(image_file)
        processed_image = preprocess_image(image_path)
        text = run_ocr(processed_image)
        print(f"Extracted text from {image_file.name}:")
        print("----------------")
        print(text)


if __name__ == "__main__":
    main()