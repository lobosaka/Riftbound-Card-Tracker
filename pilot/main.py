from PIL import Image
import pytesseract  # type: ignore[import-not-found]
import cv2


IMAGE_PATH = "input2.png"


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

    text = pytesseract.image_to_string(image, config=config)
    return text


def main():
    processed_image = preprocess_image(IMAGE_PATH)

    text = run_ocr(processed_image)

    print("Extracted text:")
    print("----------------")
    print(text)


if __name__ == "__main__":
    main()