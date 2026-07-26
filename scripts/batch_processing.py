from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from classification.inference import predict_classification
from representation_learning.inference import predict_representation_learning
from OCR.card_codes import CardRepository
from OCR.main import process_uploaded_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Batch process a directory of images with OCR, classification, or "
			"representation learning and copy the images into a renamed output dir."
		)
	)
	parser.add_argument("input_dir", help="Directory containing images to process.")
	parser.add_argument(
		"--output-dir",
		default="data/batch_outputs",
		help="Directory where renamed images will be written.",
	)
	parser.add_argument(
		"--method",
		choices=("ocr", "classification", "representation_learning"),
		required=True,
		help="Inference method to use.",
	)
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="Process images recursively under the input directory.",
	)
	return parser.parse_args()


def iter_image_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
	iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
	for path in sorted(iterator):
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
			yield path


def predict_image_id(image_path: Path, method: str) -> str:
	if method == "classification":

		with Image.open(image_path) as image:
			result = predict_classification(image)
		return (result.get("card_name") or image_path.stem).strip()

	if method == "representation_learning":
		

		with Image.open(image_path) as image:
			result = predict_representation_learning(image)
		return (result.get("card_name") or image_path.stem).strip()

	with Image.open(image_path) as image:
		ocr_result = process_uploaded_image(image, filename=image_path.name)
	code = str(ocr_result.get("code") or "").strip()
	if not code:
		return image_path.stem

	repository = CardRepository()
	matches = repository.verify_code(code)
	if len(matches) == 1:
		return matches[0].id

	if len(matches) > 1:
		return matches[0].id

	return code


def unique_output_path(output_dir: Path, image_id: str, suffix: str) -> Path:
	candidate = output_dir / f"{image_id}{suffix}"
	counter = 2
	while candidate.exists():
		candidate = output_dir / f"{image_id}__{counter}{suffix}"
		counter += 1
	return candidate
def main() -> int:
	args = parse_args()

	input_dir = Path(args.input_dir).expanduser().resolve()
	if not input_dir.exists() or not input_dir.is_dir():
		raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

	output_root = Path(args.output_dir).expanduser().resolve()
	output_dir = output_root / args.method
	output_dir.mkdir(parents=True, exist_ok=True)

	image_paths = list(iter_image_files(input_dir, args.recursive))
	if not image_paths:
		print(f"No images found in {input_dir}")
		return 0

	processed = 0
	for image_path in image_paths:
		try:
			image_id = predict_image_id(image_path, args.method)
			target_path = unique_output_path(output_dir, image_id, image_path.suffix.lower())
			shutil.copy2(image_path, target_path)
			processed += 1
			print(f"{image_path.name} -> {target_path.name}")
		except Exception as exc:
			print(f"Skipping {image_path.name}: {exc}")

	print(f"Processed {processed}/{len(image_paths)} images into {output_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
