import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import os
from classification.backbone import (
    get_normalization_params,
    get_resnet50_model,
)
from services.dataloader import build_resnet_transform

# --- Classification Pipeline Setup ---
MODEL = get_resnet50_model(num_classes=960)
BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = (
    BASE_DIR
    / "src"
    / "classification"
    / "checkpoints"
    / "riftbound_resnet50_weights.pth"
)

if not WEIGHTS_PATH.exists():
  raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")
MODEL.load_state_dict(
    torch.load(WEIGHTS_PATH, map_location=torch.device("cpu"), weights_only=True)
)
MODEL.eval()

MEAN, STD = get_normalization_params("ResNet50")
TRANSFORMS = build_resnet_transform(MEAN, STD)

# --- Label Mapping Setup ---
CARD_DICT_REV = {}
JSON_PATH = BASE_DIR / "data" / "card_to_label.json"

if JSON_PATH.exists():
  with JSON_PATH.open("r", encoding="utf-8") as f:
    card_dict = json.load(f)
  CARD_DICT_REV = {int(v): k for k, v in card_dict.items()}


def predict_classification(image: Image.Image) -> dict:
  """Accepts a PIL Image, runs classification inference, and returns prediction dict."""
  input_tensor = TRANSFORMS(image.convert("RGB")).unsqueeze(0)

  with torch.no_grad():
    outputs = MODEL(input_tensor)
    probabilities = F.softmax(outputs[0], dim=0)

  top_prob, top_prob_idx = torch.max(probabilities, dim=0)

  confidence = float(top_prob.item() * 100)
  predicted_label_idx = top_prob_idx.item()
  card_name = CARD_DICT_REV.get(predicted_label_idx, str(predicted_label_idx))

  return {
      "method": "classification",
      "card_name": card_name,
      "confidence": round(confidence, 2),
  }


if __name__ == "__main__":
  query_image_name = "sumpworks_map1.jpeg"
  query_image_path = os.path.join("data/inference_images/", query_image_name)

  print(f"Lade Suchbild: {query_image_name}...")
  with Image.open(query_image_path) as img:
    result = predict_classification(img)

  print(
      f"Das Modell hat mit einer Sicherheit von {result['confidence']:.2f}%"
      f" die Karte {result['card_name']} vorhergesagt."
  )
  print(f"Die vorhergesagte Karte ist: {result['card_name']}")
