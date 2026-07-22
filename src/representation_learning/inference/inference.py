import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from src.representation_learning.dataset.dataloader import SquarePadding
from src.representation_learning.models.backbone import (
    ResNet,
    get_normalization_params,
)


def build_gallery(model, image_dir, transform):
  gallery_embeddings, gallery_names = [], []
  all_files = sorted([f for f in os.listdir(image_dir)])

  print(f"Erstelle Galerie aus {len(all_files)} Referenzkarten...")

  with torch.no_grad():
    for filename in all_files:
      img_path = os.path.join(image_dir, filename)
      with Image.open(img_path) as img:
        img_tensor = transform(img.convert("RGB")).unsqueeze(0)
        features = model.backbone(img_tensor)
        features = F.normalize(features, p=2, dim=1)

      gallery_embeddings.append(features)
      gallery_names.append(os.path.splitext(filename)[0])

    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)

  return gallery_embeddings, gallery_names


RL_MODEL = ResNet()
RL_WEIGHTS_PATH = r"src/representation_learning/models/checkpoints/riftbound_resnet50_weights_RL.pth"

if os.path.exists(RL_WEIGHTS_PATH):
  RL_MODEL.load_state_dict(
      torch.load(RL_WEIGHTS_PATH, map_location=torch.device("cpu"))
  )
RL_MODEL.eval()

RL_MEAN, RL_STD = get_normalization_params("ResNet50")
RL_TRANSFORMS = transforms.Compose([
    SquarePadding(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=RL_MEAN, std=RL_STD),
])

CACHE_PATH = "data/gallery_cache.pt"
GALLERY_DIR = "data/card_images"

if os.path.exists(CACHE_PATH):
  cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
  RL_GALLERY_EMBEDDINGS = cache["embeddings"]
  RL_GALLERY_NAMES = cache["names"]
  print("Gallery successfully loaded from disk!")
else:
  RL_GALLERY_EMBEDDINGS, RL_GALLERY_NAMES = build_gallery(
      RL_MODEL, GALLERY_DIR, RL_TRANSFORMS
  )
  torch.save(
      {"embeddings": RL_GALLERY_EMBEDDINGS, "names": RL_GALLERY_NAMES},
      CACHE_PATH,
  )
  print(f"Gallery saved to {CACHE_PATH}")


def predict_representation_learning(image: Image.Image) -> dict:
  input_tensor = RL_TRANSFORMS(image.convert("RGB")).unsqueeze(0)

  with torch.no_grad():
    query_embedding = RL_MODEL.backbone(input_tensor)
    query_embedding = F.normalize(query_embedding, p=2, dim=1)

  similarities = torch.matmul(
      RL_GALLERY_EMBEDDINGS, query_embedding.T
  ).squeeze(1)
  best_score, best_idx = torch.max(similarities, dim=0)

  predicted_name = RL_GALLERY_NAMES[best_idx.item()]
  similarity_percentage = float(best_score.item() * 100)

  return {
      "method": "representation_learning",
      "card_name": predicted_name,
      "confidence": round(similarity_percentage, 2),
  }


if __name__ == '__main__':
  query_image_name = 'Bild_1.jpeg'
  query_image_path = os.path.join('data/inference_images/', query_image_name)

  print(f'Lade Suchbild: {query_image_name}...')
  with Image.open(query_image_path) as query_img:
    result = predict_representation_learning(query_img)

  print(f"Gefundene Karte: {result['card_name']}")
  print(f"Ähnlichkeit: {result['similarity']:.2f}%")