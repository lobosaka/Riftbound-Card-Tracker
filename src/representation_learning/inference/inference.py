import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from PIL import Image
# Own Modules
from src.representation_learning.models.backbone import ResNet, get_normalization_params
from src.representation_learning.dataset.dataloader import SquarePadding

# Build Gallery Function
def build_gallery(model, image_dir, transform):
    gallery_embeddings = []
    gallery_names  = []

    all_files = sorted([f for f in os.listdir(image_dir)])

    print(f"Erstelle Galerie aus {len(all_files)} Referenzkarten...")

    with torch.no_grad():
        for filename in all_files:
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)

            img_tensor = transform(img).unsqueeze(0)

            features = model.backbone(img_tensor)

            features = F.normalize(features, p=2, dim=1)

            gallery_embeddings.append(features)
            card_name = os.path.splitext(filename)[0]
            gallery_names.append(card_name)

        gallery_embeddings = torch.cat(gallery_embeddings, dim=0)

    return gallery_embeddings, gallery_names

# Get ResNet50 Model
model = ResNet()
# Load trained Model Weights
model.load_state_dict(torch.load(r'src/representation_learning/models/checkpoints/riftbound_resnet50_weights_RL.pth', map_location=torch.device('cpu')))
# Set Model into eval-mode
model.eval()
# Transformation for Inference
mean, std = get_normalization_params('ResNet50')
inference_transforms = transforms.Compose([
    SquarePadding(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

gallery_dir = 'data/card_images'
query_image_name = 'sumpworks_map1.jpeg'
query_image_path = os.path.join('data/inference_images/', query_image_name)

gallery_embeddings, gallery_names = build_gallery(
    model=model,
    image_dir=gallery_dir,
    transform=inference_transforms
)

print(f"Lade Suchbild: {query_image_name}...")
with Image.open(query_image_path) as img:
    query_tensor = inference_transforms(img)

query_batch = query_tensor.unsqueeze(0)

with torch.no_grad():
    query_embedding = model.backbone(query_batch)
    query_embedding = F.normalize(query_embedding, p=2, dim=1)

similarities = torch.matmul(gallery_embeddings, query_embedding.T).squeeze()

best_score, best_idx = torch.max(similarities, dim=0)

predicted_card_name = gallery_names[best_idx.item()]
similarity_percentage = best_score.item() * 100

print(f"Gefundene Karte: {predicted_card_name}")
print(f"Ähnlichkeit: {similarity_percentage:.2f}%")
