import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import json
from PIL import Image
# Own Modules
from src.models.backbone import get_resnet50_model, get_normalization_params
from src.dataset.dataloader import SquarePadding

# Get ResNet50 Model
model = get_resnet50_model(num_classes=960)
# Load trained Model Weights
model.load_state_dict(torch.load(r'src/models/checkpoints/riftbound_resnet50_weights.pth', map_location=torch.device('cpu')))
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

image = 'Bild_1.jpeg'
image_path = os.path.join('data/inference_images/', image)
with Image.open(image_path) as img:
    input_tensor = inference_transforms(img)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    outputs = model(input_batch)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

top_prob, top_prob_idx = torch.max(probabilities, dim=0)

confidence = top_prob.item() * 100
predicted_label_idx = top_prob_idx.item()

print(f"Das Modell hat mit einer Sicherheit von {confidence:.2f}% das Label {predicted_label_idx} vorhergesagt.")

# Load JSON file and create dict
json_path = "data/card_to_label.json"
with open(json_path, 'r', encoding='utf-8') as f:
    card_dict = json.load(f)
# Reverse the Dictionary to get Name as Value for idx
card_dict_rev = {v: k for k, v in card_dict.items()}
card_id = card_dict_rev[predicted_label_idx]

print(f"Die Vorhergesagte Karte ist: {card_id}")

