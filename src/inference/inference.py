import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
# Own Modules
from src.models.backbone import get_resnet50_model, get_normalization_params
from src.trans.transform import padding
from src.dataset.dataloader import create_card_dict

# Get ResNet50 Model
model = get_resnet50_model(num_classes=960)
# Load trained Model Weights
model.load_state_dict(torch.load(r'src/models/checkpoints/riftbound_resnet50_weights_100_epochs.pth', map_location=torch.device('cpu')))
# Set Model into eval-mode
model.eval()
# Transformation for Inference
mean, std = get_normalization_params('ResNet50')
inference_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

image = 'Bild_3.jpeg'
image_path = os.path.join('data/inference_images/', image)
padded_image = padding(image_path)
input_tensor = inference_transforms(padded_image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    outputs = model(input_batch)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

top_prob, top_prob_idx = torch.max(probabilities, dim=0)

confidence = top_prob.item() * 100
predicted_label_idx = top_prob_idx.item()

print(f"Das Modell hat mit einer Sicherheit von {confidence:.2f}% das Label {predicted_label_idx} vorhergesagt.")

_, card_dict = create_card_dict()
card_label = card_dict[predicted_label_idx]

print(f"Die Vorhergesagte Karte ist: {card_label}")

