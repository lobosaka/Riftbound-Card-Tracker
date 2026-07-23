import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# Create List of Card Names and Dictionary with numerical IDs
image_dir = 'data/card_images_processed'

def create_card_dict():
    card_list = os.listdir(image_dir)
    card_dict = {}

    for i, name in enumerate(card_list):
        name = os.path.splitext(name)[0]
        card_dict[name] = i
    # Create reverse Card Name Dictionary
    rev_card_dict = {v: k for k, v in card_dict.items()}

    return card_dict, rev_card_dict

# Augmentation Pipeline
augmentation = transforms.Compose([
    # Random Rotation up to 15 Degrees
    transforms.RandomRotation(degrees=15),
    # +/- 20% change in contrast, brightness, saturation
    transforms.ColorJitter(
        contrast=0.2,
        brightness=0.2,
        saturation=0.2
    ),
    # Blur
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    # Transformation in Tensor

    # NOTIZ AN MICH: Was ist mit Normalisierung für bestimmte Netzwerke, wie ResNet50?

    transforms.ToTensor()
])

class RiftboundDataset(Dataset):
    def __init__(self, image_dir, card_dict, transforms=None):
        self.image_dir = image_dir
        self.card_dict = card_dict
        self.filenames = [f for f in os.listdir(image_dir)]
        self.transforms = transforms if transforms else transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Get Filename and Image Path
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        # Open Image
        image = Image.open(img_path)
        # Get Card Name (ID) and numerical Label
        card_id = os.path.splitext(filename)[0]
        label = self.card_dict[card_id]
        # Apply Transformation
        if self.transforms:
            image = self.transforms(image)
        # return Image(Tensor) and Label
        return image, label
    
if __name__ == '__main__':
    card_dict, rev_card_dict = create_card_dict()
