import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# Create List of Card Names and Dictionary with numerical IDs
image_dir = 'data/card_images_processed'

def create_card_dict():
    card_list = sorted(os.listdir(image_dir))
    card_dict = {}

    for i, name in enumerate(card_list):
        name = os.path.splitext(name)[0]
        card_dict[name] = i

    safe_path = "data"
    os.makedirs(safe_path, exist_ok=True)
    json_path = os.path.join(safe_path, 'card_to_label.json')

    with open(json_path, mode="w", encoding='utf-8') as f:
        json.dump(card_dict, f, indent=4, ensure_ascii=False)

    print("Klassen-Mapping JSON-Datei erfolgreich erstellt!")

class SquarePadding(object):
    def __call__(self, img):
        img_rgb = img.convert('RGB')
        width, height= img_rgb.size
        max_side = max(width, height)
        # Create new symmetric picture with black pixels
        new_img = Image.new("RGB", (max_side, max_side), (0, 0, 0))
        # Calculate position of the picture from left and top to center
        left = (max_side - width) // 2
        top = (max_side - height) // 2
        # Put original picture on top of new picture with left and top margins as defined
        new_img.paste(img_rgb, (left, top))
        return new_img

# Augmentation Pipeline
def augmentation(mean, std):
    augmentation = transforms.Compose([
        # Converts to RGB and applies square padding
        SquarePadding(),
        # Random Rotation up to 15 Degrees
        transforms.RandomRotation(degrees=20),
        # +/- 40% change in contrast, brightness, +/- 30% saturation, +/- 10% hue
        transforms.ColorJitter(
            contrast=0.4,
            brightness=0.4,
            saturation=0.3,
            hue=0.1
        ),
        # Blur
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
        # Resize Picture for ResNet
        transforms.Resize((224, 224)),
        # Transformation in Tensor
        transforms.ToTensor(),
        # Normalisierung
        transforms.Normalize(mean=mean, std=std)
    ])
    return augmentation

class ContrastiveTransforms(object):
    """Returns two random augmentations of the same picture"""
    def __init__(self, base_transforms):
        self.transforms = base_transforms
    def __call__(self, x):
        img_i = self.transforms(x)
        img_j = self.transforms(x)
        return img_i, img_j

class RiftboundDataset(Dataset):
    def __init__(self, image_dir, card_dict, transforms=None):
        self.image_dir = image_dir
        self.card_dict = card_dict
        self.filenames = sorted([f for f in os.listdir(image_dir)])
        self.transforms = transforms if transforms else transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Get Filename and Image Path
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        # Open Image
        image = Image.open(img_path)
        # Apply Transformation
        image_data = self.transforms(image)
        # Get Card Name (ID) and numerical Label
        card_id = os.path.splitext(filename)[0]
        label = self.card_dict[card_id]
        # return Image(Tensor) and Label
        return image_data, label

