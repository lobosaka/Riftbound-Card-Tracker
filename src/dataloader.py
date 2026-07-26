import json
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def create_card_dict(
    image_dir: str = "data/card_images_processed",
    output_path: str = "data/card_to_label.json",
) -> dict[str, int]:
    card_list = sorted(os.listdir(image_dir))
    card_dict: dict[str, int] = {}

    for index, name in enumerate(card_list):
        card_name = os.path.splitext(name)[0]
        card_dict[card_name] = index

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode="w", encoding="utf-8") as file_handle:
        json.dump(card_dict, file_handle, indent=4, ensure_ascii=False)

    print("Klassen-Mapping JSON-Datei erfolgreich erstellt!")
    return card_dict


class SquarePadding(object):
    def __call__(self, img):
        img_rgb = img.convert("RGB")
        width, height = img_rgb.size
        max_side = max(width, height)
        new_img = Image.new("RGB", (max_side, max_side), (0, 0, 0))
        left = (max_side - width) // 2
        top = (max_side - height) // 2
        new_img.paste(img_rgb, (left, top))
        return new_img


def build_resnet_transform(mean, std):
    return transforms.Compose([
        SquarePadding(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_training_augmentation(mean, std):
    return transforms.Compose([
        SquarePadding(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            contrast=0.4,
            brightness=0.4,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

class RiftboundDataset(Dataset):
    """
    Dataset for Riftbound card images. Supports loading all files directly from 
    a directory or loading subset filenames from a JSON split file.
    """
    def __init__(
        self,
        image_dir: str,
        card_dict: dict,
        json_path: str = None,
        split: str = None,
        transform=None,
        transforms=None,
    ):
        self.image_dir = image_dir
        self.card_dict = card_dict
        
        # Standardize transform parameter handling
        self.transform = transform or transforms or transforms.ToTensor()

        # Load filenames either from a split file or directly from directory
        if json_path and split:
            with open(json_path, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            self.filenames = split_data[split]
        elif json_path and not split:
            with open(json_path, "r", encoding="utf-8") as f:
                self.filenames = json.load(f)
        else:
            self.filenames = sorted([
                f for f in os.listdir(image_dir) 
                if os.path.isfile(os.path.join(image_dir, f))
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.image_dir, filename)
        
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        card_id = os.path.splitext(filename)[0]
        label = self.card_dict[card_id]
        
        return image, label
   
class ContrastiveTransforms(object):
    """Returns two random augmentations of the same picture"""
    def __init__(self, base_transforms):
        self.transforms = base_transforms
    def __call__(self, x):
        img_i = self.transforms(x)
        img_j = self.transforms(x)
        return img_i, img_j
