import os
import time
import json
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_metric_learning import losses
import torchvision.transforms as transforms
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import faiss
import torch.nn.functional as F
# Own Modules
from src.representation_learning.models.backbone import ResNet, get_normalization_params
from src.representation_learning.dataset.dataloader import RiftboundDataset, create_card_dict, augmentation, ContrastiveTransforms, SquarePadding

# Function to extract embeddings from dataloader
def extract_embeddings(model, dataloader, device):
    model.eval()

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Feature Extraction
            features = model.backbone(images)

            # L2-Normalization
            features = F.normalize(features, p=2, dim=1)

            embeddings_list.append(features.cpu())
            labels_list.append(labels)

    # Build Tensor
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    model.train()
    return embeddings, labels

class EarlyStopping:
    def __init__(self, patience=5, delta=0.01, save_path='src/representation_learning/models/checkpoints/riftbound_resnet50_weights_RL.pth'):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        # Save Model when Score improved
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
        print(f"Validation Score improved. Model saved to {self.save_path}")

create_card_dict()

json_path = "data/card_to_label.json"
with open(json_path, 'r', encoding='utf-8') as f:
    card_dict = json.load(f)

model = "ResNet50" # <- Current Model pick
num_epochs = 200 # <- Current Number of Epochs pick
# Get Parameters for Normalization (Augmentation Pipeline)
mean, std = get_normalization_params(model)
# Transformation for Validation Gallery
inference_transforms = transforms.Compose([
    SquarePadding(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Create Dataset and Dataloader
# Datasets
train_dataset = RiftboundDataset(image_dir='data/card_images', card_dict=card_dict, split='train', transforms=ContrastiveTransforms(augmentation(mean=mean, std=std)))
val_gallery_dataset = RiftboundDataset(image_dir='data/card_images', card_dict=card_dict, split='val', transforms=inference_transforms)
val_query_dataset = RiftboundDataset(image_dir='data/card_images', card_dict=card_dict, split='val', transforms=augmentation(mean=mean, std=std))
# Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_gallery_loader = DataLoader(val_gallery_dataset, batch_size=32, shuffle=False)
val_query_loader = DataLoader(val_query_dataset, batch_size=32, shuffle=False)
# Get Model and move to Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet()
model = model.to(device)

# Fine Tuning Layers in Backbone
for param in model.backbone.layer3.parameters():
    param.requires_grad = True
for param in model.backbone.layer4.parameters():
    param.requires_grad = True
for param in model.head.parameters():
    param.requires_grad = True

# Define Loss, Optimizer, Scheduler and Calculator
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4) # pass only parameters that are not frozen (requires_grad=True)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=200, eta_min=1e-7)
criterion = losses.NTXentLoss(temperature=0.1) 
calculator = AccuracyCalculator(include=('precision_at_1',), k=1)
save_path = 'src/representation_learning/models/checkpoints/riftbound_resnet50_weights_RL.pth'
early_stopping = EarlyStopping(patience=5, save_path=save_path)

# Switch to Training Mode
model.train()

# ---TRAINING LOOP---

print("Starte Training...")

for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0

    for (images_i, images_j), labels in train_loader:
        # Move Data to Device
        images_i = images_i.to(device)
        images_j = images_j.to(device)
        labels = labels.to(device)
        # Reset Optimizer
        optimizer.zero_grad()
        # Forward Pass 
        z_i = model(images_i)
        z_j = model(images_j)
        # Combine embeddings and labels for pytorch_metric_learning
        embeddings = torch.cat([z_i, z_j], dim=0)
        batch_labels = torch.cat([labels, labels], dim=0)
        # Calculate Contrastive Loss
        loss = criterion(embeddings, batch_labels)
        # Backpropagation
        loss.backward()
        # Gradient Update
        optimizer.step()
        # Gather Loss
        running_loss += loss.item() * images_i.size(0)

    # LR Update
    scheduler.step()

    # Epoch-Statistic
    current_lr = optimizer.param_groups[0]['lr']
    epoch_loss = running_loss/len(train_dataset)
    elapsed_time = time.time() - start_time

    print(f"Epoch: {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - LR: {current_lr:.7f} - Time: {elapsed_time:.1f}")


    # Validation 
    print("Begin Validation...")

    gallery_embeddings, gallery_labels = extract_embeddings(model, val_gallery_loader, device)
    query_embeddings, query_labels = extract_embeddings(model, val_query_loader, device)

    # Accuracy Calculator
    accuracies = calculator.get_accuracy(
        query = query_embeddings,
        reference = gallery_embeddings,
        query_labels = query_labels,
        reference_labels = gallery_labels,
        ref_includes_query=False
    )

    # Extract Precision@1 as percentage
    val_p1 = accuracies['precision_at_1']
    print(f"Validation Precision@1: {val_p1:.4f}")

    # Early Stopping Check
    early_stopping(val_p1, model)
    if early_stopping.early_stop:
        print("TRAINING STOPPED! No improvement after 5 Epochs.")
        break

print("Training finished!")
print(f"Best Model saved under: {early_stopping.save_path}")