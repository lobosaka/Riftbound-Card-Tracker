import os
import time
import json
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_metric_learning import losses
# Own Modules
from src.representation_learning.models.backbone import ResNet, get_normalization_params
from src.representation_learning.dataset.dataloader import RiftboundDataset, create_card_dict, augmentation, ContrastiveTransforms

create_card_dict()

json_path = "data/card_to_label.json"
with open(json_path, 'r', encoding='utf-8') as f:
    card_dict = json.load(f)

model = "ResNet50" # <- Current Model pick
num_epochs = 200 # <- Current Number of Epochs pick
# Get Parameters for Normalization (Augmentation Pipeline)
mean, std = get_normalization_params(model)
# Create Dataset and Dataloader
train_dataset = RiftboundDataset(image_dir='data/card_images', card_dict=card_dict, transforms=ContrastiveTransforms(augmentation(mean=mean, std=std)))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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

# Define Loss, Optimizer and Scheduler
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4) # pass only parameters that are not frozen (requires_grad=True)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=200, eta_min=1e-7)
criterion = losses.NTXentLoss(temperature=0.1) 

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


print("Training erfolgreich beendet!")

# Save Model
save_dir = "src/representation_learning/models/checkpoints"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, 'riftbound_resnet50_weights_RL.pth'))
print("Modellgewichte erfolgreich gespeichert!")