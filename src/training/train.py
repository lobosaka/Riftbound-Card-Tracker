import os
import time
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
# Own Modules
from src.models.backbone import get_resnet50_model, get_normalization_params
from src.dataset.dataloader import RiftboundDataset, create_card_dict, augmentation

card_dict, _ = create_card_dict()

model = "ResNet50" # <- Current Model pick
num_epochs = 50 # <- Current amount of Epochs
# Get Parameters for Normalization (Augmentation Pipeline)
mean, std = get_normalization_params(model)
# Create Dataset and Dataloader
train_dataset = RiftboundDataset(image_dir='data/card_images_processed', card_dict=card_dict, transforms=augmentation(mean, std))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# Get Model and move to Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet50_model(num_classes=960)
model = model.to(device)
# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001) # pass only parameters that are not frozen (requires_grad=True)
# Switch to Training Mode
model.train()

# ---TRAINING LOOP---

print("Starte Training...")

for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0
    correct_predictions = 0
    total_predicitions = 0

    for images, labels in train_loader:
        # Move Data to Device
        images = images.to(device)
        labels = labels.to(device)
        # Reset Optimizer
        optimizer.zero_grad()
        # Forward Pass
        outputs = model(images)
        # Calculate Error
        loss = criterion(outputs, labels)
        # Backpropagation
        loss.backward()
        # Gradient Update
        optimizer.step()
        # Gather Loss
        running_loss += loss.item() * images.size(0)
        # Calculate Accuracy
        _, predicted = torch.max(outputs, 1)
        total_predicitions += labels.size(0)
        correct_predictions += (predicted==labels).sum().item()

    # Epoch-Statistic
    epoch_loss = running_loss/len(train_dataset)
    epoch_acc = (correct_predictions/total_predicitions) * 100
    elapsed_time = time.time() - start_time

    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch: {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f} - Time: {elapsed_time:.1f}")

print("Training erfolgreich beendet!")

# Save Model
save_dir = "src/models/checkpoints"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, 'riftbound_resnet50_weights.pth'))
print("Modellgewichte erfolgreich gespeichert!")