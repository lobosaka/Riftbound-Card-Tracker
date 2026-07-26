import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def get_normalization_params(model='ResNet50'):
    if model == "ResNet50":
        model = models.ResNet50_Weights.DEFAULT
        return model.transforms().mean, model.transforms().std
    
class ResNet(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()
        # Load ResNet50 with standard weights
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)
        # Freeze all Layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Get dimension of hidden layer before head
        in_features = self.backbone.fc.in_features
        # Delete Head by applying Identity
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.ReLU(),
                nn.Linear(in_features, projection_dim)
            )

    def forward(self, x):
        # Send pictures through backbone until last hidden layer
        features = self.backbone(x)
        # Pass features of last hidden layer of backbone through projection head
        out = self.head(features)
        # L2-Normalzation of the Output-Embeddings aus dem Projection Head
        out = F.normalize(out, p=2, dim=1)
        return out


        