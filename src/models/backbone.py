import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet50_model(num_classes=960):
    # Standardgewichte des ResNet50 Modells laden
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    # Alle Layer des Netzwerks einfrieren
    for param in model.parameters():
        param.requires_grad = False
    # Klassifikationskopf austauschen
    # Input Features aus dem letzen Hidden Layer extrahieren
    in_features = model.fc.in_features
    # Klassifikationskopf mit neuen Anzahl Klassen überschreiben
    model.fc = nn.Linear(in_features, num_classes)
    # Model zurückgeben. fc-Schicht hat automatisch requires_grad=True
    return model   

def get_normalization_params(model):
    if model == "ResNet50":
        model = models.ResNet50_Weights.DEFAULT
        return model.transforms().mean, model.transforms().std
        