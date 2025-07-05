# COMSYS/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset
import math
import timm
import cv2

class AlbumentationsDataset(Dataset):
    """
    A wrapper for torchvision.datasets.ImageFolder to apply albumentations transforms.
    """
    def __init__(self, image_folder_dataset, transform=None):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
    def __len__(self):
        return len(self.image_folder_dataset)
    def __getitem__(self, idx):
        path, label = self.image_folder_dataset.samples[idx]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

def GenderClassifier(num_classes=2, pretrained=False):
    """
    Task A Model: A ConvNeXt Small model for gender classification.
    """
    model = timm.create_model('convnext_small', pretrained=pretrained, num_classes=num_classes)
    return model

class ArcFace(nn.Module):
    """
    Implementation of the ArcFace loss function for face recognition.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features, self.out_features, self.s, self.m = in_features, out_features, s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.eps, self.cos_m, self.sin_m = 1e-7, math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi-m), math.sin(math.pi-m)*m
    def forward(self, x, l):
        c = F.linear(F.normalize(x), F.normalize(self.weight))
        s = torch.sqrt(1.-c.pow(2)+self.eps)
        p = c*self.cos_m-s*self.sin_m
        p = torch.where(c > self.th, p, c-self.mm)
        o = torch.zeros(c.size(), device=x.device)
        o.scatter_(1, l.view(-1, 1).long(), 1)
        out = (o * p) + ((1.-o)*c)
        out *= self.s
        return out

class ResNetArcFaceModel(nn.Module):
    """
    Task B Model 1: ResNet34 backbone with an ArcFace head.
    """
    def __init__(self, num_classes, dropout_p=0.4):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        e = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(e, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_p)
        )
        self.head = ArcFace(512, num_classes)
    def forward(self, x, label=None):
        f = self.backbone(x)
        return self.head(f, label) if label is not None else f

class ConvNeXtArcFaceModel(nn.Module):
    """
    Task B Model 2: ConvNeXt backbone with an ArcFace head.
    """
    def __init__(self, num_classes, model_name='convnext_tiny', dropout_p=0.4):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        e = self.backbone.num_features
        self.embedding_layer = nn.Sequential(
            nn.Linear(e, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_p) # Corrected from 'd' to 'dropout_p'
        )
        self.head = ArcFace(512, num_classes)
    def forward(self, x, label=None):
        f = self.backbone(x)
        e = self.embedding_layer(f)
        return self.head(e, label) if label is not None else e