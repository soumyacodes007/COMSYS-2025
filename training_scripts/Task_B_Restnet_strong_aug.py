"""
task_B_Restnet_strong_aug.py

This script trains a Face Verification model (Task B) for the COMSYS Hackathon-5.
The key challenge is that training and validation identities are disjoint, making this
a metric learning problem, not a classification one.

This script trains a ResNet-34 backbone from scratch with a custom ArcFace head.
The goal is to learn a function that maps face images to a 512-dimensional
embedding space where images of the same identity are clustered together.

Aggressive data augmentations are used to ensure the model is robust to the
adverse conditions present in the dataset.

Example Usage:
    python src/train_verification.py \
        --data-path ./data/FACECOM_dataset/Comys_Hackathon5/Task_B \
        --save-path ./models/task_b_verification_model.pth \
        --epochs 40 \
        --batch-size 32 \
        --max-lr 1e-3 \
        --dropout 0.4
"""

#  Imports
import argparse
import os
import time
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Import our custom model definitions from src/models.py
from models import RegularizedArcFaceModel, AlbumentationsDataset

# loading data 
def create_dataloaders(data_dir, batch_size):
    """
    Creates a training dataloader for the metric learning task.
    Validation is handled separately by an evaluation script.

    Args:
        data_dir (str): Path to the root directory of the Task B dataset.
        batch_size (int): The number of samples per batch.

    Returns:
        tuple: A tuple containing the train DataLoader and the number of classes.
    """
    print("--- Preparing DataLoaders with Strong Augmentations ---")

    # Define strong augmentations to create a robust feature extractor
    train_transforms = A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.6),
        A.OneOf([
            A.MotionBlur(p=1.0), A.GaussianBlur(p=1.0), A.GaussNoise(p=1.0),
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0), A.ColorJitter(p=1.0),
        ], p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Create a base ImageFolder dataset to get class information
    base_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    
    # Wrap with our custom Albumentations dataset class
    train_dataset = AlbumentationsDataset(base_train_dataset, transform=train_transforms)
    
    # Create the DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    
    num_classes = len(base_train_dataset.classes)
    print(f"✅ DataLoaders ready. Training on {num_classes} identities.")
    return train_loader, num_classes

# train fn 
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device):
    """
    Handles the main training loop for the ArcFace model.

    Args:
        model: The PyTorch model to be trained.
        train_loader: The DataLoader for the training data.
        criterion: The loss function (CrossEntropyLoss for ArcFace output).
        optimizer: The optimization algorithm.
        scheduler: The learning rate scheduler.
        num_epochs (int): The total number of epochs to train for.
        device: The device to train on ('cuda' or 'cpu').
    """
    start_time = time.time()
    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Use tqdm for a nice progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # The ArcFace model requires labels during the forward pass for training
            outputs = model(inputs, labels)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            # OneCycleLR scheduler steps after each batch
            if scheduler:
                scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} complete. Train Loss: {epoch_loss:.4f}")

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

# main block 
def main(args):
    """
    Orchestrates the entire training process for the verification model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # Create dataloaders
    train_loader, num_classes = create_dataloaders(args.data_path, args.batch_size)

    # Initialize model
    print("\n--- Initializing ResNet34 + ArcFace model ---")
    model = RegularizedArcFaceModel(num_classes=num_classes, dropout_p=args.dropout)
    model.to(device)
    print("✅ Model initialized.")

    # Define loss and optimizer
    # The output of ArcFace is a logit, so CrossEntropyLoss is appropriate.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.max_lr)
    
    # OneCycleLR is a powerful scheduler that warms up, peaks, and cools down the
    # learning rate, often leading to faster and better convergence.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.max_lr, 
        epochs=args.epochs, 
        steps_per_epoch=len(train_loader)
    )

    # Start training
    print("\n--- Starting Model Training from Scratch ---")
    train_model(model, train_loader, criterion, optimizer, scheduler, args.epochs, device)
    
    # Save the final model
    print(f"\n--- Saving Final Model to {args.save_path} ---")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print("✅ Model saved successfully.")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train Face Verification Model (Task B) for COMSYS Hackathon-5")
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the root directory of the Task B dataset.')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Path to save the final trained model weights (.pth file).')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--max-lr', type=float, default=1e-3,
                        help='Maximum learning rate for the OneCycleLR scheduler.')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout probability for the embedding layer.')

    args = parser.parse_args()
    
    main(args)