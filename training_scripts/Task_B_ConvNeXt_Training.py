"""
task_b_convnext_training.py

This script trains a Face Verification model (Task B) for the COMSYS Hackathon-5,
specifically using a ConvNeXt-Tiny backbone. This serves as an alternative and
potentially more powerful feature extractor to the ResNet-based model.

The training methodology remains the same: metric learning using ArcFace on disjoint
identity sets. This script also includes a validation loop to monitor the validation
loss on the unseen identities, saving the model checkpoint with the lowest validation
loss. This serves as a proxy for the quality of the learned embedding space.

Example Usage:
    python src/train_verification_convnext.py \
        --data-path ./data/FACECOM_dataset/Comys_Hackathon5/Task_B \
        --save-path ./models/task_b_verification_convnext.pth \
        --epochs 40 \
        --batch-size 32 \
        --max-lr 1e-3 \
        --dropout 0.4
"""

# imports 
import argparse
import os
import time
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# whil fix the train py after fixing this .
from models import ConvNeXtArcFaceModel, AlbumentationsDataset

# -
def create_dataloaders(data_dir, batch_size):
    """
    Creates training and validation dataloaders for the metric learning task.
    Simple augmentations are used for this experiment to test the raw power of
    the ConvNeXt architecture.

    Args:
        data_dir (str): Path to the root directory of the Task B dataset.
        batch_size (int): The number of samples per batch.

    Returns:
        tuple: A tuple containing the dataloaders dictionary and number of classes.
    """
    print("--- Preparing DataLoaders with Simple Augmentations ---")

    # Simple augmentations for training
    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Validation transforms (no augmentation besides resizing and normalizing)
    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create base ImageFolder datasets
    base_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    base_val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'))

    # Wrap with our custom Albumentations dataset class
    train_dataset = AlbumentationsDataset(base_train_dataset, transform=train_transforms)
    val_dataset = AlbumentationsDataset(base_val_dataset, transform=val_transforms)
    
    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    }
    
    num_classes = len(base_train_dataset.classes)
    print(f" DataLoaders ready. Training on {num_classes} identities.")
    return dataloaders, num_classes

# train loop 
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, save_path):
    """
    Handles the main training and validation loop, including checkpointing.

    Args:
        model: The PyTorch model to be trained.
        dataloaders (dict): A dictionary containing 'train' and 'val' DataLoaders.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        scheduler: The learning rate scheduler.
        num_epochs (int): The total number of epochs to train for.
        device: The device to train on ('cuda' or 'cpu').
        save_path (str): The path to save the best model checkpoint.
    """
    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'\n--- Epoch {epoch+1}/{num_epochs} ---')

        
        model.train()
        running_train_loss = 0.0
        pbar_train = tqdm(dataloaders['train'], desc="Training")
        for inputs, labels in pbar_train:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            running_train_loss += loss.item() * inputs.size(0)
            pbar_train.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        epoch_train_loss = running_train_loss / len(dataloaders['train'].dataset)
        print(f"Train Loss: {epoch_train_loss:.4f}")

        # phase - validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(dataloaders['val'], desc="Validating")
            for inputs, labels in pbar_val:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, labels)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = running_val_loss / len(dataloaders['val'].dataset)
        print(f"Validation Loss: {epoch_val_loss:.4f}")

        # checkpointing 
        if epoch_val_loss < best_val_loss:
            print(f" New best model found! Val Loss improved from {best_val_loss:.4f} to {epoch_val_loss:.4f}.")
            print(f"Saving checkpoint to {save_path}...")
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(" Checkpoint saved.")

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🏆 Best Validation Loss: {best_val_loss:.4f}')

# exicution block
def main(args):
    """
    Orchestrates the entire training process for the ConvNeXt verification model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    dataloaders, num_classes = create_dataloaders(args.data_path, args.batch_size)

    print("\n--- Initializing ConvNeXt-Tiny + ArcFace model ---")
    model = ConvNeXtArcFaceModel(num_classes=num_classes, model_name='convnext_tiny', dropout_p=args.dropout)
    model.to(device)
    print(" Model initialized.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.max_lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(dataloaders['train'])
    )

    print("\n--- Starting Model Training ---")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    train_model(model, dataloaders, criterion, optimizer, scheduler, args.epochs, device, args.save_path)
    print(f"\n Training complete. The best model is saved at '{args.save_path}'.")

if __name__ == '__main__':
    # --- 5. Argument Parser ---
    parser = argparse.ArgumentParser(description="Train ConvNeXt Face Verification Model for COMSYS Hackathon-5")
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the root directory of the Task B dataset.')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Path to save the final best model checkpoint (.pth file).')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training and validation.')
    parser.add_argument('--max-lr', type=float, default=1e-3,
                        help='Maximum learning rate for the OneCycleLR scheduler.')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout probability for the embedding layer.')
    
    args = parser.parse_args()
    
    main(args)