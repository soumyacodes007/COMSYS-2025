"""
task_A_Training.py

This script trains a Gender Classification model (Task A) for the COMSYS Hackathon-5.
It uses a pre-trained ConvNeXt-Small model and fine-tunes it on the FACECOM dataset
with aggressive data augmentations to handle adverse visual conditions.

The script is designed to be run from the command line with configurable arguments.

Example Usage:
    python src/task_A_Training.py \
        --data-path ./data/FACECOM_dataset/Comys_Hackathon5/Task_A \
        --save-path ./models/task_a_gender_classifier.pth \
        --epochs 20 \
        --batch-size 16 \
        --lr 1e-4
"""

#  all imports are necessary for the script to function correctly
import argparse
import os
import copy
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

# using albumentations for data augmentation
class AlbumentationsDataset(Dataset):
    """
    A custom PyTorch Dataset that wraps a standard ImageFolder dataset
    to apply transformations from the Albumentations library.
    """
    def __init__(self, image_folder_dataset, transform=None):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform

    def __len__(self):
        return len(self.image_folder_dataset)

    def __getitem__(self, idx):
        # Get the path and label from the underlying ImageFolder dataset
        path, label = self.image_folder_dataset.samples[idx]
        
        # Read image with OpenCV and convert from BGR to RGB
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        # Apply Albumentations transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        return image, label

# loading the data 
def create_dataloaders(data_dir, batch_size):
    """
    Creates training and validation dataloaders with specified data augmentations.
    
    Args:
        data_dir (str): Path to the root directory of the Task A dataset.
        batch_size (int): The number of samples per batch.

    Returns:
        tuple: A tuple containing the dataloaders dictionary and class names list.
    """
    print("--- Preparing DataLoaders ---")
    
    # Define aggressive augmentations for the training set to build robustness
    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.7, shift_limit=0.1, scale_limit=0.15, rotate_limit=20),
        A.OneOf([A.MotionBlur(p=1.), A.GaussianBlur(p=1.), A.GaussNoise(p=1.)], p=0.6),
        A.OneOf([A.RandomBrightnessContrast(p=1.), A.ColorJitter(p=1.), A.CLAHE(p=1.)], p=0.6),
        A.CoarseDropout(max_holes=8, max_height=25, max_width=25, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Define simple augmentations for the validation set (only resize and normalize)
    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create base ImageFolder datasets
    base_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    base_val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'))

    # Wrap them with our custom Albumentations dataset class
    train_dataset = AlbumentationsDataset(base_train_dataset, transform=train_transforms)
    val_dataset = AlbumentationsDataset(base_val_dataset, transform=val_transforms)

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    }
    
    class_names = base_train_dataset.classes
    print(f"✅ DataLoaders ready. Classes: {class_names}")
    return dataloaders, class_names

# training the model
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device):
    """
    The main function to handle the training and validation loop.

    Args:
        model: The PyTorch model to be trained.
        dataloaders (dict): A dictionary containing 'train' and 'val' DataLoaders.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        scheduler: The learning rate scheduler.
        num_epochs (int): The total number of epochs to train for.
        device: The device to train on ('cuda' or 'cpu').

    Returns:
        The model with the best validation accuracy.
    """
    start_time = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if we have a new best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                print(f"    ✨ New best validation accuracy: {epoch_acc:.4f}")
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # Step the scheduler after each epoch
        if scheduler:
            scheduler.step()

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🏆 Best Validation Accuracy: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# executing the main function
def main(args):
    """
    Orchestrates the entire training process.
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # Create dataloaders
    dataloaders, class_names = create_dataloaders(args.data_path, args.batch_size)
    num_classes = len(class_names)

    # Initialize model
    print("\n--- Initializing ConvNeXt-Small model ---")
    model = timm.create_model('convnext_small', pretrained=True, num_classes=num_classes)
    model.to(device)
    print("✅ Model initialized.")

    # Define loss, optimizer, and scheduler
    # Label smoothing helps prevent overconfidence and improves generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # AdamW is a robust optimizer, often better than standard Adam for vision tasks
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    
    # Cosine Annealing scheduler adjusts the learning rate in a cosine curve,
    # often leading to better convergence.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Start training
    print("\n--- Starting Model Training ---")
    best_model = train_model(model, dataloaders, criterion, optimizer, scheduler, args.epochs, device)
    
    # Save the best model
    print(f"\n--- Saving Best Model to {args.save_path} ---")
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(best_model.state_dict(), args.save_path)
    print(f"✅ Model saved successfully.")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train Gender Classification Model for COMSYS Hackathon-5")
    
    parser.add_argument('--data-path', type=str, required=True, 
                        help='Path to the root directory of the Task A dataset.')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Path to save the final best model weights (.pth file).')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=16, 
                        help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Initial learning rate.')

    args = parser.parse_args()
    
    main(args)