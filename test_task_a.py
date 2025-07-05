# COMSYS/test_task_a.py

import torch
import argparse
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import models from the models.py file in the same directory
from models import GenderClassifier, AlbumentationsDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def evaluate_task_a(data_path, model_weights_path, device):
    """
    Evaluates the gender classification model (Task A).
    """
    print("\n" + "="*20 + " Evaluating Task A: Gender Classification " + "="*20)
    
    # --- Model Loading ---
    try:
        model = GenderClassifier(num_classes=2)
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.to(device).eval()
        print(f"✅ Task A model loaded successfully from {model_weights_path}")
    except Exception as e:
        print(f"🚨 ERROR: Could not load Task A model. {e}")
        return

    # --- Data Loading ---
    transforms = A.Compose([
        A.Resize(224, 224), 
        A.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]), 
        ToTensorV2()
    ])
    try:
        full_data_path = os.path.abspath(data_path)
        if not os.path.isdir(full_data_path):
             print(f"🚨 ERROR: The data path does not exist or is not a directory: {full_data_path}")
             return
        dataset = AlbumentationsDataset(datasets.ImageFolder(full_data_path), transform=transforms)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        print(f"✅ Loaded {len(dataset)} images from {full_data_path}")
    except Exception as e:
        print(f"🚨 ERROR: Could not create Task A dataloader from path: {full_data_path}. {e}")
        return

    # --- Evaluation Loop ---
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Task A Evaluation"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Print Results ---
    print("\n--- Task A: Gender Classification Results ---")
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds, zero_division=0, average='macro'):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, zero_division=0, average='macro'):.4f}")
    print(f"F1-Score:  {f1_score(all_labels, all_preds, zero_division=0, average='macro'):.4f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation Script for COMSYS-5 Task A", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to the root test data folder for Task A.')
    parser.add_argument('--model_weights_path', type=str, required=True, help='Path to the .pth file for the Task A model.')
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    evaluate_task_a(args.data_path, args.model_weights_path, device)