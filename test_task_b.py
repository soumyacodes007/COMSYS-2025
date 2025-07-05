# test_task_b.py

import torch
import torch.nn.functional as F
import argparse
import random
import os
import sys
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

# Updated import to include all four metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import models from the models.py file in the same directory
from models import ResNetArcFaceModel, ConvNeXtArcFaceModel, AlbumentationsDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def evaluate_task_b(data_path, resnet_weights, convnext_weights, threshold, device):
    """
    Evaluates the face verification models (Task B) by ensembling their features.
    """
    print("\n" + "="*20 + " Evaluating Task B: Face Verification " + "="*20)

    # --- 1. Path and Argument Validation ---
    if not os.path.isdir(data_path):
        print(f"🚨 ERROR: Data path not found or is not a directory: '{data_path}'")
        sys.exit(1)
    if not os.path.isfile(resnet_weights):
        print(f"🚨 ERROR: ResNet model weights not found: '{resnet_weights}'")
        sys.exit(1)
    if not os.path.isfile(convnext_weights):
        print(f"🚨 ERROR: ConvNeXt model weights not found: '{convnext_weights}'")
        sys.exit(1)
        
    # --- 2. Model Loading ---
    # This is the number of identities the ArcFace models were trained on. It must match.
    NUM_TRAIN_IDENTITIES = 877 
    try:
        print("Loading ResNet-ArcFace model...")
        model_resnet = ResNetArcFaceModel(num_classes=NUM_TRAIN_IDENTITIES)
        model_resnet.load_state_dict(torch.load(resnet_weights, map_location=device))
        model_resnet.to(device).eval()

        print("Loading ConvNeXt-ArcFace model...")
        model_convnext = ConvNeXtArcFaceModel(num_classes=NUM_TRAIN_IDENTITIES)
        model_convnext.load_state_dict(torch.load(convnext_weights, map_location=device))
        model_convnext.to(device).eval()
        
        print("✅ Both Task B models loaded successfully.")
    except Exception as e:
        print(f"🚨 ERROR: Could not load one or more Task B models. Ensure weights match the architecture. Details: {e}")
        sys.exit(1)

    # --- 3. Data Loading and Embedding Generation ---
    transforms = A.Compose([
        A.Resize(224, 224), 
        A.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]), 
        ToTensorV2()
    ])
    try:
        dataset = AlbumentationsDataset(datasets.ImageFolder(data_path), transform=transforms)
        if len(dataset) == 0:
            print(f"🚨 ERROR: No images found in '{data_path}'. Please check the folder structure.")
            sys.exit(1)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        print(f"✅ Loaded {len(dataset)} images from '{data_path}' for embedding generation.")
    except Exception as e:
        print(f"🚨 ERROR: Could not create dataloader. Details: {e}")
        sys.exit(1)

    print("\n--- Generating Ensembled Embeddings from Test Data ---")
    embeddings_map = {}
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Generating Embeddings"):
            images = images.to(device)
            # Get features from both models
            features_resnet = model_resnet(images, label=None)
            features_convnext = model_convnext(images, label=None)
            
            # Ensemble by averaging normalized features, then re-normalize
            ensembled = F.normalize(F.normalize(features_resnet) + F.normalize(features_convnext))
            
            for j, label in enumerate(labels):
                identity_id = label.item()
                if identity_id not in embeddings_map:
                    embeddings_map[identity_id] = []
                embeddings_map[identity_id].append(ensembled[j])

    # --- 4. Pair Generation and Evaluation ---
    print("\n--- Creating Test Pairs and Evaluating with Fixed Threshold ---")
    positive_sim, negative_sim = [], []
    # Get all identities that have at least two images to form a positive pair
    valid_labels = [label for label, embs in embeddings_map.items() if len(embs) >= 2]
    
    if len(valid_labels) < 2:
        print("🚨 ERROR: Not enough identities with multiple images to create negative pairs. Need at least 2.")
        sys.exit(1)

    # Generate a fixed number of pairs for consistent evaluation
    for _ in range(4000):
        # 50% chance to generate a positive pair
        if random.random() > 0.5 and len(valid_labels) > 0:
            p_label = random.choice(valid_labels)
            emb1, emb2 = random.sample(embeddings_map[p_label], 2)
            positive_sim.append(F.cosine_similarity(emb1, emb2, dim=0).item())
        # Otherwise, generate a negative pair
        else:
            p1_label, p2_label = random.sample(valid_labels, 2)
            emb1 = random.choice(embeddings_map[p1_label])
            emb2 = random.choice(embeddings_map[p2_label])
            negative_sim.append(F.cosine_similarity(emb1, emb2, dim=0).item())

    # --- 5. Report Final Metrics (Updated to show all four) ---
    gt_labels = [1] * len(positive_sim) + [0] * len(negative_sim)
    pred_scores = positive_sim + negative_sim
    pred_labels = [1 if s > threshold else 0 for s in pred_scores]
    
    # Calculate all four metrics for the binary pair classification task
    final_accuracy = accuracy_score(gt_labels, pred_labels)
    final_precision = precision_score(gt_labels, pred_labels, zero_division=0)
    final_recall = recall_score(gt_labels, pred_labels, zero_division=0)
    final_f1 = f1_score(gt_labels, pred_labels, zero_division=0)

    print(f"\n--- Task B: Final Performance Metrics (Threshold = {threshold}) ---")
    print(f"Generated {len(positive_sim)} positive pairs and {len(negative_sim)} negative pairs.")
    print(f"Accuracy:          {final_accuracy:.4f}")
    print(f"Precision:         {final_precision:.4f}")
    print(f"Recall:            {final_recall:.4f}")
    print(f"F1-Score:          {final_f1:.4f}")
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standalone Evaluation Script for COMSYS-5 Task B (Face Verification)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the root folder of the Task B test dataset.')
    parser.add_argument('--resnet_weights', type=str, required=True, 
                        help='Path to the trained .pth file for the Task B ResNet-ArcFace model.')
    parser.add_argument('--convnext_weights', type=str, required=True, 
                        help='Path to the trained .pth file for the Task B ConvNeXt-ArcFace model.')
    parser.add_argument('--threshold', type=float, required=True, 
                        help='Optimal cosine similarity threshold determined from the validation set.')
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    evaluate_task_b(args.data_path, 
                    args.resnet_weights, 
                    args.convnext_weights, 
                    args.threshold, 
                    device)