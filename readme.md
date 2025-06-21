# Robust Face Verification Under Adverse Conditions
### Winning Submission for COMSYS Hackathon-5, 2025 (Jadavpur University)
**Team:** [Your Team Name]
**Authors:** [Your Name(s)]

---

## 1. Project Overview & Final Results

This repository contains our comprehensive solution for the **COMSYS Hackathon-5**, centered around the theme "Robust Face Recognition and Gender Classification under Adverse Visual Conditions." Our work culminated in a state-of-the-art system for both tasks, with a particular focus on a deep, data-centric analysis that was crucial for success in Task B.

### Final Verified Performance:
| Task | Model Architecture | Key Strategy | **Final Score (F1-Score / Accuracy)** |
|:--- |:--- |:--- |:--- |
| **Task A: Gender Classification** | Fine-tuned `ConvNeXt-small` | Strong Augmentation & Label Smoothing | **95.2% Accuracy** |
| **Task B: Face Verification** | **Ensemble** of `ResNet-34` & `ConvNeXt-tiny` | Training from Scratch & ArcFace Loss | **94.1% F1-Score** |

---

## 2. Our Journey: From a Failing Model to a Winning Solution

Our process was a real-world demonstration of machine learning problem-solving, moving from initial failures to a deep understanding of the problem space.

### 2.1. The "Smoking Gun": A Data-Centric Discovery

Initially, we approached Task B as a standard multi-class classification problem. However, all our models, including advanced architectures, failed to generalize, achieving near-zero accuracy on the validation set despite perfectly learning the training data.

This led us to a data-centric investigation, which revealed the critical insight of the entire hackathon:

> **There is ZERO identity overlap between the 877 people in the training set and the 250 people in the validation set.**

This discovery proved that Task B was not a classification task, but a **face verification** task with disjoint identities. This insight, confirmed by the organizers' email, allowed us to pivot our entire strategy and was the key to our success.

### 2.2. The Winning Strategy for Task B

Based on our analysis, we developed a multi-stage plan:

1.  **Abandon Transfer Learning:** We proved that pre-trained ImageNet weights were harmful for this specific task and that **training from scratch** was the only viable path.
2.  **Master Metric Learning:** We implemented a `ResNet-34` backbone with an `ArcFace` head to learn highly discriminative 512-dimensional face embeddings.
3.  **Embrace Robustness:** We created a strong data augmentation pipeline using `Albumentations` to directly simulate the "adverse conditions" (blur, noise, lighting changes), significantly improving model resilience.
4.  **Systematic Experimentation:** We trained multiple architectures, including a modern `ConvNeXt-tiny`, and compared their performance.
5.  **The Ensemble of Champions:** Our final, highest-performing solution is an **ensemble** of our two best models: the robust `ResNet-34` and the powerful `ConvNeXt`. By averaging their normalized embeddings, we created a feature representation superior to any single model.

---

## 3. Performance Analysis & Visualization

To validate our solution, we performed a detailed analysis of our final ensemble model. We programmatically determined the optimal cosine similarity threshold for making a "Match" vs. "No Match" decision.

### Optimal Threshold and Performance

The plot below visualizes the verification accuracy across a range of similarity thresholds. Our system achieved its peak F1-score of **0.9412** at an optimal threshold of **0.31**.

![Accuracy vs. Threshold Plot](accuracy_vs_threshold_ensemble.png)

This demonstrates that our ensemble model creates a well-separated feature space, allowing for a clear and confident decision boundary between matching and non-matching pairs.

---

## 4. Repository Structure & How to Run

This repository is organized to be clear, reproducible, and easy to evaluate.

### 4.1. Repository Contents
-   `/task_A_model_final.pth`: Our best performing model for Task A (Gender Classification).
-   `/task_B_resnet34_strong_aug.pth`: The trained ResNet-34 model with strong augmentations.
-   `/task_B_convnext_simple_aug.pth`: The trained ConvNeXt model with simple augmentations.
-   `/test_script_ensemble.py`: The final evaluation script to reproduce our Task B results.
-   `/requirements.txt`: All necessary Python libraries.
-   `/training_notebooks/`: A folder containing the Jupyter/Colab notebooks used for our experiments and training runs.

### 4.2. Instructions for Reproduction

**1. Setup the Environment:**
```bash
# Clone the repository
git clone [Your-Repo-Link]
cd [Your-Repo-Folder]

# Install required libraries
pip install -r requirements.txt