# Robust Face Intelligence Under Adverse Conditions
### COMSYS Hackathon-5, 2025 Submission

TEAM : GEN SONIC 




## 🛠️ Steps to Run the Solution

To ensure a smooth and reproducible evaluation, please follow these steps precisely. All commands should be executed from the root `COMSYS-2025/` directory.

---

### 🔧 A. Environment Setup

#### 1. Clone Repository

```bash
git clone https://github.com/soumyacodes007/COMSYS-2025
cd COMSYS-2025
```

#### 2. Create and Activate a Virtual Environment (Recommended)

This prevents conflicts with system-wide packages.

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate
```

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

Install all required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---
## 🔽 Model Download & Setup

To run the evaluation scripts, you need to download the pretrained model weights and place them correctly. Follow the steps below:

---

### ✅ Steps:


## ⬇️ Download Model Weights (Mandatory Step)

Our model weights are hosted on **Google Drive** to keep the repository lightweight. You can download them automatically using our script (**recommended**) or manually.

---

### 🔧 Run the Download Script

From the root of the project, run the following command:

```bash
python download_models.py
```

---

### 📤 Expected Output

The script will:

- Check if the `models/` directory exists (create it if missing)
- Download any missing `.pth` model files from Google Drive
- Confirm setup status

Sample output:

```bash
✅ Model setup complete. All required models are in place.
```

> ⏳ **Note:** Download time may vary depending on your internet speed.

---

💡 *If you face issues with the script, you can also follow the manual download method [described above](#-model-download--setup).*




MANUAL SET UP 

1. **Create the `models` directory**  
   In the root of the project (`COMSYS/`), create a new folder named:

   ```bash
   mkdir models
   ```

2. **Download each model file**  
   Click the links below to download the required `.pth` files:

| Model for...                          | File Name                    | Download Link                                                                                   |
|--------------------------------------|------------------------------|--------------------------------------------------------------------------------------------------|
| Task A (Gender Classification)       | `task_A_BEST_model.pth`      | [Download](https://drive.google.com/file/d/1DEHfGpFBVTe1-IAt7gBOh8yuU6uXSBW9/view?usp=sharing)   |
| Task B (ConvNeXt Component)          | `task_B_convnext_best.pth`   | [Download](https://drive.google.com/file/d/1oI_dU9bmqLbtIG2uzrEm5Hy3wjNoYaP_/view?usp=sharing)  |
| Task B (ResNet Component)            | `task_B_model_strong_aug.pth`| ✅ *This file is already present in the GitHub repo under `models/` folder.*                     |

---

3. **Place the Files**  
   After downloading, move all three `.pth` files into the `models/` directory you created:

   ```bash
   mv *.pth models/
   ```

---

> ⚠️ **Note:** If you are using Google Colab or a cloud environment, make sure to upload these files to the appropriate location or mount your Drive before accessing them.








### 🚀 B. Running Evaluation Scripts

After setting up the environment, use the following commands to evaluate each task:

---

#### ✅ To Evaluate Task A (Gender Classification)

```bash
python test_task_a.py --data_path "/path/to/your/TaskA_data" --model_weights_path "./models/task_A_BEST_model.pth"
```

> **Note**: Replace `/path/to/your/TaskA_data` with the absolute or relative path to the Task A test dataset folder.

---

#### ✅ To Evaluate Task B (Face Verification)

```bash
python test_task_b.py --data_path "/path/to/your/TaskB_data" --resnet_weights "./models/task_B_model_strong_aug.pth" --convnext_weights "./models/task_B_convnext_best.pth" --threshold 0.31
```

> **Note**: Replace `/path/to/your/TaskB_data` with the path to the Task B test dataset folder.  
> The threshold is pre-set to the optimal value **(0.31)** found during validation.















## 📊 3. Training and Validation Results

The following tables show the performance of our final models on the training and validation sets.

---

### **Task A: Gender Classification**

| Phase      | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| Training   | 0.9985   | 0.9979    | 0.9979 | 0.9979   |
| Validation | 0.9479   | 0.9177    | 0.9095 | 0.9135   |

---


## Task B: Face Verification

**Metrics based on 4000 generated pairs with an optimal threshold of 0.31**

| Phase       | Accuracy | Precision | Recall  | F1-Score |
|-------------|----------|-----------|---------|----------|
| Training    | 0.9912   | ~0.990    | ~0.990  | 0.9902   |
| Validation  | 0.9580   | 0.9814    | 0.9327  | 0.9564   |




## 🧠 Model Architecture Description

---

### 🔹 Task A: Gender Classification Model

- **Architecture**: ConvNeXt-Small  
- **Description**:  
  We leverage the **ConvNeXt-Small** model, a modern and highly effective convolutional neural network architecture, implemented via the `timm` library.  
  The model takes a **224×224** image as input and passes it through deep feature extraction layers. The final stage is a **linear classifier** that outputs logits for two classes (**male/female**).  

- **Evaluation Technique**:  
  Our evaluation script `test_task_a.py` uses **Test-Time Augmentation (TTA)** — predictions for an image and its **horizontal flip** are averaged to improve robustness.

---

### 🔹 Task B: Face Verification Model

- **Architecture**: Ensemble of **ResNet34-ArcFace** and **ConvNeXt-Tiny-ArcFace**  
- **Description**:  
  Our solution for face verification is a powerful **ensemble** of two distinct models to enhance feature representation and robustness.

  - **ResNet34-ArcFace**:  
    Uses a **ResNet34** backbone to extract **512-dimensional** feature embeddings.  
    Trained using the **ArcFace loss function**, which enforces angular margin between identities for better class separation and compactness.

  - **ConvNeXt-Tiny-ArcFace**:  
    A **ConvNeXt-Tiny** model trained using the same **ArcFace** metric-learning approach.  
    Adds architectural diversity by capturing features the ResNet might overlook.

- **Ensemble Strategy**:
  - During inference, both models generate **512-d embeddings** for a given image.
  - These embeddings are **L2-normalized**, **added together**, and the result is normalized again.
  - The final ensembled embedding is used for **cosine similarity** comparison between image pairs.
  - Verification is done by comparing the cosine similarity score against a **fixed threshold of 0.31**.



8. Training Methodology
Framework: PyTorch
Optimizer: AdamW
Learning Rate Scheduler: Cosine Annealing with Warmup.
Loss Functions:
Task A: Cross-Entropy Loss.
Task B: Additive Angular Margin Loss (ArcFace).
Data Augmentation: We used the albumentations library for strong augmentations during training, including random flips, rotations, color jitter, and coarse dropout, to ensure the models generalize well.



### 2. Prepare the Data

This repository does not include the dataset. Please download and unzip the dataset so that your folder structure looks like this:

```
COMSYS/
├── Comys_Hackathon5/
│   ├── Comys_Hackathon5/   <-- Note the nested folder
│   │   ├── Task_A/
│   │   │   └── val/
│   │   └── Task_B/
│   │       └── val/
├── models/
├── test.py
└── ... (other files)
```






## 🛡️ Robustness & Fair Evaluation Card

We are committed to a scientifically rigorous and fair evaluation. Our methodology was designed to produce a score that reflects true real-world performance.

| Principle | Our Action | Why It Matters |
| :--- | :--- | :--- |
| **No Data Leakage** | The decision threshold for Task B (`0.31`) was determined **exclusively on the validation set**, *before* the final test script was run. | This prevents "peeking" at the test set. Our score is an honest measure of generalization on unseen data, not an overfitted result. |
| **Real-World Scenario** | Our evaluation tests the model on thousands of random **positive (match)** and **negative (impostor)** pairs. | This is the true test of a verification system: it must not only identify correct matches but also correctly reject incorrect ones. |
| **Training from Scratch**| Our feature extractors for Task B were trained from scratch (`pretrained=False`) using ArcFace loss. | We identified that standard ImageNet pre-training constitutes "negative transfer" for this specific metric learning task. Training from scratch allowed our models to learn a more specialized and powerful embedding space. |
| **Reproducibility** | Our entire testing pipeline is encapsulated in a single script (`test.py`) that can be run with one command. | Ensures that the judges can easily and reliably reproduce our reported results. |

---

## 🚀 What We Did (Our Journey)

1.  **Initial Analysis & The Great Filter:** We began by treating Task B as a multi-class classification problem. This failed completely, leading to a crucial data-centric investigation. **We discovered that the training and validation identities were 100% disjoint.** This insight was the cornerstone of our entire project.

2.  **Strategic Pivot to Metric Learning:** Realizing the true nature of the task, we abandoned classification and adopted a state-of-the-art metric learning approach. We chose **ArcFace loss** to train our models to produce highly discriminative embeddings.

3.  **Systematic Experimentation:**
    *   **Task A:** We fine-tuned a powerful `ConvNeXt-small` model with aggressive data augmentation to achieve high accuracy on the gender classification task.
    *   **Task B:** We trained two separate backbones from scratch—a `ResNet-34` with strong augmentation and a `ConvNeXt-Tiny`—to serve as our feature extractors.

4.  **The Winning Ensemble:** Our best performance was achieved by ensembling the embeddings from our ResNet and ConvNeXt models, creating a single, robust feature vector that leverages the strengths of both architectures.

5.  **Final Submission:** We packaged our entire solution into a clean, reproducible repository with a single evaluation script, ensuring our work is transparent, professional, and easy to verify.

## 🏆 Project Overview & Key Achievements

This project presents our state-of-the-art solution for the "Robust Face Recognition and Gender Classification" challenge. Our approach was driven by a rigorous, data-centric methodology that led to a crucial insight: **Task B was not a classification problem, but a sophisticated face verification task with disjoint identities.**

By correctly framing the problem and employing advanced metric learning techniques, we built a powerful ensemble model that achieves exceptional performance on a fair and robust evaluation protocol.

### Key Achievements:
*   **Correctly Identified the True Nature of Task B:** Uncovered the disjoint identity sets between training and validation, pivoting our strategy from classification to verification.
*   **Developed a State-of-the-Art Ensemble Model:** Combined a ResNet-34 and a ConvNeXt-Tiny, trained from scratch with ArcFace loss, to create a highly discriminative feature extractor.
*   **Implemented a Scientifically Valid Evaluation:** Used a fixed decision threshold determined on the validation set to prevent data leakage and ensure our results reflect true generalization performance.

---


## 📁 Codebase Structure

Our project is organized into a clean and logical directory structure to separate concerns like data, model definitions, final model weights, and evaluation scripts. This modular approach ensures clarity and makes the project easy to navigate and evaluate.

---

### 🌳 Visual File Tree

```plaintext
COMSYS/
│
├── Comys_Hackathon5/
│   └── Comys_Hackathon5/
│       ├── Task_A/                  # Root data folder for Task A
│       │   ├── train/              # Training images for Task A
│       │   └── val/                # Validation images for Task A
│       ├── Task_B/                  # Root data folder for Task B
│       │   ├── train/              # Training images for Task B
│       │   └── val/                # Validation images for Task B
│       └── Eda/                    # (Reference) Exploratory Data Analysis notebooks
│
├── models/
│   ├── task_A_BEST_model.pth       # Final weights for Task A model
│   ├── task_B_convnext_best.pth    # Final weights for Task B ConvNeXt model
│   └── task_B_model_strong_aug.pth # Final weights for Task B ResNet model
│
├── training_scripts/               # (Reference) Scripts used for model training
│
├── __pycache__/                    # Auto-generated Python cache directory
│
├── .gitignore                      # Specifies files for Git to ignore
├── models.py                       # Core module with all model class definitions
├── README.md                       # This documentation file
├── requirements.txt                # All Python dependencies
├── test_task_a.py                  # << EXECUTABLE SCRIPT FOR TASK A >>
└── test_task_b.py                  # << EXECUTABLE SCRIPT FOR TASK B >>
```

---

### 🔍 Component Breakdown

#### 📂 COMSYS/ (Root Directory)
The main project folder that contains all other files and directories.  
➡️ **All commands should be run from here.**

---

#### ⚙️ Runnable Scripts

- `test_task_a.py`  
  👉 Official evaluation script for Task A. Loads data/model, performs evaluation, and prints metrics.

- `test_task_b.py`  
  👉 Official evaluation script for Task B. Loads ensemble models, performs face verification using cosine similarity, and reports accuracy and F1-score.

---

#### 🧠 Core Model Logic

- `models.py`  
  📌 Central file containing all model definitions:
  - `GenderClassifier`
  - `ResNetArcFaceModel`
  - `ConvNeXtArcFaceModel`
  - `AlbumentationsDataset`

  The evaluation scripts import these classes directly from here.

---

#### 💾 Model Weights

- `models/`  
  📁 Stores all final trained `.pth` model weights:
  - `task_A_BEST_model.pth`
  - `task_B_convnext_best.pth`
  - `task_B_model_strong_aug.pth`

---


```mermaid
graph TD
    subgraph "Input Processing"
        A[Input Image]
    end

    subgraph "Parallel Feature Extraction"
        B(ResNet-34 Backbone)
        C(ConvNeXt-Tiny Backbone)
    end

    subgraph "Embedding Generation"
        D[512-d Embedding A]
        E[512-d Embedding B]
    end

    subgraph "Ensemble Fusion"
        F{Normalize & Add}
    end

    subgraph "Final Output"
        G([Final Ensembled Embedding])
    end

    A --> B
    A --> C
    B --> D
    C --> E
    D --> F
    E --> F
    F --> G

