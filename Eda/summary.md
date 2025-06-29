This exploratory analysis was crucial for defining our project's direction.

1.  **For Task A (Gender Classification):**
    -   **Insight:** The dataset is balanced but visually challenging.
    -   **Action:** Proceed with a standard classification model but use **heavy data augmentation** to build robustness against blur, noise, and lighting issues.

2.  **For Task B (Face Verification):**
    -   **Insight:** The training and validation sets are **disjoint**. This is a verification task, not classification.
    -   **Action:** Build an embedding model using **metric learning (ArcFace Loss)**. Evaluate performance by calculating cosine similarity between image pairs and optimizing a decision threshold. This insight prevents us from wasting time on a flawed classification approach and sets us on the correct path.
