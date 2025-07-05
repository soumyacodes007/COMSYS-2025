# this will download models from google drive


import gdown
import os

# A list of dictionaries, where each dictionary contains info for one model
MODELS_TO_DOWNLOAD = [
    {
        "file_id": "1DEHfGpFBVTe1-IAt7gBOh8yuU6uXSBW9",
        "output_name": "task_A_BEST_model.pth",
        "description": "Task A Best Model"
    },
    {
        "file_id": "1oI_dU9bmqLbtIG2uzrEm5Hy3wjNoYaP_",
        "output_name": "task_B_convnext_best.pth",
        "description": "Task B ConvNext Best Model"
    }
]

# The target directory based on your project structure
# os.path.join makes it work on any OS (Windows, Mac, Linux)
TARGET_DIRECTORY = os.path.join("models")


def download_all_models():
    """
    Checks for and downloads all required models from Google Drive.
    """
    # First, create the target directory if it doesn't exist
    os.makedirs(TARGET_DIRECTORY, exist_ok=True)
    print(f"Ensured model directory exists: {TARGET_DIRECTORY}")

    # Loop through each model in our list
    for model_info in MODELS_TO_DOWNLOAD:
        file_id = model_info["file_id"]
        output_name = model_info["output_name"]
        description = model_info["description"]
        
        # Create the full path for the output file
        output_path = os.path.join(TARGET_DIRECTORY, output_name)
        
        print("-" * 50)
        print(f"Processing: {description}")

        # Check if the file already exists to avoid re-downloading
        if os.path.exists(output_path):
            print(f"-> Model '{output_name}' already exists. Skipping download.")
            continue
        
        # If it doesn't exist, download it
        print(f"-> Model not found. Downloading '{output_name}'...")
        download_url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            gdown.download(download_url, output_path, quiet=False)
            print(f"-> Successfully downloaded to {output_path}")
        except Exception as e:
            print(f"!! FAILED to download {output_name}. Error: {e}")
            print("!! Please check the file ID and your internet connection.")

if __name__ == "__main__":
    download_all_models()
    print("-" * 50)
    print("\n✅ Model setup complete. All required models are in place.")