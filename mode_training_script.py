import os
import torch
from ultralytics import YOLO

# Set device to CUDA if available, otherwise CPU
device = torch.device(0 if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the path to the dataset configuration file
DATASET_YAML = os.path.join('datasets', 'gtathermalAiDetect-3',
                            'data.yaml')  # Use os.path.join for cross-platform compatibility


def main():
    # Load a pre-trained YOLOv8 model (Small in this case)
    model = YOLO('../GtaModels/S45images/train4/weights/best.pt')  # You can replace with 'yolov8n.pt' for nano, etc.

    # Check if the dataset exists at the specified path
    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(f"Dataset YAML file not found at {DATASET_YAML}. Please check the dataset path.")

    print(f"Training using dataset: {DATASET_YAML}")

    # Train the model
    model.train(
        data=DATASET_YAML,  # Dataset YAML file path
        epochs=11,  # Number of training epochs
        batch=32,  # Batch size
        workers=12,  # Number of data loader workers
        device=0,  # Use GPU (0) or CPU (as detected by 'torch.device')
        #save_period=10, # Save model every time 10 epochs
        #patience=7,  # Stop training if no improvement for 10 epochs
        amp=True,  # Add this to the train method to enable AMP
        imgsz=640 # image size to send into model

    )

    # Validate the model after training
    model.val()

    # Save the trained model in the specified format
    model.export(format='TensorRT')  # You can export in other formats like ONNX, CoreML, .pt(default), etc.


# Protect the entry point of the program
if __name__ == "__main__":
    # Required for Windows multiprocessing to avoid RuntimeError
    torch.multiprocessing.freeze_support()
    main()
