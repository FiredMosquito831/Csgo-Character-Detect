from ultralytics import YOLO
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

model = YOLO('/runs/CurrentBEST/train2/weights/best.pt').to(device)

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolov8n.engine'

# Load the exported TensorRT model
#tensorrt_model = YOLO('C:/Users/fgghk/PycharmProjects/TranscriptionAppRelease/CsgoAiDetectCharacters/runs/detect/train2/weights/best.engine')
