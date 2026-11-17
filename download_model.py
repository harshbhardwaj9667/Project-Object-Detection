# Import YOLO
from ultralytics import YOLO

# Load pre-trained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Display confirmation
print("YOLO Model Downloaded Successfully!")

print("Model Name:", model.model_name if hasattr(model, "model_name") else "YOLOv8 Nano")