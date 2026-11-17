# Simple Yolo object detction-project
# Import necessary libraries
from ultralytics import YOLO    
from pathlib import Path

# step1: Define file paths
image_path = Path("input/input.jpg")  # Path to the input image
output_path = Path("output/detected_image.jpg")  # Path to save the output image

# step2:check input image
if not image_path.exists():
    raise FileNotFoundError("Input image not found !")

# step3:create output folder
output_path.parent.mkdir(parents=True, exist_ok=True)

# step4: load YOLO Nano model
model = YOLO("yolov8n.pt")  # Load the pre-trained YOLOv8 Nano model

# perform object detection 
results = model.predict(
    source=str(image_path), 
    imgsz=320,
    conf=0.35,
    device="cpu",
    max_det=20,
    verbose=False
    )  # Perform object detection and save results

# get detection results
result = results[0]  # Get the first result (since we are processing a single image)
result.save(filename=str(output_path))

# STEP 8: Display detected objects

print("\nDETECTED OBJECTS")
print("------------------------")

for box in result.boxes:

    # Get class ID
    class_id = int(box.cls[0])

    # Get confidence score
    confidence = float(box.conf[0])

    # Get object name
    object_name = model.names[class_id]

    print(f"Object: {object_name}")
    print(f"Confidence: {confidence:.2f}")
    print("------------------------")

# STEP 9: Completion message


print("\nObject Detection Completed!")
print("Output Image:", output_path)