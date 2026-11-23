🎯 YOLO Object Detection Web App

Real-time object detection in your browser, powered by YOLOv8 and Streamlit.
🚀 Try the Live Demo @ https://project-object-detection-qcxbl7pnhec8egf4kknrh2.streamlit.app/
📌 Overview

This project is an end-to-end computer vision application that detects and labels objects in any uploaded image. It uses YOLOv8 Nano (yolov8n.pt), a lightweight, fast, state-of-the-art detection model from Ultralytics, and wraps it in a clean Streamlit interface so anyone can try it without installing anything.

Upload an image, click Detect Objects, and instantly get:

The image annotated with bounding boxes and class labels
The total number of objects detected
A per-class count (e.g. person: 3, car: 2)

The model is pretrained on the COCO dataset, so it recognises 80 everyday object classes such as people, vehicles, animals, and household items.
✨ Features
📤 Simple image upload supporting JPG, JPEG, and PNG
🧠 YOLOv8n inference with bounding boxes and class labels drawn on the image
📊 Detection summary with total object count and a per-class breakdown
⚡ Optimised for CPU so it runs smoothly on free cloud hosting
🗂️ Cached model loading via st.cache_resource, so the model loads once and not on every interaction
🖥️ Clean, minimal UI with a loading spinner and success feedback
🛠️ Tech Stack
Category	Tools
Language	Python
Model	YOLOv8 Nano (Ultralytics)
Computer Vision	OpenCV, Pillow, NumPy
Web Framework	Streamlit
Deployment	Streamlit Community Cloud
⚙️ How It Works
 Upload image  →  Convert to RGB / NumPy array  →  YOLOv8n inference
        →  Draw bounding boxes  →  BGR → RGB conversion  →  Display results + object counts
The user uploads an image through the Streamlit file uploader.
The image is converted to RGB and then to a NumPy array.
YOLOv8n runs inference with these settings:
Parameter	Value	Purpose
imgsz	320	Smaller input size for faster CPU inference
conf	0.35	Minimum confidence threshold to filter weak detections
max_det	20	Cap on detections per image
device	cpu	Runs without needing a GPU
Detected boxes are drawn on the image with result.plot(), then converted from BGR to RGB for display.
Class IDs are mapped to names and tallied to produce the summary.
📁 Project Structure
Project-Object-Detection/
├── app.py               # Streamlit web application
├── main.py              # Core detection script
├── download_model.py    # Helper to download the YOLO weights
├── installation.py      # Environment / dependency setup helper
├── requirements.txt     # Python dependencies
├── yolov8n.pt           # Pretrained YOLOv8 Nano weights
├── input/               # Sample input images
└── output/              # Detection results
🚀 Getting Started
Prerequisites
Python 3.9 or higher
pip
Installation
bash
# 1. Clone the repository
git clone https://github.com/harshbhardwaj9667/Project-Object-Detection.git
cd Project-Object-Detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
Run the app
bash
streamlit run app.py

Then open http://localhost:8501 in your browser.

📸 Usage
Open the app (live or local).
Click Browse files and upload a JPG or PNG image.
Click Detect Objects.
View the annotated image, the total object count, and the per-class breakdown.
🔮 Future Improvements
[ ] Real-time detection from webcam / video streams
[ ] Adjustable confidence threshold slider in the UI
[ ] Selectable model sizes (YOLOv8s / m / l) for accuracy vs. speed trade-offs
[ ] Video file upload and processing
[ ] Downloadable annotated output
[ ] Custom-trained model on a domain-specific dataset
📚 What I Learned
Integrating a pretrained deep learning model into a user-facing web app
Optimising inference for CPU-only environments (input size, confidence threshold, detection cap)
Using Streamlit caching to avoid redundant model loading
Handling colour-space conversions between PIL, NumPy, and OpenCV (RGB ↔ BGR)
Deploying an ML app end to end on Streamlit Community Cloud
🙌 Acknowledgements
Ultralytics YOLOv8
COCO Dataset
Streamlit
👤 Author
https://github.com/harshbhardwaj9667
LinkedIn: https://www.linkedin.com/in/harsh-bhardwaj-66a4a4218/
⭐ If you found this project useful, consider giving it a star!
