
# ==========================================
# YOLO OBJECT DETECTION USING STREAMLIT
# ==========================================

# Import libraries
import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="",
    layout="centered"
)

st.title("YOLO Object Detection")
st.write("Upload an image to detect objects using YOLOv8 Nano.")

# ------------------------------------------
# LOAD YOLO MODEL
# ------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------------------
# OBJECT DETECTION
# ------------------------------------------

if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display original image
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Convert image into NumPy array
    image_array = np.array(image)

    # Perform YOLO detection
    if st.button("Detect Objects"):

        with st.spinner("Detecting objects..."):

            results = model.predict(
                source=image_array,
                imgsz=320,
                conf=0.35,
                device="cpu",
                max_det=20,
                verbose=False
            )

            result = results[0]

            # Draw bounding boxes
            detected_image = result.plot()

            # Convert BGR to RGB
            detected_image = cv2.cvtColor(
                detected_image,
                cv2.COLOR_BGR2RGB
            )

        # Display detected image
        st.subheader("Detection Results")

        st.image(
            detected_image,
            use_container_width=True
        )

        # ----------------------------------
        # DISPLAY DETECTION SUMMARY
        # ----------------------------------

        st.subheader("Detected Objects")

        object_counts = {}

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            object_name = model.names[class_id]

            object_counts[object_name] = (
                object_counts.get(object_name, 0) + 1
            )

        # Total objects
        st.metric("Total Objects", len(result.boxes))

        # Individual object counts
        for name, count in object_counts.items():

            st.write(f"**{name}:** {count}")

        st.success("Object Detection Completed!")
