import torch
import cv2
import numpy as np
import ultralytics

print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Ultralytics:", ultralytics.__version__)

print("CUDA Available:", torch.cuda.is_available())

print("\nAll Libraries Installed Successfully!")
