# Face Mask Detection using CNN

## Version 1

### Features
- CNN built using PyTorch.
- Image classification (Mask / No Mask).
- Streamlit web application.
- Prediction on uploaded images.

---

## Version 2 (OpenCV + CNN)

### Features
- OpenCV Haar Cascade Face Detection.
- Automatic face detection.
- Automatic face cropping.
- CNN predicts whether the detected face is wearing a mask.
- Bounding box around the detected face.
- Prediction displayed above the detected face.

---

## Workflow

Image
↓
OpenCV Face Detection
↓
Face Cropping
↓
Image Preprocessing
↓
CNN Prediction
↓
Bounding Box + Label

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- Pillow
- Streamlit
- Torchvision
- Matplotlib

---

## Results

- CNN Test Accuracy: **94.75%**
- Successfully detects frontal faces.
- Correctly predicts mask/no-mask on most dataset-like images.

---

## Challenges Faced

- Haar Cascade struggles with:
  - Small faces
  - Tilted faces
  - Poor lighting
  - Occluded faces

- Incorrect face detection leads to incorrect CNN predictions because the cropped image is inaccurate.

---

## Future Improvements

- Replace Haar Cascade with YOLO Face Detection.
- Real-time webcam detection.
- Improve performance on multiple faces.
- Deploy online using Streamlit Cloud.

---

## Lessons Learned

During this project, I learned:

- Building CNNs using PyTorch.
- Training and evaluating image classification models.
- Saving and loading trained models.
- Integrating OpenCV with deep learning.
- Face detection using Haar Cascades.
- Image preprocessing for inference.
- Deploying AI models using Streamlit.
- Writing professional documentation and maintaining projects using GitHub.