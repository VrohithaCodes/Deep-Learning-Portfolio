# 😷 Face Mask Detection using Convolutional Neural Networks (CNN)

## 📌 Project Overview

This project implements a **Face Mask Detection System** using **PyTorch** and a **Convolutional Neural Network (CNN)**. The model classifies an uploaded face image into one of two categories:

* 😷 With Mask
* 🙂 Without Mask

A simple **Streamlit web application** was developed to allow users to upload an image and receive the prediction along with the confidence score.

---

# 🚀 Features

* Image classification using CNN
* Binary classification (With Mask / Without Mask)
* Image preprocessing using TorchVision
* Train/Test dataset split
* Model saving and loading
* Prediction on custom images
* Confidence score using Softmax
* Streamlit web interface

---

# 🛠 Technologies Used

* Python
* PyTorch
* TorchVision
* Streamlit
* Pillow (PIL)
* Matplotlib

---

# 📂 Dataset

The project uses the Face Mask Detection dataset containing two classes:

```
dataset/
│
├── with_mask/
└── without_mask/
```

The dataset was split into:

* 80% Training
* 20% Testing

---

# 🧠 CNN Architecture

```
Input Image
      │
Conv2D (3 → 32)
      │
ReLU
      │
MaxPooling
      │
Conv2D (32 → 64)
      │
ReLU
      │
MaxPooling
      │
Conv2D (64 → 128)
      │
ReLU
      │
MaxPooling
      │
Flatten
      │
Fully Connected (25088 → 256)
      │
ReLU
      │
Fully Connected (256 → 2)
      │
Prediction
```

---

# ⚙ Training Details

* Optimizer: Adam
* Learning Rate: 0.0001
* Loss Function: CrossEntropyLoss
* Batch Size: 32
* Epochs: 10

---

# 📊 Model Performance

Training Images: **3275**

Testing Images: **819**

Test Accuracy:

**94.75%**

---

# 🌐 Streamlit Application

The web application allows users to:

* Upload an image
* Display the uploaded image
* Predict mask status
* Display confidence score

---

# 📷 Example Output

```
Prediction:
😷 With Mask

Confidence:
89.15%
```

---

# Challenges Faced

During the development of this project, several practical challenges were encountered:

* Understanding how image datasets should be organized for PyTorch.
* Splitting the dataset into training and testing folders.
* Designing the CNN architecture manually.
* Selecting appropriate convolutional layers and fully connected layers.
* Determining the correct input size (25088) for the fully connected layer after flattening.
* Understanding tensor dimensions after every convolution and pooling operation.
* Saving and loading trained model weights correctly.
* Building a Streamlit web application from scratch.
* Integrating the trained CNN model into the web interface.
* Debugging prediction and model loading issues.

Each of these challenges helped improve understanding of deep learning workflows and model deployment.

---

# Limitations

Although the model achieved **94.75% testing accuracy**, testing on completely unseen real-world images revealed some incorrect predictions.

Observed limitations include:

* **Incorrect predictions on some real-world face images.**
* Lower performance under different lighting conditions.
* Sensitivity to background variations.
* Performance decreases for images significantly different from the training dataset.
* The current model predicts only a single face in an image.
* Multiple people in a single image are not supported.

These limitations occur because the model was trained only on the provided dataset and has limited exposure to diverse real-world conditions.

---

# Future Improvements

Future versions of this project can include:

* **Data augmentation (Random Flip, Rotation, Color Jitter)**
* **Transfer Learning using ResNet18 or MobileNetV2**
* Face Detection before classification
* Real-time webcam mask detection
* Multiple face detection in a single image
* Model deployment on cloud platforms
* Mobile application integration

---

# Learning Outcomes

This project helped in understanding:

* CNN architecture
* Image preprocessing
* Binary image classification
* PyTorch model implementation
* Training and testing pipelines
* Model serialization
* Deep learning deployment using Streamlit

---

# Conclusion

This project demonstrates a complete deep learning workflow—from dataset preparation and CNN model development to deployment as an interactive web application. While the model performs well on the testing dataset, evaluating it on real-world images highlighted important practical limitations. These observations provide valuable insights for future improvements using transfer learning, data augmentation, and face detection, making this project a strong foundation for more advanced computer vision applications.
