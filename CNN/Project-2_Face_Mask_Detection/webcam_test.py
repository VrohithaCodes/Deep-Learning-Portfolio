import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ---------------------------------------
# Image Transform
# ---------------------------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---------------------------------------
# CNN Model
# ---------------------------------------

class FaceMaskCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(25088, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(-1, 25088)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ---------------------------------------
# Load Trained Model
# ---------------------------------------

model = FaceMaskCNN()

model.load_state_dict(
    torch.load(
        "/Users/vrohitha/Desktop/neuronnetworks/CNN/Project-2_Face_Mask_Detection/face_mask_cnn.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

classes = ["With Mask", "Without Mask"]

# ---------------------------------------
# Load Haar Cascade
# ---------------------------------------

face_detector = cv2.CascadeClassifier(
    "/Users/vrohitha/Desktop/neuronnetworks/CNN/Project-2_Face_Mask_Detection/face_detector/haarcascade_frontalface_default.xml"
)

# ---------------------------------------
# Open Webcam
# ---------------------------------------

cap = cv2.VideoCapture(0)

# ---------------------------------------
# Webcam Loop
# ---------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        # Crop Face
        face = frame[y:y+h, x:x+w]

        # Convert BGR -> RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # PIL Image
        face = Image.fromarray(face)

        # Transform
        face = transform(face)

        # Batch Dimension
        face = face.unsqueeze(0)

        # Prediction
        with torch.no_grad():

            output = model(face)

            _, predicted = torch.max(output, 1)

        label = classes[predicted.item()]

        # Draw Rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        # Draw Prediction
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("Live Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()