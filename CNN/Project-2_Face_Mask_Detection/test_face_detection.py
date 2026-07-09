import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Load the image
image = cv2.imread("/Users/vrohitha/Desktop/neuronnetworks/CNN/Project-2_Face_Mask_Detection/test_image/photo 2.jpeg")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the Haar Cascade face detector
face_detector = cv2.CascadeClassifier(
    "/Users/vrohitha/Desktop/neuronnetworks/CNN/Project-2_Face_Mask_Detection/face_detector/haarcascade_frontalface_default.xml"
)

# Detect faces
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Faces Found:", len(faces))

# Draw green rectangles around detected faces
for (x, y, w, h) in faces:

    face = image[y:y+h, x:x+w]

    cv2.rectangle(
        image,
        (x, y),
        (x+w, y+h),
        (0,255,0),
        2)

    cv2.imshow("Detected Face", face)
    print(face.shape)
cv2.imshow("Face Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

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
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 25088)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
model = FaceMaskCNN()

model.load_state_dict(
    torch.load("/Users/vrohitha/Desktop/neuronnetworks/CNN/Project-2_Face_Mask_Detection/face_mask_cnn.pth")
)
model.eval()
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

face = Image.fromarray(face)

face = transform(face)

face = face.unsqueeze(0)
with torch.no_grad():

    output = model(face)

    _, predicted = torch.max(output, 1)

    classes = ["With Mask", "Without Mask"]

label = classes[predicted.item()]

print(label)

cv2.putText(
    image,
    label,
    (x, y-10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0,255,0),
    2
)
 
    

