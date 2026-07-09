import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
    torch.load("/Users/vrohitha/Desktop/neuronnetworks/CNN/project_2_face_maskrecognization/face_mask_cnn.pth")
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

image = Image.open(
    "/Users/vrohitha/Desktop/neuronnetworks/CNN/project_2_face_maskrecognization/test_image/testimg.jpg"
).convert("RGB")

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

with torch.no_grad():

    outputs = model(image_tensor)
    probabilities = F.softmax(outputs, dim=1)

    _, predicted = torch.max(outputs, 1)
    confidence = probabilities[0][predicted.item()] * 100

classes = [
    "With Mask",
    "Without Mask"
]    

print(f"Prediction : {classes[predicted.item()]}")
print(f"Confidence : {confidence:.2f}%")

plt.imshow(image)
plt.title(
    f"{classes[predicted.item()]}\nConfidence: {confidence:.2f}%"
)
plt.axis("off")
plt.show()

