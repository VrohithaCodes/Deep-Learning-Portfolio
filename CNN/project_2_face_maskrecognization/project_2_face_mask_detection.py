import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_dataset = datasets.ImageFolder(
    root="/Users/vrohitha/Desktop/neuronnetworks/data/Face-Mask-Detection-master/dataset/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="/Users/vrohitha/Desktop/neuronnetworks/data/Face-Mask-Detection-master/dataset/test",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("Training Images :", len(train_dataset))
print("Testing Images :", len(test_dataset))

print("Classes :", train_dataset.classes)

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

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(10):

    model.train()

    for images, labels in train_loader:

        # 1. Clear old gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(images)

        # 3. Calculate loss
        loss = criterion(outputs, labels)

        # 4. Backpropagation
        loss.backward()

        # 5. Update weights
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")

import matplotlib.pyplot as plt

classes = train_dataset.classes

images, labels = next(iter(test_loader))

model.eval()

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

plt.figure(figsize=(12,8))

for i in range(6):

    plt.subplot(2,3,i+1)

    img = images[i].permute(1,2,0).numpy()

    img = img * 0.5 + 0.5

    plt.imshow(img)

    plt.title(
        f"Pred: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}"
    )

    plt.axis("off")

plt.show()

torch.save(
    model.state_dict(),
    "face_mask_cnn.pth"
)

print("Model Saved Successfully!")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")