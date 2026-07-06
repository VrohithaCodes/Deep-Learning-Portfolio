import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

# -----------------------------
# Dataset
# -----------------------------
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# -----------------------------
# DataLoader
# -----------------------------
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

# -----------------------------
# CNN Model
# -----------------------------
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x


# -----------------------------
# Model
# -----------------------------
model = CNN()

# -----------------------------
# Loss
# -----------------------------
criterion = nn.CrossEntropyLoss()

# -----------------------------
# Optimizer
# -----------------------------
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------
# Training
# -----------------------------
num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# -----------------------------
# Testing
# -----------------------------
import matplotlib.pyplot as plt

model.eval()

images, labels = next(iter(test_loader))

with torch.no_grad():

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

plt.figure(figsize=(12,6))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.imshow(images[i].squeeze(), cmap="gray")

    plt.title(f"P:{predicted[i].item()}\nT:{labels[i].item()}")

    plt.axis("off")

plt.tight_layout()
plt.show()