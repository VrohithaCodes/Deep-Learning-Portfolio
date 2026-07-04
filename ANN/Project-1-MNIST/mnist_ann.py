import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

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
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)
import matplotlib.pyplot as plt

plt.imshow(images[0].squeeze(), cmap="gray")
plt.title(f"Label: {labels[0]}")
plt.show()
import torch.nn as nn
class ANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):

        x = x.view(-1, 28 * 28)

        x = self.layer1(x)

        x = self.relu(x)

        x = self.layer2(x)

        return x
model = ANN()
print(model)
criterion = nn.CrossEntropyLoss()
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)    
for epoch in range(5):
    for images, labels in train_loader:

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        output = model(images)

        _, predicted = torch.max(output, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")        