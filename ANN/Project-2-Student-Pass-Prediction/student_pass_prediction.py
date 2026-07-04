import pandas as pd
from sklearn.metrics import accuracy_score
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("/Users/vrohitha/Desktop/neuronnetworks/student_result - Sheet1 (1).csv")

print(data.head())
X = data[["Hours_Studied", "Attendance", "Previous_Marks"]]

y = data["Pass"]
print(X.head())

print(y.head())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)
import torch.nn as nn

class StudentANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(3, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 2)

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x
model = StudentANN()

print(model)
criterion = nn.CrossEntropyLoss()

import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(100):

    output = model(X_train)

    loss = criterion(output, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()

with torch.no_grad():

    output = model(X_test)

    predicted = torch.argmax(output, dim=1)

    print("Predicted:", predicted)
    print("Actual:", y_test)
    
accuracy = accuracy_score(
    y_test.numpy(),
    predicted.numpy()
)

print(f"Accuracy: {accuracy * 100:.2f}%")    

print(f"Accuracy: {accuracy * 100:.2f}%")

new_student = [[7, 90, 72]]

new_student = scaler.transform(new_student)

new_student = torch.tensor(new_student, dtype=torch.float32)

model.eval()

with torch.no_grad():

    prediction = model(new_student)

    predicted_class = torch.argmax(prediction, dim=1)

    print(predicted_class)