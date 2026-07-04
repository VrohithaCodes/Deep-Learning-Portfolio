import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# -------------------------------
# Step 1: Create Dataset
# -------------------------------

data = {
    "Area": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
    "Age": [20, 15, 10, 8, 5, 3, 2, 1, 1, 1],
    "Price": [35, 42, 55, 65, 80, 95, 105, 120, 135, 150]
}

df = pd.DataFrame(data)

df.to_csv("house_prices.csv", index=False)

print("CSV Created Successfully!\n")

# -------------------------------
# Step 2: Read CSV
# -------------------------------

df = pd.read_csv("house_prices.csv")

print(df)

# -------------------------------
# Step 3: Separate Inputs and Output
# -------------------------------

X = df.drop("Price", axis=1)
y = df["Price"]

# -------------------------------
# Step 4: Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# Step 5: Normalize Inputs
# -------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# -------------------------------
# Step 6: Convert to Tensors
# -------------------------------

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

y_test = torch.tensor(
    y_test.values,
    dtype=torch.float32
).view(-1, 1)

# -------------------------------
# Step 7: Build ANN
# -------------------------------

class ANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(3, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 1)

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x


model = ANN()

# -------------------------------
# Step 8: Loss and Optimizer
# -------------------------------

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# -------------------------------
# Step 9: Training
# -------------------------------

for epoch in range(100):

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        model.eval()

with torch.no_grad():
    train_predictions = model(X_train)

print(train_predictions)
print(y_train)    
if epoch == 0:
    print(outputs[:5])

if epoch == 99:
    print(outputs[:5])

print(model.layer1.weight)
print(model.layer2.weight)    

    

# -------------------------------
# Step 10: Testing
# -------------------------------
print("\nFinal Training Loss:", loss.item())
model.eval()

with torch.no_grad():

    predictions = model(X_test)

print("\nPredicted Prices:")
print(predictions)

print("\nActual Prices:")
print(y_test)

# -------------------------------
# Step 11: Calculate MSE
# -------------------------------

mse = mean_squared_error(
    y_test.numpy(),
    predictions.numpy()
)

print("\nMean Squared Error:", mse)

# -------------------------------
# Step 12: Predict New House
# -------------------------------

new_house = [[1700, 3, 6]]

new_house = scaler.transform(new_house)

new_house = torch.tensor(
    new_house,
    dtype=torch.float32
)

with torch.no_grad():

    predicted_price = model(new_house)

print("\nPredicted Price for New House:")
print(predicted_price)