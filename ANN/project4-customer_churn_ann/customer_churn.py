import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler






df = pd.read_csv("/Users/vrohitha/Desktop/neuronnetworks/ANN/project4-customer_churn_ann/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())

print(df.info())

print(df.isnull().sum())

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values again
print(df.isnull().sum())

# Remove missing rows
df.dropna(inplace=True)
print(df.isnull().sum())
print(df.shape)
binary_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn"
]

label_encoder = LabelEncoder()

binary_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn"
]

for column in binary_columns:
    df[column] = label_encoder.fit_transform(df[column])
print(df.head())

categorical_columns = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

print(df.head())
print(df.shape)

X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)
class ANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(30, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 2)

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x


model = ANN()
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(500):

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# -------------------------------
# Testing
# -------------------------------

model.eval()

with torch.no_grad():

    predictions = model(X_test)

    _, predicted = torch.max(predictions, 1)

accuracy = accuracy_score(
    y_test.numpy(),
    predicted.numpy()
)

print(f"\nAccuracy: {accuracy*100:.2f}%")
