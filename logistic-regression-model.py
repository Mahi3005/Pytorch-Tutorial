import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

# Generate dataset
X, y = make_classification(
    n_samples=1000,
    n_features=17,
    n_informative=10,
    n_redundant=7,
    n_classes=2,
    random_state=21
)

n_samples, n_features = X.shape

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert NumPy to Tensor
X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor  = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
y_test_tensor  = torch.from_numpy(y_test).float().view(-1, 1)

# Logistic Regression NN with ELU
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.layer1 = nn.Linear(n_input_features, 20)
        self.elu = nn.ELU()
        self.layer2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.elu(self.layer1(x))
        y_pred = torch.sigmoid(self.layer2(x))
        return y_pred

model = LogisticRegression(n_features)

# Loss + Optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
num_epoch = 5000
for epoch in range(num_epoch):
    outputs = model(X_train_tensor)
    
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}")

#model evaluation

# ----- EVALUATION -----

model.eval()   # set model to evaluation mode

with torch.no_grad():
    # Forward pass on train & test
    train_outputs = model(X_train_tensor)
    test_outputs  = model(X_test_tensor)

# Convert probabilities → class labels (0 or 1)
train_preds = (train_outputs >= 0.5).float()
test_preds  = (test_outputs  >= 0.5).float()

# Accuracy
train_accuracy = (train_preds.eq(y_train_tensor).sum().item() / len(y_train_tensor)) * 100
test_accuracy  = (test_preds.eq(y_test_tensor).sum().item() / len(y_test_tensor)) * 100

print(f"\nTrain Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy:  {test_accuracy:.2f}%")

    
