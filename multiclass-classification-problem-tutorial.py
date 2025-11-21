import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data
y=iris.target



iris_df=pd.DataFrame({
'x1': X[:,0],
'x2': X[:,1],
'x3': X[:,2],
'x4': X[:,3],

})

# 3. Split BEFORE scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Scale ONLY the feature data, NOT y
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor=torch.tensor(X_train,dtype=torch.float32)
X_test_tensor=torch.tensor(X_test,dtype=torch.float32)
y_train_tensor=torch.tensor(y_train,dtype=torch.int64)
y_test_tensor=torch.tensor(y_test,dtype=torch.int64)

class SimpleClassifier(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.layer_1=nn.Linear(in_features,120)
    self.layer_2=nn.Linear(120,10)
    self.layer_3=nn.Linear(10,out_features)

  def forward(self,x):
    x=self.layer_3(self.layer_2(self.layer_1(x)))
    return x
  

in_features=X_train.shape[1]
num_classes = len(set(y))
model = SimpleClassifier(in_features,num_classes)

# we will calculate the loss and optimizers

criterion  = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(),lr=0.1)



## now we will train the model

num_epochs=200
for epoch in range(num_epochs):
  outputs=model(X_train_tensor)
  loss=criterion(outputs,y_train_tensor)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


  print(f'Epoch [{epoch + 1} / {num_epochs}], loss : {loss.item():.2f}')


  # EVALUATION
model.eval()  # set model to evaluation mode

with torch.no_grad():
    train_outputs = model(X_train_tensor)
    test_outputs = model(X_test_tensor)

# Convert logits → predicted class (argmax)
train_preds = torch.argmax(train_outputs, dim=1)
test_preds = torch.argmax(test_outputs, dim=1)

# Accuracy
train_acc = accuracy_score(y_train, train_preds.numpy())
test_acc = accuracy_score(y_test, test_preds.numpy())

print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy:  {test_acc*100:.2f}%")