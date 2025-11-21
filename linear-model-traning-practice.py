import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

X=np.array([x for x in range(100)])
X=X.reshape(-1,1)  #only single col
# print(X)

y = 2 * X.flatten() + 46

# print(y)

#now we will normalization

x_mean = X.mean()
x_std=X.std()
y_mean=y.mean()
y_std=y.std()

X_normalized = (X-x_mean)/x_std
y_normalized = (y-y_mean)/y_std

# print(X_normalized)
# print(y_normalized)  as from the result we can see that the data is now normalized


# now we will create the final tensors

X_tensor = torch.tensor(X_normalized,dtype=torch.float32)
y_tensor=torch.tensor(y_normalized,dtype=torch.float32)

# print(X_tensor.shape)
# print(y_tensor.shape)


## now we will create a linear regression model

class LinearRegressionModel(nn.Module):
  def __init__(self,in_features,out_features):
    super().__init__()
    self.linear=nn.Linear(in_features,out_features)

  def forward(self,x):
    return self.linear(x).squeeze(1)



in_features=1
out_features=1

model=LinearRegressionModel(in_features,out_features)

##now we will calculate the loss and try to optimize it

# print(model.parameters())

criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.1) # parameter = parameter - learnig rate x Gradient

#now we will train the model

num_epoch=20

for epochs in range(num_epoch):
  #forward pass

  outputs=model(X_tensor)

  #calculate the loss

  loss=criterion(outputs,y_tensor)

  #now backward pass and optimization to decrese the loss

  optimizer.zero_grad()
  loss.backward() ##will calculate the new gradients here
  optimizer.step() #here will we updatet the weight and bias

  print(f'Epoch [{epochs + 1} / {num_epoch}], loss : {loss.item():.2f}') # as we training it on 20 epochs we will get the exact output that we cna see in the console as loss=0.0 in the 20th epoch

  new_x = 121

new_x_normalized = (new_x - x_mean) / x_std

new_x_tensor=torch.tensor(new_x_normalized,dtype=torch.float32).view(1,-1)

model.eval()

with torch.no_grad():
  prediction_normalized = model(new_x_tensor)


prediction_denormalized = prediction_normalized.item() * y_std + y_mean

print(f"Predictedvalue for x ={new_x} : {prediction_denormalized}")


#now we will plot the new input with generalize model to check the accuracy of the output

# Plot the original data
plt.figure(figsize=(10, 6))  # bigger, cleaner plot
plt.scatter(X, y, label="Original Data", s=40, alpha=0.7)

# Predicted line (denormalized)
fit_line = model(X_tensor).detach().numpy() * y_std + y_mean

plt.plot(X, fit_line, 
         label="Fitted Line (PyTorch)",
         linewidth=2)

# Add labels, title, grid
plt.xlabel("X", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.title("Linear Regression Fit Using PyTorch", fontsize=16)

plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)  # light grid for readability
plt.tight_layout()
plt.show()