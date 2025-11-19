import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

X=np.array([x for x in range(100)])
X=X.reshape(-1,1)
y=2*X.flatten()+46 ## this is a linear regression formula for y=mx+c here m = 2 and c = 46, m is slope and c is a intercept in the linear graph

# print(X)
# print(y)

#now we will plot it before the pytorch as the initial values

# plt.scatter(X,y,label="initial data")
# plt.title("Pre Pytorch")
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()
# plt.show() # this will plot a simple linear regression graph

#now we will normalize the data

x_mean,x_std=X.mean(),X.std()

X_normalized=(X-x_mean)/x_std

X_tensor = torch.tensor(X_normalized,dtype=torch.float32)

print(X_tensor.shape)

y_mean, y_std = y.mean(), y.std()

y_normalized = (y - y_mean) / y_std

y_tensor = torch.tensor(y_normalized, dtype=torch.float32)

print(y_tensor.shape)



## now we are creating a linear regresssion class to train the model


class LinearRegressionModel(nn.Module):
  def __init__(self,in_features,out_features):
    super().__init__()
    self.linear = nn.Linear(in_features,out_features)
  
  def forward(self,x):
    return self.linear(x).squeeze(1)
  

in_features=1
out_features=1


model=LinearRegressionModel(in_features,out_features)

criterion=nn.MSELoss()

optimizer=optim.SGD(model.parameters(),lr=0.1)

# parameter = parameter - learnig rate x Gradient


#epochs = how many times we are going to run our models

num_epochs = 10

for epoch in range(num_epochs):

  #forward pass
  outputs=model(X_tensor)
  #calculated loss
  loss = criterion(outputs,y_tensor)
  # backward pass and optimization is here
  optimizer.zero_grad()
  loss.backward()  ## here we are computing the gradients
  optimizer.step()  ## this will updated the weights

  print(f'Epoch [{epoch + 1} / {num_epochs}], loss : {loss.item():.2f}')


new_x = 121

new_x_normalized = (new_x - x_mean) / x_std

new_x_tensor=torch.tensor(new_x_normalized,dtype=torch.float32).view(1,-1)

model.eval()

with torch.no_grad():
  prediction_normalized = model(new_x_tensor)


prediction_denormalized = prediction_normalized.item() * y_std + y_mean

print(f"Predictedvalue for x ={new_x} : {prediction_denormalized}")




plt.scatter(X,y,label="initial data")
fit_line=model(X_tensor).detach().numpy()*y_std+y_mean
plt.plot(X,fit_line,label='pytorch line')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')

plt.title("Pytorch with predictions")
plt.show()