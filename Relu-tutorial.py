# #so Relu() is an activation fucntions which makes network non linear so now we can give the neural network the ability to learn complex patterns.
# Without activation function, no matter how big your model is, it behaves like simple linear regression.

# Relu(x)=max(0,x)

#which means for negative value the number is 0 and for the positive value the number is the number itself  

#now we will create a simple neural network with the help of leaky ReLu

import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
  def __init__(self,input_size,hidden_size,output_size):
    super(SimpleNN,self).__init__()
    self.layer1=nn.Linear(input_size,hidden_size)
    self.LeakyReLu=nn.LeakyReLU(negative_slope=0.01)
    self.layer2=nn.Linear(hidden_size,output_size)

  def forward(self,x):
    x=self.layer1(x)
    x=self.LeakyReLu(x)
    x=self.layer2(x)
    return x
  

input_size=10
hidden_size=20
output_size=1

model=SimpleNN(input_size,hidden_size,output_size)

input_data=torch.randn(5,input_size)
output_data=model(input_data)

print(input_data)
print(output_data)

    