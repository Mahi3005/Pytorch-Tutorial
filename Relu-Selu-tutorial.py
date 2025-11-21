import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        self.layer2 = nn.Linear(hidden_size, output_size)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.selu(x)

        return x   # return ONLY final output


input_size = 10
hidden_size = 20
output_size = 1

model = SimpleNN(input_size, hidden_size, output_size)

input_data = torch.randn(5, input_size)
output_data = model(input_data)

print(output_data)
