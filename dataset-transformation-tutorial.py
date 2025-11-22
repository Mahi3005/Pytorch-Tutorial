import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np

class TabularData(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor:
    def __call__(self, sample):
        features, label = sample
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

class Normalization:
    def __call__(self, sample):
        features, label = sample
        normalized = (features - np.mean(features)) / np.std(features)
        return (normalized, label)

# Dummy data
dummy_data = [(np.random.rand(2), np.random.rand()) for _ in range(100)]

transform = transforms.Compose([Normalization(), ToTensor()])

dataset = TabularData(data=dummy_data, transform=transform)

#now we will create a dataloader 

dataloader = DataLoader(dataset , batch_size=16 , shuffle=True)

##now we will create a neural network

class SimpleNN(nn.Module):
    def __init__(self,input_size):
        super(SimpleNN,self).__init__()
        self.fc = nn.Linear(input_size,1)

    def forward(self,x):
        x=self.fc(x)
        return x
    

model = SimpleNN(input_size=2)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

num_epoch = 10

for epoch in range(num_epoch):
    model.train()
    total_loss=0

    for batch in dataloader:
        features , label = batch['features'] , batch['label']

        outputs=model(features)
        loss=criterion(outputs,label.view(-1,1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    
    print(f"Epoch {epoch+1}/{num_epoch}, Loss: {total_loss / len(dataloader):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    sample = torch.tensor([[0.5, 0.7]], dtype=torch.float32)   # dummy input
    prediction = model(sample)
    print("\nExample prediction:", prediction.item())