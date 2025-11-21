import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

digit = load_digits()

X=digit.data
y=digit.target

X_train , X_test ,y_train , y_test = train_test_split(X,y,random_state=42,test_size=0.2)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#now we will make our custum data sets as well

class CustomDataset(Dataset):
  def __init__(self,data,target):
    self.data=torch.tensor(data,dtype=torch.float32)
    self.target=torch.tensor(target,dtype=torch.long)

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    sample = {'data':self.data[index],'target':self.target[index]}
    return sample
  
#now we will create dataset using this class

train_dataset = CustomDataset(X_train,y_train)
test_dataset = CustomDataset(X_test,y_test)


# print(len(train_dataset))
# print(len(test_dataset))

#now we will move to the data loader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,     # you can change batch size
    shuffle=True       # shuffle only for training
)


test_loader = DataLoader(
    test_dataset,
    batch_size=32,     # same batch size
    shuffle=False      # do NOT shuffle test data
)

## now we make a model

class SimpleNN(nn.Module):
  def __init__(self,input_size,hidden_size,output_size):
    super(SimpleNN,self).__init__()
    self.layer1=nn.Linear(input_size,hidden_size)
    self.relu=nn.ReLU()
    self.layer2=nn.Linear(hidden_size,output_size)

  
  def forward(self,x):
    x=self.layer1(x)
    x=self.relu(x)
    x=self.layer2(x)

    return x


input_size=X_train.shape[1]
hidden_size=64
output_size=len(set(y_train))

model=SimpleNN(input_size,hidden_size,output_size)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

num_epoch=100
for epoch in range(num_epoch):
  model.train()
  running_loss=0.0

  for batch in train_loader:
    inputs=batch['data']
    targets=batch['target']

    outputs=model(inputs)
    loss=criterion(outputs,targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss+=loss.item()


  print(f"Epoch {epoch+1}/{num_epoch}, loss {running_loss/len(train_loader)}/")
  

#model evaluation

model.eval()   # evaluation mode

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs  = batch['data']
        targets = batch['target']

        outputs = model(inputs)                  # raw logits
        _, predicted = torch.max(outputs, 1)     # argmax

        all_preds.append(predicted)
        all_labels.append(targets)

# Convert list of tensors → one tensor
all_preds  = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# Accuracy
accuracy = (all_preds == all_labels).float().mean().item() * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

