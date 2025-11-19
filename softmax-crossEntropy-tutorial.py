import torch

import torch.nn as nn

logits=torch.tensor([[2.7,0.2,0.3],[0.7,1.8,0.3],[0.2,0.5,2.3]])

target=torch.tensor([0,1,2])


softmax=torch.softmax(logits,dim=1)  ## Softmax converts raw logits into probabilities such that:

# Every value is between 0 and 1

# All probabilities in each row sum to 1

# That’s the exact purpose of Softmax.

print(softmax)

## now cross entropy is used to find the loss here

criterion = nn.CrossEntropyLoss()

loss=criterion(logits,target)

print(loss.item())


##example 2 

logits2=torch.rand(5,10)
target2=torch.LongTensor([1,7,2,4,3])

softmax2=torch.softmax(logits2,dim=1)
print(logits2)
print(target2)
print(softmax2)

criterion=nn.CrossEntropyLoss()

loss2=criterion(logits2,target2)

print(loss2.item())  ## so as we can see that loss is more then 2 so there is super problem in our code
