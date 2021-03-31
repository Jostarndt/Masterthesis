import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

x =torch.eye(1)
print(x)
x.requires_grad_(True)


def approx_function(x):
    return x**2

class nn_model(nn.Module):
    def __init__(self):
        super(nn_model, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc3 = nn.Linear(50,1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x
    
Writer = SummaryWriter()
test_function = nn_model()

x = [torch.tensor([[i/1000]]) for i in range(1000)]
y = [approx_function(i) for i in x]

optimizer = optim.SGD(test_function.parameters(), lr= 0.01)

for i in range(100):
    Writer.add_scalar('test/pretraining', test_function(torch.tensor([[i/100]], dtype = torch.float)), i)

indices = np.arange(len(x))#x.shape[0])
np.random.shuffle(indices)

for epoch in range(30):
    print("epoch")
    for i in indices:
        optimizer.zero_grad()
        loss =(test_function(x[i])**2 + test_function(x[i])-1 ) **2
        loss.backward()
        optimizer.step()

for i in range(100):
    Writer.add_scalar('test/after training', test_function(torch.tensor([[i/100]], dtype = torch.float)), i)
 
