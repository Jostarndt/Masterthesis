import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import dgl
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader

x =torch.eye(1)
print(x)
x.requires_grad_(True)


def approx_function(x):
    return x**2

def approx_function_two(x):
    return (x+1)**2

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

#x = [torch.tensor([[np.random.rand(1)]]) for i in range(10)]
#y = [approx_function(i) for i in x]

optimizer = optim.SGD(test_function.parameters(), lr=0.1)
costs = dgl.cost_functional()
loss_fn = nn.MSELoss(reduction = 'mean')

for i in range(100):
    Writer.add_scalar('test/pretraining', test_function(torch.tensor([[i/100]], dtype = torch.float)), i)

indices = np.arange(len(x))
np.random.shuffle(indices)

'''
for epoch in range(5):
    print("epoch")
    x = [torch.tensor([[np.random.rand(1)]], dtype= torch.float, requires_grad = True) for i in range(1000)]
    x_square = [i**2 for i in x]
    u = [torch.tensor([[test_function(i)]], dtype = torch.float) for i in x]
    for i in range(len(x)):
        optimizer.zero_grad()
        loss = (x_square[i]-test_function(x[i]))**2
        loss.backward()
        optimizer.step()
        Writer.add_scalar('loss: ', loss, epoch)

for epoch in range(100):
    print("epoch")
    x = [np.random.rand(1) for i in range(10000)]
    x_square= [i**2 for i in x]
    x = torch.tensor(x, dtype=torch.float)
    x_square = torch.tensor(x_square, dtype=torch.float)
    
    test_function.train()

    optimizer.zero_grad()
    y_hat = test_function(x)
    loss = loss_fn(y_hat,  x_square)
    print(loss)
    loss.backward()
    optimizer.step()
'''
x = [np.random.rand(1) for i in range(10000)]
x_square= [i**2 for i in x]
x = torch.tensor(x, dtype=torch.float)
x_square = torch.tensor(x_square, dtype=torch.float)

train_data = TensorDataset(x, x_square)
print(train_data[0])
train_loader = DataLoader(dataset= train_data, batch_size = 32, shuffle = False)

for epoch in range(10):
    print("epoch")
    for x_batch, y_batch in train_loader:
        test_function.train()

        y_hat = test_function(x_batch)

        loss = loss_fn(y_batch, y_hat)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss.item())


for i in range(100):
    Writer.add_scalar('test/test_function', test_function(torch.tensor([[i/100]], dtype = torch.float)), i)
 
