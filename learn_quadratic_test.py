import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

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
test_function_two = nn_model()

x = [torch.tensor([[i/1000]]) for i in range(1000)]
y = [approx_function(i) for i in x]
y_two = [approx_function_two(i) for i in x]

optimizer = optim.SGD(test_function.parameters(), lr= 0.01)
optimizer_two = optim.SGD(test_function_two.parameters(), lr= 0.01)

for i in range(100):
    Writer.add_scalar('test/pretraining', test_function(torch.tensor([[i/100]], dtype = torch.float)), i)

indices = np.arange(len(x))#x.shape[0])
np.random.shuffle(indices)

for epoch in range(20):
    print("epoch")
    for i in indices:
        optimizer.zero_grad()
        #optimizer_two.zero_grad()
        loss =(y[i] - test_function(x[i]) ) **2
        #loss_two = (y[i]-y_two[i]- (test_function_two(x[i]) - test_function_two(x[i]+1)))**2
        loss.backward()
        #loss_two.backward()
        optimizer.step()
        #optimizer_two.step()
        #Writer.add_scalar('loss_two'+str(epoch), loss_two, i)
    #print(loss_two)

print(y[-1], x[-1], test_function(x[-1]), y_two[-1], test_function(x[-1]+1))
print((y[-1]+y_two[-1]- (test_function(x[-1]) + test_function(x[-1]+1)))**2)

test_function_two = copy.deepcopy(test_function)
'''
for epoch in range(20):
    print("epoch")
    for i in indices:
        #optimizer.zero_grad()
        optimizer_two.zero_grad()
        #loss =(y[i] - test_function(x[i]) ) **2
        loss_two = (y[i]-y_two[i]- (test_function_two(x[i]) - test_function_two(x[i]+1)))**2
        #loss.backward()
        loss_two.backward()
        #optimizer.step()
        optimizer_two.step()
        Writer.add_scalar('loss_two'+str(epoch), loss_two, i)
    print(loss_two)
'''

for i in range(100):
    Writer.add_scalar('test/test_function', test_function(torch.tensor([[i/100]], dtype = torch.float)), i)
    #Writer.add_scalar('test/test_function_two', test_function_two(torch.tensor([[i/100]], dtype = torch.float)), i)
 
