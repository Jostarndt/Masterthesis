import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

import model

x =torch.eye(1)
print(x)
x.requires_grad_(True)


def value_truth(x):
    return 0.5* x**2

def control_truth(x):
    return -0.2*x

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
value_function = model.critic()
control_function = model.actor()

x_value = [torch.tensor([[i/1000]]) for i in range(1000)]
y_value = [value_truth(i) for i in x_value]

x_control = [torch.tensor([[i/1000]]) for i in range(1000)]
y_control = [control_truth(i) for i in x_control]

optimizer_value = optim.SGD(value_function.parameters(), lr= 0.01)
optimizer_control = optim.SGD(control_function.parameters(), lr= 0.01)


indices = np.arange(len(x_value))
np.random.shuffle(indices)

for epoch in range(15):
    print("epoch")
    for i in indices:
        optimizer_value.zero_grad()
        loss =(y_value[i] - value_function(x_value[i]) ) **2
        loss.backward()
        optimizer_value.step()

for epoch in range(15):
    print("epoch")
    for i in indices:
        optimizer_control.zero_grad()
        loss =(y_control[i] - control_function(x_control[i]) ) **2
        loss.backward()
        optimizer_control.step()

for i in range(100):
    Writer.add_scalar('control_function', control_function(torch.tensor([[i/100]], dtype = torch.float)), i)
    Writer.add_scalar('value_function', value_function(torch.tensor([[i/100]], dtype = torch.float)), i)


''' saving the model'''


torch.save(value_function.state_dict(), './models/pretrained_value')
torch.save(control_function.state_dict(), './models/pretrained_control')
