import numpy as np
import dgl
import model
import torch
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pdb

#https://discuss.pytorch.org/t/what-will-happen-with-multiple-gradients/20485

W = nn.Parameter(torch.ones(1)*3) 
print("W is ", W)
x  = torch.ones(1) * 2
print("x is ", x)

for _ in range(3):
    print('yo')
    output = (x**2) * W
    print(x, 'this is x')
    print(output, 'this is the output')
output.backward()
print(W.grad, 'this is the gradient')


x_two = torch.ones(1) * 7
