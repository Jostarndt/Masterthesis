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

W = nn.Parameter(torch.ones(1))
print("W is ", W)
x  = torch.ones(1) * 2
print("x is ", x)

for _ in range(3):
    print('yo')
    x = x * W
    print(x)
    x.backward(retain_graph = True)
    print(W.grad)
