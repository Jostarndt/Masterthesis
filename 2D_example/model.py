import numpy as np
import torch
#import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import dgl


space_dim = 1
control_dim = 1
batch_size = 2


class actor(nn.Module):
    def __init__(self,control_dim=1, space_dim=1, stabilizing = False):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(space_dim, 50*space_dim)
        self.fc2 = nn.Linear(50*space_dim, 50*space_dim)
        self.fc3 = nn.Linear(50*space_dim, control_dim)
        if stabilizing:
            self.fc1.weight = torch.nn.parameter.Parameter(torch.zeros(self.fc1.weight.shape))
            self.fc1.bias = torch.nn.parameter.Parameter(torch.zeros(self.fc1.bias.shape))
            
            self.fc2.weight = torch.nn.parameter.Parameter(torch.zeros(self.fc2.weight.shape))
            self.fc2.bias = torch.nn.parameter.Parameter(torch.zeros(self.fc2.bias.shape))
            
            self.fc3.weight = torch.nn.parameter.Parameter(torch.zeros(self.fc3.weight.shape))
            self.fc3.bias = torch.nn.parameter.Parameter(torch.zeros(self.fc3.bias.shape))
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class critic_nn(nn.Module):
    def __init__(self, space_dim=1, positive = False):
        super(critic, self).__init__()
        self.s_dim = space_dim
        self.fc1 = nn.Linear(space_dim, 30)
        self.fc2 = nn.Linear(30, 1)
        
        if positive:
            pass
            self.fc1.weight = torch.nn.parameter.Parameter(torch.abs(self.fc1.weight))
            self.fc2.weight =  torch.nn.parameter.Parameter(torch.abs(self.fc2.weight))
            self.fc1.bias =  torch.nn.parameter.Parameter(torch.abs(self.fc1.bias))
            self.fc2.bias =  torch.nn.parameter.Parameter(torch.abs(self.fc2.bias))

        '''
        if positive:
            pass
            self.fc1.weight = torch.nn.parameter.Parameter(torch.zeros(self.fc1.weight.shape))
            self.fc2.weight =  torch.nn.parameter.Parameter(torch.zeros(self.fc2.weight.shape))
            self.fc1.bias =  torch.nn.parameter.Parameter(torch.zeros(self.fc1.bias.shape))
            self.fc2.bias =  torch.nn.parameter.Parameter(torch.zeros(self.fc2.bias.shape))
        '''

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.abs(x)

class critic(nn.Module):
    def __init__(self, space_dim=1, positive = False):
        super(critic, self).__init__()
        self.bias_weights = torch.nn.parameter.Parameter(torch.randn(1))
        self.linear_weights= torch.nn.parameter.Parameter(torch.randn(space_dim))
        self.square_weights = torch.nn.parameter.Parameter(torch.randn(space_dim))
        #self.square_weights = torch.nn.parameter.Parameter(torch.tensor([0.5, 1]))
        
    def forward(self, x):
        x_square = torch.square(x)
        output = torch.matmul(x_square, self.square_weights)#+ torch.matmul(x, self.linear_weights) 
        return output


if __name__ == '__main__':
    critic = critic(space_dim)
    #act = actor(control_dim, space_dim)
    #linear = dgl.dgl(space_dim, control_dim)

