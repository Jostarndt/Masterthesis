import numpy as np
import torch
#import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl


space_dim = 1
control_dim = 1
batch_size = 2


class actor(nn.Module):
    def __init__(self,control_dim=1, space_dim=1, stabilizing = False):
        super(actor, self).__init__()
        self.c_dim = control_dim
        #constant output
        if not stabilizing:
            self.fc1 = nn.Linear(space_dim, control_dim)
            #self.fc2 = nn.Linear(10, control_dim)
        elif stabilizing:
            self.fc1 = nn.Linear(space_dim,control_dim)
            #self.fc2 = nn.Linear(10, control_dim)
            self.fc1.weight = torch.nn.parameter.Parameter(torch.zeros(self.fc1.weight.shape))
            self.fc1.bias = torch.nn.parameter.Parameter(torch.zeros(self.fc1.bias.shape))
        else:
            raise InitializError('neither stable nor instable?')


    def forward(self, x):
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        return self.fc1(x)
    def eps_greedy(self,x, epsilon = 0.5) :
        error =2* self.epsilon* self.noise
        step = self.forward(x) + error
        return torch.rand(self.c_dim)


class critic(nn.Module):
    def __init__(self, space_dim=1, positive = False):
        super(critic, self).__init__()
        self.s_dim = space_dim
        self.fc1 = nn.Linear(space_dim, 40)#6
        self.fc2 = nn.Linear(40, 1)
        
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
        return x
    

if __name__ == '__main__':
    critic = critic(space_dim)
    #act = actor(control_dim, space_dim)
    #linear = dgl.dgl(space_dim, control_dim)

