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
    def __init__(self,control_dim=1, space_dim=1):
        super(actor, self).__init__()
        self.c_dim = control_dim
        #constant output
        self.fc1 = nn.Linear(space_dim, 10)
        self.fc2 = nn.Linear(10, control_dim)

        self.noise = torch.rand(self.c_dim)-0.5*torch.ones(self.c_dim)#is this +-1? or +-0?

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        return self.fc2(x)
    def eps_greedy(self,x, epsilon = 0.5) :
        error =2* self.epsilon* self.noise
        step = self.forward(x) + error
        return torch.rand(self.c_dim)


class critic(nn.Module):
    def __init__(self, space_dim=1):
        super(critic, self).__init__()
        self.s_dim = space_dim
        self.fc1 = nn.Linear(space_dim, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10,int(1))
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))
    def jacobi_matrix(self):
        pass
    

if __name__ == '__main__':
    critic = critic(space_dim)
    #act = actor(control_dim, space_dim)
    #linear = dgl.dgl(space_dim, control_dim)

