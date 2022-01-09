import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torch.utils.data import TensorDataset, ConcatDataset
import torch.nn.functional as F

#import timeit
import time
import itertools
import pdb
import matplotlib.pyplot as plt


class actor(nn.Module):
    def __init__(self,control_dim=1, space_dim=1, stabilizing = False):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(space_dim, 4)
        self.fc3 = nn.Linear(4, control_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc3(x)

class critic(nn.Module):
    def __init__(self, space_dim=1, positive = False):
        super(critic, self).__init__()
        self.s_dim = space_dim
        self.fc1 = nn.Linear(space_dim, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 5)
        self.fc4 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.abs(x)



def optimal_value_function(traj):
    return (0.5* traj[0,0]**2 + traj[0,1]**2)


if __name__ == '__main__':
    new_control = actor(stabilizing = False, control_dim = 1, space_dim = 2)
    value_function = critic(positive = True, space_dim = 2)
    

    control_optimizer_p = optim.SGD(new_control.parameters(), lr=0.3) #50
    value_optimizer_p = optim.SGD(value_function.parameters(), lr=0.5)#100

    #pretraining to get minimal possible convergence result
    x_pretrain = 0.1* torch.tensor(np.random.rand(10000, 2), dtype=torch.float) 
    v_pretrain = 0.5*x_pretrain[:,0]*x_pretrain[:,0] + x_pretrain[:,1]*x_pretrain[:,0]
    u_pretrain =- x_pretrain[:,0]*x_pretrain[:,1]#TODO this is wrong!
    
    loss_plot = 0
    for epoch in range(1,100,1):
        #value function
        if True:
            print('pretrain epoch: ', epoch)
            value_optimizer_p.zero_grad()
            value_function.train()

            loss = torch.square(value_function(x_pretrain) - v_pretrain).mean()
            loss.backward()
            value_optimizer_p.step()
            print('value_loss: ', loss)
            #print('wrong_value_loss: ',torch.square((value_function(x_pretrain)-value_function(zeros).detach()) - v_pretrain).mean())

        #control-function
        if False:
            control_optimizer_p.zero_grad()
            new_control.train()
            loss = torch.square(new_control(x_pretrain) - u_pretrain).mean()

            loss.backward()
            control_optimizer_p.step()
            print('control loss: ',loss)

        loss_plot = np.vstack((loss_plot, loss.detach()))        
    plt.plot(loss_plot[1:])
    plt.show()



    print('done')
