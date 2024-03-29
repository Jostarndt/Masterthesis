import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torch.utils.data import TensorDataset, ConcatDataset
import torch.nn.functional as F

import time
import timeit
import itertools
import pdb
import matplotlib.pyplot as plt


batchsize = 4096



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

class critic(nn.Module):
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






class Dataset():
    def __init__(self):
        self.support_points =20  #amount of euler steps / steps in the integral
        self.amount_x = 2 #amount of points of x on which V is getting trained. Note: trajectory has actual lengh of amount_x * support_points
        self.stepsize = 0.01 #stepsize for the euler steps, stepsize = distance of x from above / support points => distance of 0.3 
        #NOTE!! if you change this you have to change 'approx_costs' in train.py TWO TIMES!
        self.dataset = []
        self.datasets = []
        self.pde = dgl()
        self.kosten = cost_functional()

    def create_double(self, starting_point = None, control_value = None):
        '''creation of starting point'''
        trajectory, controls = self.pde.euler_step(stepsize = self.stepsize, total_steps = self.support_points, last_point = starting_point, control= control_value)
        return trajectory, controls

    def create_dataset(self, control_value = None, starting_point = None):
        x_output = torch.empty(self.amount_x, self.support_points +1 , starting_point.size()[0], starting_point.size()[1])
        u_output = torch.empty(self.amount_x, self.support_points +1 , starting_point.size()[0], 1)
 
        x, u = self.create_double(starting_point = starting_point, control_value = control_value)
        x_output[0] = x
        u_output[0] = u
        for i in range(self.amount_x-1):
            x, u = self.create_double(starting_point = x_output[i][-1], control_value = control_value)
            x_output[i+1] = x
            u_output[i+1] = u
        dataset = TensorDataset(x_output, u_output)
        return dataset

    def create_dataset_different_control_and_starts(self, amount_startpoints):
        controls = [[-i/10,-j/10] for i,j in zip(range(1, 5), range(1, 5))]
        #controls = [[-i/10,-j/10] for i,j in itertools.product(range(1, 5), range(1, 5))]
        #print(controls)
        controls = torch.unsqueeze(torch.tensor(controls, dtype= torch.float), 1)#TODO this in unncescessary - the controls & staring points are vectors anyways?
        #controls = [-i/2 for i in range(7)]
        #controls = torch.unsqueeze(torch.unsqueeze(torch.tensor(controls, dtype= torch.float), 1), 1)#TODO this in unncescessary - the controls & staring points are vectors anyways?
        starting_points =np.random.rand(amount_startpoints,2)
        starting_points =torch.unsqueeze(torch.tensor(starting_points, dtype= torch.float), 1)*0.1 #multiplication to adapt to starting point given in problem formulation

        #print(starting_points)



        datasets=[]
        for i in itertools.product(controls, starting_points):
            datasets.append(self.create_dataset(control_value=i[0], starting_point=i[1]))
        datasets= ConcatDataset(datasets)
        return datasets

class dgl:
    def __init__(self, x_0 = None):
        self.x_0 = torch.tensor([[1]], dtype= torch.float) if not x_0 else x_0

    def rhs(self, x, u):
        first_part =torch.unsqueeze(torch.unsqueeze( -x[0][0] + x[0][1], 0), 1)
        second_part = -0.5*(x[0][0] + x[0][1]) + 0.5* x[0][0]**2*x[0][1] + x[0][0]*u
        output = torch.stack((first_part, second_part))
        return torch.squeeze(output ,1)#torch.matmul(self.A,x) +torch.matmul(self.B,u)

    def euler_step(self, stepsize = 0.1,total_steps = 1, last_point=None, control = None):
        output_traj = torch.unsqueeze(last_point, 1)
        con_value = torch.matmul(control, last_point.transpose(0,1))
        output_control = torch.unsqueeze(con_value, 1)#maybe do not unsqueeze it?!
        for i in range(total_steps):
            last_point = last_point + stepsize * self.rhs(last_point, con_value).transpose(-1,0)
            output_traj = torch.cat((output_traj,torch.unsqueeze(last_point, 1)))
            con_value = torch.matmul(control, last_point.transpose(0,1))
            output_control = torch.cat((output_control,torch.unsqueeze(con_value, 1)))

        return output_traj, output_control

class cost_functional:
    def __init__(self):
        self.Q = torch.tensor([[1]],dtype=torch.float)
        self.R = torch.tensor([[1]], dtype = torch.float)

    def approx_costs(self, x_values, l_control_values, r_control_values, x_size):
        '''
        #returns the integral over xQx + uRU$
        '''

        '''
        #this is for checking if the square works also in batches:

        print('x_values: ', x_values)
        print('x_values times 1',torch.matmul(self.Q,x_values))
        print('quadrat: ', torch.matmul(x_values, torch.matmul(self.Q,x_values)))
        '''
        points = torch.matmul(x_values, torch.matmul(self.Q,x_values))+ torch.matmul(l_control_values, torch.matmul(self.R,r_control_values))
        #summands = torch.square(points)
        summands = torch.abs(points)
        integral = torch.mean(summands)
        return integral * x_size




class error():
    def __init__(self):
        self.costs = cost_functional()
        self.Q = torch.tensor([[1, 0], [0, 1]],dtype=torch.float)
        self.R = torch.tensor([[1]], dtype = torch.float)

    def value_iteration_left(self,trajectory, control, old_control, new_control, value_function, on_optimum):
        traj = torch.squeeze(trajectory, 0)
        old_controls = old_control(traj).detach().reshape_as(control)
        new_controls= new_control(traj).detach().reshape_as(control)
        diff = torch.squeeze(old_controls - control, 0)
        oc = torch.squeeze(old_controls, 0)
        points_a = torch.matmul(traj, torch.matmul(self.Q, traj.transpose(-1,-2))) + torch.matmul(oc, torch.matmul(self.R, oc.transpose(-1,-2)))
        points_b = torch.matmul(new_controls, torch.matmul(self.R, diff))
        points_together = 2*points_b - points_a
        control_loss =0.2* torch.mean(points_together, dim=-3)
        
        if on_optimum == True:
            overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[:,0]))) - 0.5*torch.square(trajectory[:,0,0,0])- torch.square(trajectory[:,0,0,1]) #TODO make sure this is correct
        elif on_optimum == False:
            overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[:,0]))) - torch.squeeze(torch.squeeze(value_function(trajectory[:,-1]).detach())) + torch.squeeze(control_loss)
            #overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[0][0]))) - torch.squeeze(torch.squeeze(value_function(trajectory[0][-1]))) + control_loss 

        overall_loss = torch.square(overall_loss)# + torch.square(value_function(torch.tensor([[0,0]], dtype = torch.float)))
        
        overall_loss = torch.mean(overall_loss)
        return overall_loss
    
    def value_iteration_right(self,trajectory, control, old_control, new_control, value_function, on_optimum):
        traj = torch.squeeze(trajectory, 0)

        old_controls = old_control(traj).detach()
        new_controls= new_control(traj).detach()
        diff = torch.squeeze(old_controls - control, 0)
        oc = torch.squeeze(old_controls, 0)

        points_a = torch.matmul(traj, torch.matmul(self.Q, traj.transpose(-1,-2))) + torch.matmul(oc, torch.matmul(self.R, oc.transpose(-1,-2)))
        points_b = torch.matmul(new_controls, torch.matmul(self.R, diff))
        points_together = 2*points_b - points_a

        control_loss =0.2* torch.mean(points_together, dim=-3)
        
        if on_optimum == True:
            overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[:,0]))) - 0.5*torch.square(trajectory[:,0,0,0])- torch.square(trajectory[:,0,0,1]) #TODO make sure this is correct
        elif on_optimum == False:
            overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[:,0]).detach())) - torch.squeeze(torch.squeeze(value_function(trajectory[:,-1]))) + torch.squeeze(control_loss)

        overall_loss = torch.square(overall_loss)# + torch.square(value_function(torch.tensor([[0,0]], dtype = torch.float)))
        
        #overall_loss = torch.mean(overall_loss)
        return overall_loss

    def policy_improvement(self,trajectory, control, old_control, new_control, value_function, op_factor = 0.5, noise_factor = 0):
        traj = torch.squeeze(trajectory, 0)
        
        old_controls = old_control(traj).detach().reshape_as(control)
        new_controls= new_control(traj).reshape_as(control)
        diff = torch.squeeze(old_controls - control, 0)
        oc= torch.squeeze(old_controls, 0)

        points_a = torch.matmul(traj, torch.matmul(self.Q, traj.transpose(-1,-2))) + torch.matmul(oc, torch.matmul(self.R, oc.transpose(-1,-2)))
        points_b = torch.matmul(new_controls, torch.matmul(self.R, diff.transpose(-1,-2)))
        points_together = 2*points_b - points_a
        control_loss =0.2* torch.mean(points_together, dim=-3)
        overall_loss = (1-op_factor)*(value_function(trajectory[:,0]).detach() - value_function(trajectory[:,-1]).detach()).reshape_as(control_loss) + (op_factor)*(0.5* trajectory[:,0,0,0]**2 + trajectory[:,0,0,1]**2 - 0.5* trajectory[:,-1,0,0]**2 - trajectory[:,-1,0,1]**2).reshape_as(control_loss)  + control_loss
        
        
        overall_loss = torch.square(overall_loss)#TODO: square instead?

        overall_loss = torch.mean(overall_loss)
        return overall_loss

    def policy_warmup(self,trajectory, control, old_control, new_control, value_function, op_factor = 0.5):
        traj = torch.squeeze(trajectory, 0)
        
        new_control.zero_grad()
        overall_loss = 0.2*torch.mean(torch.abs(new_control(traj) + torch.unsqueeze(traj[:,:,0]*traj[:,:,1],1)))
        return overall_loss

def optimal_value_function(traj):
    return (0.5* traj[0,0]**2 + traj[0,1]**2)

def present_results(value_function, new_control, stepname= "after_training"):
    int_const = value_function(torch.tensor([[0,0]], dtype= torch.float)).detach()
    print(int_const, ' - this is the integration constant')
    op_val_img = np.zeros((3,100,100))
    op_con_img = np.zeros((3,100,100))
    con_img = np.zeros((3,100,100))
    val_img = np.zeros((3,100,100))
    for x_1 in range(100):
        for x_2 in range(100):
            #print("tuple: ", x_1/1000, x_2/1000)
            op_con_img[0][x_1][x_2] = - x_1/1000 * x_2/1000 
            op_val_img[0][x_1][x_2] = 0.5*(x_1/1000) **2 + (x_2/1000 )**2
            val_img[0][x_1][x_2] = value_function(torch.tensor([[x_1/1000 , x_2/1000 ]], dtype = torch.float)).detach() - int_const
            #print(x_1/1000, x_2/1000)
            con_img[0][x_1][x_2] = new_control(torch.tensor([[x_1/1000 , x_2/1000]], dtype = torch.float)).detach()
    
    #rescaling the images:
    val_max = max(np.amax(val_img), np.amax(op_val_img))
    val_min = min(np.amin(val_img), np.amin(op_val_img))
    con_max = max(np.amax(con_img), np.amax(op_con_img))
    con_min = min(np.amin(con_img), np.amin(op_con_img))
    
    print(con_img[0], 'con img')
    print(op_con_img[0], 'op con img')
    print('-----------------------')

    val_img = (val_img-val_min)/(val_max - val_min)
    op_val_img = (op_val_img-val_min)/(val_max-val_min)

    con_img = (con_img-con_min)/(con_max - con_min)
    op_con_img = (op_con_img-con_min)/(con_max-con_min)
    
    Writer.add_image('optimal value/'+stepname, op_val_img , 0)
    Writer.add_image('optimal control function/'+stepname, op_con_img , 0)
    Writer.add_image('value_function/'+stepname, val_img , 0)
    Writer.add_image('control_function/'+stepname, con_img , 0)
    Writer.close()
    print(con_img[0], 'con img')
    print(op_con_img[0], 'op con img')
    
    print('-----------------------')
    print(val_img[0], 'val img')
    print(op_val_img[0], 'op val img')


if __name__ == '__main__':
    start = time.time()
    error = error()
    Writer = SummaryWriter()
    for experiment in range(10):
        dataset = Dataset()
        trainset = dataset.create_dataset_different_control_and_starts(amount_startpoints=100)
        train_loader = DataLoader(dataset = trainset, batch_size = batchsize, shuffle =True)

        
        print("##################################")
        old_control = actor(stabilizing = False, control_dim = 1, space_dim = 2)
        new_control = actor(stabilizing = False, control_dim = 1, space_dim = 2)
        value_function = critic(positive = True, space_dim = 2)
        costs = cost_functional()

        control_optimizer = optim.LBFGS(new_control.parameters(), lr=0.005,history_size=2400, max_iter = 30, tolerance_change=1e-15)
        value_optimizer = optim.LBFGS(value_function.parameters(), lr=0.001,history_size = 2400, max_iter = 200, tolerance_change=1e-15)
        
        Q = torch.tensor([[1, 0], [0, 1]],dtype=torch.float)
        R = torch.tensor([[1]], dtype = torch.float)


        #Training and Testing
        plt_loss = 0
        for epoch in range(1, 20, 1):
            print("epoch: ", epoch)
            old_control.train()
            value_function.train()
            new_control.train()
            for j,(x, u) in enumerate(train_loader):
                #-------------Value iteration------------
                traj = torch.squeeze(x, 0)
                old_controls = old_control(traj).detach().reshape_as(u)
                new_controls= new_control(traj).detach().reshape_as(u)
                diff = torch.squeeze(old_controls - u, 0)
                oc = torch.squeeze(u, 0)

                points_a = torch.matmul(traj, torch.matmul(Q, traj.transpose(-1,-2))) + torch.matmul(oc, torch.matmul(R, oc.transpose(-1,-2)))
                points_b = torch.matmul(new_controls, torch.matmul(R, diff))
                points_together = 2*points_b - points_a
                control_loss =0.2* torch.mean(points_together, dim=-3)

                def closure():
                    control_optimizer.zero_grad()
                    value_optimizer.zero_grad()
                    overall_loss =  torch.squeeze(torch.squeeze(value_function(x[:,0]))) - torch.squeeze(torch.squeeze(value_function(x[:,-1]).detach())) + torch.squeeze(control_loss)

                    overall_loss = torch.square(overall_loss).mean()# + torch.square(value_function(torch.tensor([[0,0]], dtype = torch.float)))
                    overall_loss.backward()
                    return overall_loss
                value_optimizer.step(closure)
                overall_loss = torch.square(torch.squeeze(torch.squeeze(value_function(x[:,0]))) - torch.squeeze(torch.squeeze(value_function(x[:,-1]).detach())) + torch.squeeze(control_loss)).mean()
                print('l2 res after value update', overall_loss )
                plt_loss = np.vstack((plt_loss,overall_loss.detach()))

                #--------policy improvement------------
                overall_loss_init =(value_function(x[:,0]).detach() - value_function(x[:,-1]).detach())
                def closure():
                    control_optimizer.zero_grad()
                    value_optimizer.zero_grad()
            
                    new_controls= new_control(traj).reshape_as(u)

                    points_b = torch.matmul(new_controls, torch.matmul(R, diff.transpose(-1,-2)))#this matmul can be outsourced
                    points_together = 2*points_b - points_a
                    control_loss =0.2* torch.mean(points_together, dim=-3)
                    overall_loss =overall_loss_init.reshape_as(control_loss) + control_loss
                    overall_loss = torch.square(overall_loss).mean()
                    overall_loss.backward()
                    return overall_loss
                control_optimizer.step(closure)

                new_controls= new_control(traj).reshape_as(u)
                points_b = torch.matmul(new_controls, torch.matmul(R, diff.transpose(-1,-2)))#this matmul can be outsourced
                points_together = 2*points_b - points_a
                control_loss =0.2* torch.mean(points_together, dim=-3)
                overall_loss =overall_loss_init.reshape_as(control_loss) + control_loss
                overall_loss = torch.square(overall_loss).mean()
                print('l2 res after control update', overall_loss)
                plt_loss = np.vstack((plt_loss,overall_loss.detach()))
        plt.plot(plt_loss, label = 'residual'+str(experiment))
        np.savetxt(str(experiment), plt_loss)
    
    end = time.time()
    print('10 runs took ', end - start, ' seconds')
    plt.show()
    print('done')
