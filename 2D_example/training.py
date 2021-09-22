import numpy as np
import dgl
import torch
import torch.optim as optim
import torch.nn as nn


from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader
import itertools
import pdb
from torch.utils.data import TensorDataset, ConcatDataset



batchsize = 4096
class polynomial_linear_actor(nn.Module):
    def __init__(self):
        super(polynomial_linear_actor, self).__init__()
    def forward(self, x):
        square = torch.square(x) #(x,y) -> (x^2, y^2)
        prod = torch.prod(x, 3).unsqueeze(3) #(x, y) -> x*y
        output = torch.cat((x, square, prod), 3)
        return output

class polynomial_linear_critic(nn.Module):
    def __init__(self):
        print('correct function')
        super(polynomial_linear_critic, self).__init__()
    def forward(self, x):
        square = torch.square(x)
        prod = torch.prod(x,2).unsqueeze(2)
        const = torch.ones_like(prod)

        #output = torch.cat((square, prod, const), 2)#adding a const leads to missing invertability
        output = torch.cat((square, prod), 2)
        return output


class Dataset():
    def __init__(self):
        self.support_points =20  #amount of euler steps / steps in the integral
        self.amount_x = 2 #amount of points of x on which V is getting trained. Note: trajectory has actual lengh of amount_x * support_points
        self.stepsize = 0.01 #stepsize for the euler steps, stepsize = distance of x from above / support points => distance of 0.3 
        #NOTE!! if you change this you have to change 'approx_costs' in train.py TWO TIMES!
        self.dataset = []
        self.datasets = []
        self.pde = dgl()

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
        print(controls)
        controls = torch.unsqueeze(torch.tensor(controls, dtype= torch.float), 1)#TODO this in unncescessary - the controls & staring points are vectors anyways?
        #controls = [-i/2 for i in range(7)]
        #controls = torch.unsqueeze(torch.unsqueeze(torch.tensor(controls, dtype= torch.float), 1), 1)#TODO this in unncescessary - the controls & staring points are vectors anyways?
        starting_points =np.random.rand(amount_startpoints,2)
        starting_points =torch.unsqueeze(torch.tensor(starting_points, dtype= torch.float), 1)*0.1 #multiplication to adapt to starting point given in problem formulation

        print(starting_points)



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






class error():
    def __init__(self):
        self.Q = torch.tensor([[1, 0], [0, 1]],dtype=torch.float)
        self.R = torch.tensor([[1]], dtype = torch.float)

    def both_iterations_direct_solution(self,trajectory, control, old_control, value_function, theta_u, theta_v):
        #new_control: model that delivers a vector of monomials
        #old_control: exactly the same as new_control -> not needed at all! TODO remove
        #value_function: also list of monomials - in the paper ist theta_v
        #TODO assert dimensionality of theta and monomial basis
        control_monomials = old_control(trajectory)#TODO should have same shape as control! -> this is going to be difficult?
        rho_q = torch.matmul(trajectory, torch.matmul(self.Q, trajectory.transpose(-1,-2))).mean((1,2,3)) #MEAN

        #print(control.size())
        #print(torch.matmul(value_function(trajectory[:,0]), theta_v))
        rho_delta_phi = value_function(trajectory[:,0]) - value_function(trajectory[:,-1])#(torch.matmul(value_function(trajectory[:,0]) - value_function(trajectory[:,-1]), theta_v)) 
        
        control_approx = torch.matmul(control_monomials, theta_u)
        rho_psi = torch.mul(torch.matmul(control_approx, self.R), control_approx).mean(1)
        
        rho_u_psi= 2* torch.matmul(control_approx.unsqueeze(3) - control , control_monomials).mean(1)#torch.matmul(control*control_monomials, theta_u)
        
        pi = rho_q + rho_psi.squeeze()

        #print(rho_delta_phi * theta_v+ rho_u_psi * theta_u) # should be same as cat(rho, rho) * cat(theta, theta)

        #torch.matmul(torch.cat((rho_delta_phi, rho_u_psi), 2), torch.cat((theta_v, theta_u)))
        z = torch.cat((rho_delta_phi, rho_u_psi), 2).squeeze()
        
        #the equation system is now:     z*torch.cat((theta_v, theta_u)) = pi
        

        grad_step = torch.matmul( torch.matmul(z.transpose(0,1),  z), torch.cat((theta_v, theta_u))) - torch.matmul(z.transpose(0,1), pi) #TODO: sure that pi is correct?



        
      
        theta = torch.cat((theta_v, theta_u))  -  2 * 0.05 * grad_step #TODO check if correct.
        theta_v, theta_u = torch.split(theta, [3,4], dim = 0)#TODO implement torch.size()[] instead of hard coding
        
        residual = torch.abs(torch.matmul(z, theta) - pi).sum()
        return residual, theta_v, theta_u
    

if __name__ == '__main__':
    error = error()
    dataset = Dataset()
    trainset = dataset.create_dataset_different_control_and_starts(amount_startpoints=500)
    train_loader = DataLoader(dataset = trainset, batch_size = batchsize, shuffle =True)

    
    testset = dataset.create_dataset_different_control_and_starts(amount_startpoints=10)
    test_loader = DataLoader(dataset = testset, batch_size = 2, shuffle =True)#2 to bring this into batch format


    print("##################################")
    

    control_function = polynomial_linear_actor()
    value_function = polynomial_linear_critic()
    theta_u = torch.ones(5)

    theta_v = torch.ones(3)
    
    #theta_u =torch.tensor([0, 0,0,0, -1], dtype = torch.float)
    #theta_v =torch.tensor([0.5, 1, 0], dtype = torch.float)



    #present results!

    #Training and Testing
    for epoch in range(40):
        print("epoch: ", epoch)
        print('++++++++++++++++++++Train++++++++++++')
        for j,(x, u) in enumerate(train_loader):

            residual, theta_v, theta_u = error.both_iterations_direct_solution(trajectory= x, control=u, old_control = control_function, value_function = value_function , theta_u= theta_u, theta_v= theta_v)
            print(residual, theta_v, theta_u)
