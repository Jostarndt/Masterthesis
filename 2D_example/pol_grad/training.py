import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import itertools
import pdb
import time
from torch.utils.data import TensorDataset, ConcatDataset
import matplotlib.pyplot as plt


batchsize = 4096 #512#4096


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
        super(polynomial_linear_critic, self).__init__()
    def forward(self, x):
        square = torch.square(x)
        prod = torch.prod(x,2).unsqueeze(2)
        output = torch.cat((square, prod), 2)#output is: (x,y) -> (x^2, y^2, xy)
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
        controls = torch.unsqueeze(torch.tensor(controls, dtype= torch.float), 1)
        starting_points =np.random.rand(amount_startpoints,2)
        starting_points =torch.unsqueeze(torch.tensor(starting_points, dtype= torch.float), 1)*0.1 #multiplication to adapt to starting point given in problem formulation


        print(controls)

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

    def both_iterations_direct_solution(self,trajectory, control, old_control, value_function, theta_u, theta_v, epoch):
        epoch = epoch / 10000
        control_monomials = old_control(trajectory)
        rho_q = torch.mul(torch.matmul(trajectory, torch.matmul(self.Q, trajectory.transpose(-1,-2))).mean((1,2,3)), 0.2) #MEAN

        rho_delta_phi = value_function(trajectory[:,0]) - value_function(trajectory[:,-1])
        
        control_approx = torch.matmul(control_monomials, theta_u)
        rho_psi =torch.mul(torch.mul(torch.matmul(control_approx, self.R), control_approx).mean(1), 0.2)#should be the same as torch.square(control_approx).mean()
        
        rho_u_psi= torch.mul(torch.mul( torch.matmul(control_approx.unsqueeze(3) - control , control_monomials), 2).mean(1),0.2)
        
        pi = rho_q + rho_psi.squeeze()

        z = torch.cat((rho_delta_phi, rho_u_psi), 2).squeeze()
        
        
        '''repeat this :'''
        stepsize_v = 80
        stepsize_u = 10
        if epoch >= 10:
            stepsize_u = 160
            if epoch >= 30:
                stepsize_u = 300
                if epoch >= 50:
                    stepsize_u = 400
                    #if epoch >= 70:
                        #stepsize_u = 450
        for i in range(1):#10000
            grad_step = torch.matmul( torch.matmul(z.transpose(0,1),  z), torch.cat((theta_v, theta_u))) - torch.matmul(z.transpose(0,1), pi) #TODO: sure that pi is correct?

            grad_step_v, grad_step_u = torch.split(grad_step, [3,5], dim = 0)

            theta_v = theta_v - 2 * stepsize_v * grad_step_v
            theta_u = theta_u - 2 * stepsize_u * grad_step_u
            


            theta = torch.cat((theta_v, theta_u))
            #pdb.set_trace()
            residual = torch.abs(torch.matmul(z, theta) - pi).sum()
            #print(residual)
            
            #pdb.set_trace()
            #if torch.abs(grad_step).sum() < 0.0000001:
            if torch.max(torch.abs(grad_step)) <  10**(-6-0.1*epoch):
                pass
                #print(torch.abs(grad_step).sum(), 'grad step small er than ->>>',10**(-6-0.2*epoch) )
                #break
        #print('grad step has size: ::::: ', torch.max(torch.abs(grad_step)))

        theta = torch.cat((theta_v, theta_u))

        '''
        theta = torch.cat((theta_v, theta_u))  -  2 * 100 * grad_step #TODO check if correct.

        theta_v, theta_u = torch.split(theta, [3,5], dim = 0)#TODO implement torch.size()[] instead of hard coding
        '''
        residual = torch.square(torch.matmul(z, theta) - pi).sum()
        return residual, theta_v, theta_u
    

if __name__ == '__main__':
    error = error()
    dataset = Dataset()
    trainset = dataset.create_dataset_different_control_and_starts(amount_startpoints=300)
    train_loader = DataLoader(dataset = trainset, batch_size = batchsize, shuffle =False)

    
    testset = dataset.create_dataset_different_control_and_starts(amount_startpoints=10)
    test_loader = DataLoader(dataset = testset, batch_size = 2, shuffle =True)#2 to bring this into batch format


    print("##################################")
    
    control_function = polynomial_linear_actor()
    value_function = polynomial_linear_critic()
    theta_u = torch.ones(5)
    theta_v = torch.ones(3)
    
    #theta_u =torch.tensor([0, 0,0,0, -1.1], dtype = torch.float)
    #theta_v =torch.tensor([0.5, 1, 0], dtype = torch.float)

    theta_u_plot = theta_u
    theta_v_plot = theta_v
    #Training and Testing
    start = time.time()
    for epoch in range(1,1000000, 1):
        for j,(x, u) in enumerate(train_loader):
            #theta_u =torch.tensor([0, 0,0,0, -1], dtype = torch.float)
            #theta_v =torch.tensor([2.5, 5, 0], dtype = torch.float)
            residual, theta_v, theta_u = error.both_iterations_direct_solution(trajectory= x, control=u, old_control = control_function, value_function = value_function , theta_u= theta_u, theta_v= theta_v, epoch = epoch)


        if epoch %10000 == 0:
            theta_u_plot = np.vstack((theta_u_plot, theta_u))
            theta_v_plot = np.vstack((theta_v_plot, theta_v))
            print("epoch: ", epoch)
            print('residual, theta_v, theta_u')
            print(residual, theta_v, theta_u)
            print('differences: v, u: , ', torch.abs(theta_v - torch.tensor([0.5, 1 ,0], dtype = torch.float)).mean(), torch.abs(theta_u - torch.tensor([0, 0 ,0,0,-1], dtype = torch.float)).mean())
            end = time.time()
            print('elapsed total time: ', end-start)




    plt.plot(theta_u_plot)
    plt.show()
    plt.plot(theta_v_plot)
    plt.show()
