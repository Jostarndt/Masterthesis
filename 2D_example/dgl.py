import numpy as np
import torch
import model
import itertools
import timeit
import pdb

from torch.utils.data import TensorDataset, ConcatDataset


class Dataset():
    def __init__(self):
        self.support_points =5  #amount of euler steps / steps in the integral
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




if __name__ == '__main__':
    print('start')
    dataset = Dataset()
    dataset.create_dataset_different_control_and_starts()
