import numpy as np
import torch
import model
import itertools


class Dataset():
    def __init__(self):
        self.support_points = 10
        self.amount_x = 10
        self.stepsize = 0.01
        #self.amount_controls = 11
        self.dataset = []
        self.datasets = []
        self.pde = dgl()
        self.kosten = cost_functional()

    def create_triple(self, starting_point = None, control_value = None):
        if control_value == None:
            print("no control value")
            control_value = torch.tensor([[0]], dtype= torch.float)
        elif isinstance(control_value, int) or isinstance(control_value, float):
            print("creation of tensor controls")
            control_value = torch.tensor([[control_value]], dtype= torch.float)
        else:
            print("control value during triple: ",control_value)
            raise ControlError('unknown control format during creation of triple')

        '''creation of starting point'''
        if starting_point == None:
            print("no starting point")
        elif isinstance(starting_point, int) or isinstance(starting_point, float):
            print("starting point is int or float")
            starting_point = torch.tensor([[starting_point]], dtype = torch.float)
        else:
            pass
            #print("starting point is: ", type(starting_point))

        #print("+++++control_value: ", control_value)
        trajectory = self.pde.euler_step(stepsize = self.stepsize, total_steps = self.support_points, last_point = starting_point, control= control_value)
        #print("trajectory: ", trajectory)

        #price = self.kosten.approx_costs(trajectory,control_value) 

        #print("first step: ",trajectory[0], ", last_step: ",trajectory[-1], ", costs: ",price)
        return [trajectory, control_value]

    def create_dataset(self, control_value = None, starting_point = None):
        self.dataset = []
        self.dataset.append(self.create_triple(starting_point = starting_point, control_value = control_value))
        for i in range(self.amount_x-1):
            self.dataset.append(self.create_triple(starting_point= self.dataset[-1][0][-1], control_value = control_value))
            '''if self.dataset[0][-1] > self.dataset[0][0]:
                print("Growing state!")
                break
            if self.dataset[0][-1] < - self.dataset[0][0]:
                print("state runs negative")
                break
            '''
        print(self.dataset)
        return self.dataset

    def create_dataset_different_controls(self):
        controls = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
        #assert len(controls) == self.amount_controls
        for con in controls:
            self.datasets.append(self.create_dataset(control = con))
        return self.datasets

    def create_dataset_different_control_and_starts(self):
        controls = [0, -2, -4, -6, -8, -10, -12, -16]
        starting_points = [0,1,0.3, 0.6]
        #assert len(controls) == self.amount_controls
        for i in itertools.product(controls, starting_points):
            print(i)
            self.datasets.append(self.create_dataset(control_value = i[0], starting_point = i[1]))
        return self.datasets



class dgl:
    def __init__(self, x_0 = None):
        self.A = torch.tensor([[1]], dtype= torch.float)
        self.B = torch.tensor([[0.2]], dtype= torch.float)
        self.x_0 = torch.tensor([[1]], dtype= torch.float) if not x_0 else x_0

    def rhs(self, x, u):
        return torch.matmul(self.A,x) +torch.matmul(self.B,u)

    def euler_step(self, stepsize = 0.1,total_steps = 1, last_point=None, control = None):
        control = torch.tensor([[0]], dtype=torch.float) if not control else control
        print("control value", control)
        
        assert len(control) ==1 #for testing purpose

        if not last_point: last_point = self.x_0
        output = [last_point]
        for i in range(total_steps):
            last_point = last_point + stepsize * self.rhs(last_point, torch.matmul(control, last_point))
            output.append(last_point)
        #print("last point of trajectory: ",last_point)
        return output
    def linear_feedback(self):
        pass


class cost_functional:
    def __init__(self):
        self.Q = torch.tensor([[1]],dtype=torch.float)
        self.R = torch.tensor([[0.5]], dtype = torch.float)

    def approx_costs(self, x_values, l_control_values, r_control_values):
        '''returns the integral over xQx + uRU'''
        assert len(x_values) == len(l_control_values)
        assert len(l_control_values) == len(r_control_values)

        #points = [torch.matmul(x, torch.matmul(self.Q,x))+ torch.matmul(l_u, torch.matmul(self.R,r_u)) for (x,l_u, r_u) in zip(x_values, l_control_values, r_control_values)]
        
        points = [torch.matmul(x, torch.matmul(self.Q,x))+ torch.matmul(l_u, torch.matmul(self.R,r_u)) for (x,l_u, r_u) in zip(x_values, l_control_values, r_control_values)]

        #integral = np.trapz(points,dx = 0.1)

        '''points = torch.Tensor(points)
        integral = torch.trapz(points, dx = 0.1, dim = -1)#TODO sure about this?'''

        integral = 0
        for support_point in points:
            integral += support_point
        integral = integral/len(points)
        return integral


if __name__ == '__main__':
    dataset = Dataset()
    #dataset.create_dataset()
    output = dataset.create_dataset_different_controls()
    print(output)



