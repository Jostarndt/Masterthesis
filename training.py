import numpy as np
import dgl
import model
import torch
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy


class error():
    def __init__(self):
        #self.old_control = model.actor()
        #self.new_control = model.actor()
        #self.value_function = model.critic()
        self.costs = dgl.cost_functional()

    def value_error_external(self,point, old_control, new_control, value_function):
        old_controls = [old_control(x) for x in point[0]]
        new_controls = [new_control(x) for x in point[0]]
        
        #kosten returns integral of  xQx + uRu
        kosten = self.costs.approx_costs(x_values = point[0],l_control_values=old_controls, r_control_values =  old_controls, x_size = 0.3)
        
        zeros = [torch.tensor([[0]], dtype = torch.float) for i in point[0]]
        difference = [old_control(x)-torch.matmul(point[1],x) for x in point[0]]
        kosten_two = self.costs.approx_costs(x_values = zeros,l_control_values = new_controls, r_control_values =difference, x_size = 0.3)
        '''
        with torch.no_grad():
            next_value =value_function(point[0][0])
        '''
        return 2*kosten_two - kosten + value_function(point[0][0]) - value_function(point[0][-1])
    '''
    def value_error(self,point):
        old_controls = [self.old_control(x) for x in point[0]]
        new_controls = [self.new_control(x) for x in point[0]]
        
        #kosten returns integral of  xQx + uRu
        kosten = self.costs.approx_costs(x_values = point[0],l_control_values=old_controls, r_control_values =  old_controls, x_size = 0.3)
        
        zeros = [torch.tensor([[0]], dtype = torch.float) for i in point[0]]
        difference = [self.old_control(x)-torch.matmul(point[1], x) for x in point[0]]
        kosten_two = self.costs.approx_costs(x_values = zeros,l_control_values = new_controls, r_control_values =difference, x_size = 0.3)
        return 2*kosten_two - kosten + self.value_function(point[0][0]) - self.value_function(point[0][-1])

    '''
    '''
    def train(self, point):
        for param in self.old_control.parameters():
            param.require_grad = False
        parameters = list(self.new_control.parameters()) + list(self.value_function.parameters())
        optimizer = optim.Adam(parameters, lr=0.005)
        optimizer.zero_grad()
        loss = self.value_error(point)
        #loss = torch.abs(self.value_error(point))
        #print("loss: ",loss)
        loss.backward()
        optimizer.step()
    '''


if __name__ == '__main__':
    error = error()
    dataset = dgl.Dataset()

    #output = dataset.create_dataset_different_controls()
    output = dataset.create_dataset_different_control_and_starts()
    #print(output)
    '''
    print("the pretrained optimal control for x_0=1 is: ", error.new_control(torch.tensor([[1]], dtype=torch.float)))
    print("the pretrained value for x_0 = 1 is: ", error.value_function(torch.tensor([[1]], dtype=torch.float)))
    print("-----------------------------")

    for epoch in range(10):
        for trajectory in dataset.datasets:
            for point in trajectory:
                #print("data point: ", point)
                #print("error: ", error.value_error(point))
                error.train(point)
    error.old_control = deepcopy(error.new_control)

    print("the optimal control for x_0 = 1 is: ", error.new_control(torch.tensor([[1]], dtype=torch.float)))
    print("the pretrained value for x_0 = 1 is: ", error.value_function(torch.tensor([[1]], dtype=torch.float)))

    '''

    print("##################################")

    old_control = model.actor(stabilizing = True)#TODO: make this admissible!
    for param in old_control.parameters():
        print(param)
    new_control = model.actor()
    value_function = model.critic()
    costs = dgl.cost_functional()

    control_optimizer = optim.SGD(new_control.parameters(), lr=0.005)
    value_optimizer = optim.SGD(value_function.parameters(), lr= 0.005)

    print("the pretrained optimal control for x_0=1 is: ", new_control(torch.tensor([[1]], dtype=torch.float)))
    print("the pretrained value for x_0 = 1 is: ", value_function(torch.tensor([[1]], dtype=torch.float)))
    for epoch in range(10):
        print("epoch")
        '''
        for trajectory in dataset.datasets:
            #batch_error = 0
            control_optimizer.zero_grad()
            value_optimizer.zero_grad()
            for point in trajectory:
                #batch_error = nn.MSELoss(error.value_error_external(point,old_control, new_control, value_function))
                batch_error = (error.value_error_external(point,old_control, new_control, value_function))**2
                batch_error.backward()
                control_optimizer.step()
                value_optimizer.step()
                #batch_error = 0 
        '''
        for step in range(len(dataset.datasets[1])):
            for trajectory in dataset.datasets:
                batch_error = (error.value_error_external(trajectory[step], old_control, new_control, value_function))**2
                batch_error.backward()
                control_optimizer.step()#this is a 30sample batch update
                control_optimizer.zero_grad()

                value_optimizer.step() # this is a 1sample batch update
                value_optimizer.zero_grad()

                old_control = deepcopy(new_control)
        

        old_control = deepcopy(new_control)

    print("the optimal control for x_0=1 is: ", new_control(torch.tensor([[1]], dtype=torch.float)))
    print("the optimal control for x_0=2 is: ", new_control(torch.tensor([[2]], dtype=torch.float)))
    print("the optimal control for x_0=0.5 is: ", new_control(torch.tensor([[0.5]], dtype=torch.float)))

 
    print("the optimal value for x_0 = 1 is: ", value_function(torch.tensor([[1]], dtype=torch.float)))
    print("the optimal value for x_0 = 2 is: ", value_function(torch.tensor([[2]], dtype=torch.float)))
    print("the optimal value for x_0 = 0.5 is: ", value_function(torch.tensor([[0.5]], dtype=torch.float)))
