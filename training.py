import numpy as np
import dgl
import model
import torch
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter


class error():
    def __init__(self):
        #self.new_control = model.actor()
        #self.value_function = model.critic()
        self.costs = dgl.cost_functional()

    def value_error_external(self,point, old_control, new_control, value_function):
        '''
        point consists of [[[x values], control], [[x values], control]]]

        '''
        old_controls = [old_control(x) for x in point[0]]
        new_controls = [new_control(x) for x in point[0]]
        
        #kosten returns integral of  xQx + uRu
        kosten = self.costs.approx_costs(x_values = point[0],l_control_values=old_controls, r_control_values =  old_controls, x_size = 0.1)
        
        zeros = [torch.tensor([[0]], dtype = torch.float) for i in point[0]]
        difference = [old_control(x)-torch.matmul(point[1],x) for x in point[0]]
        kosten_two = self.costs.approx_costs(x_values = zeros,l_control_values = new_controls, r_control_values =difference, x_size = 0.1)
        
#        with torch.no_grad():
#            next_value =value_function(point[0][-1])
        
        return 2*kosten_two - kosten + value_function(point[0][0]) - value_function(point[0][-1])






if __name__ == '__main__':
    Writer = SummaryWriter()
    error = error()
    dataset = dgl.Dataset()

    #output = dataset.create_dataset_different_controls()
    output = dataset.create_dataset_different_control_and_starts()
   

    ''' plotting of the trajectories
    for i, trajectory in enumerate(dataset.datasets):
        for j, section in enumerate(trajectory):
            for k, point in enumerate(section[0][:-1]):
                Writer.add_scalar('trajectory/'+str(i)+' control: '+str(section[1]), point, k+(len(section[0])-1)*j)
    '''

    print("##################################")

    old_control = model.actor(stabilizing = True)#TODO: make this admissible!
    new_control = model.actor()
    value_function = model.critic(positive = True)
    costs = dgl.cost_functional()

    control_optimizer = optim.SGD(new_control.parameters(), lr=0.5)
    value_optimizer = optim.SGD(value_function.parameters(), lr=0.05)

    value_function.load_state_dict(torch.load('./models/pretrained_value'))
    old_control.load_state_dict(torch.load('./models/pretrained_control'))
    new_control.load_state_dict(torch.load('./models/pretrained_control'))


    for i in range(100):
        Writer.add_scalar('cost_function/pretraining', value_function(torch.tensor([[i/100]], dtype = torch.float)), i)
        Writer.add_scalar('control/pretraining', new_control(torch.tensor([[i/100]], dtype = torch.float)),i)


    #Training
    for epoch in range(30):
        print("epoch")
        epoch_error = 0
        #control_error = 0 
        #control_optimizer.zero_grad()
        #value_optimizer.zero_grad()
        for step in range(len(dataset.datasets[1])):
            for i,trajectory in enumerate(dataset.datasets):
                if i %20 == 0:
                    control_error = 0 
                    control_optimizer.zero_grad()
                    value_optimizer.zero_grad()
                control_error += (error.value_error_external(trajectory[step], old_control, new_control, value_function))**2
                if i%20 == 19:
                    #print(i)
                    control_error.backward()
                    value_optimizer.step() 
                    control_optimizer.step()
                    old_control = deepcopy(new_control) #TODO wohin?
                    epoch_error += control_error
        print(epoch_error)
        #print(control_error/value_function(dataset.dataset[0][0][0]))



    for i in range(100):
        #Writer.add_scalar('cost_function/approximation', value_function(torch.tensor([[i/100]], dtype = torch.float)), i)
        Writer.add_scalar('control/approximation on control', new_control(torch.tensor([[i/100]], dtype = torch.float)),i)
        Writer.add_scalar('control/optimal', -0.2*i/100, i)
        Writer.add_scalar('control/suboptimal', 10.2 * i/100, i)
        Writer.add_scalar('cost_function/optimal', 0.5 * i/100 * i/100, i)
        Writer.add_scalar('cost_function/suboptimal', -25.5 * i/100 * i/100, i)
        
    for i in range(1000):
        Writer.add_scalar('cost_function/approximation', value_function(torch.tensor([[i/1000]], dtype = torch.float)), i)
