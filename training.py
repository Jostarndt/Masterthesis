import numpy as np
import dgl
import model
import torch
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pdb

class error():
    def __init__(self):
        self.costs = dgl.cost_functional()
        self.Q = torch.tensor([[1]],dtype=torch.float)
        self.R = torch.tensor([[0.5]], dtype = torch.float)

    def value_error_external(self,trajectory, control, old_control, new_control, value_function):
        old_controls = old_control(trajectory)
        new_controls= new_control(trajectory)
        difference = old_controls - control

        points_a = torch.matmul(trajectory, torch.matmul(self.Q,trajectory))+ torch.matmul(old_controls, torch.matmul(self.R, old_controls))
        
        points_b = torch.matmul(new_controls, torch.matmul(self.R, difference))

        points_together = 2*points_b - points_a
        control_loss =0.1* torch.mean(points_together)#WHY 0.1???
        #overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[0][0]))) - torch.squeeze(torch.squeeze(value_function(trajectory[0][-1]).detach())) + control_loss

        #WITH OPTIMAL VALUE FUNCTION:
        #overall_loss =  torch.squeeze(torch.squeeze(0.5*torch.square(trajectory[0][0]))) - torch.squeeze(torch.squeeze(0.5*torch.square(trajectory[0][-1]))) + control_loss


        #WITH OPTIMAL CONTROL
        
        optimal_vector = 0.04*torch.matmul(trajectory,torch.matmul(self.R,trajectory)) + 0.4*torch.matmul(trajectory, torch.matmul(self.R,control)) - torch.matmul(trajectory, torch.matmul(self.Q, trajectory))
        optimal_vector = 0.1*torch.mean(optimal_vector)
        #optimal_vector_two =-(  0.5*torch.square(trajectory[0][0]) - 0.5*torch.square(trajectory[0][-1]))

        overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[0][0]))) - torch.squeeze(torch.squeeze(value_function(trajectory[0][-1]))).detach() +optimal_vector
        #overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[0][0]))) - 0.5*torch.square(trajectory[0][-1]) +optimal_vector
        #overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[0][0]))) - 0.5*torch.square(trajectory[0][0])
        

        overall_loss = torch.square(overall_loss) + torch.square(value_function(torch.tensor([[0]], dtype = torch.float)))
        return overall_loss

if __name__ == '__main__':
    Writer = SummaryWriter()
    error = error()
    dataset = dgl.Dataset()
    output = dataset.create_dataset_different_control_and_starts()
    train_loader = DataLoader(dataset = output, batch_size = 1, shuffle = True)

    ''' 
    ##########
    #plotting of the trajectories
    ##########
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
    value_optimizer = optim.SGD(value_function.parameters(), lr=0.3)

    #value_function.load_state_dict(torch.load('./models/pretrained_value'))
    #old_control.load_state_dict(torch.load('./models/pretrained_control'))
    #new_control.load_state_dict(torch.load('./models/pretrained_control'))


    for i in range(100):
        Writer.add_scalar('cost_function/pretraining', value_function(torch.tensor([[i/100]], dtype = torch.float)), i)
        Writer.add_scalar('control/pretraining', new_control(torch.tensor([[i/100]], dtype = torch.float)),i)


    #Training
    j = 0
    for epoch in range(10):
        print("epoch")
        for x, u in train_loader:
            #print(x, u)
            control_optimizer.zero_grad()
            value_optimizer.zero_grad()

            control_error = error.value_error_external(x, u, old_control, new_control, value_function)
            control_error.backward()
            value_optimizer.step()
            #control_optimizer.step()
            if j%100 == 0:
                Writer.add_scalar('error', control_error, j)
                old_control = deepcopy(new_control)
            j +=1

    for i in range(100):
        Writer.add_scalar('control/after training', new_control(torch.tensor([[i/100]], dtype = torch.float)),i)
        Writer.add_scalar('control/optimal', -0.2*i/100, i)
        Writer.add_scalar('control/suboptimal', 10.2 * i/100, i)
        Writer.add_scalar('cost_function/optimal', 0.5 * i/100 * i/100, i)
        Writer.add_scalar('cost_function/suboptimal', -25.5 * i/100 * i/100, i)
    #pdb.set_trace()
    for i in range(1000):
        Writer.add_scalar('cost_function/approximation', value_function(torch.tensor([[i/1000]], dtype = torch.float)), i)
