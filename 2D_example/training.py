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
        self.Q = torch.tensor([[1, 0], [0, 1]],dtype=torch.float)
        self.R = torch.tensor([[1]], dtype = torch.float)

    def value_iteration(self,trajectory, control, old_control, new_control, value_function):
        traj = torch.squeeze(trajectory, 0)

        old_controls = old_control(traj).detach()
        new_controls= new_control(traj).detach()
        difference = old_controls - control
        #points_a = torch.matmul(trajectory, torch.matmul(self.Q,trajectory))+ torch.matmul(old_controls, torch.matmul(self.R, old_controls))
        oc = torch.squeeze(old_controls, 0)
        points_a = torch.matmul(traj, torch.matmul(self.Q, traj.transpose(1,2))) + torch.matmul(oc, torch.matmul(self.R, oc.transpose(1,2)))

        diff = torch.squeeze(difference, 0)
        points_b = torch.matmul(new_controls, torch.matmul(self.R, diff))
        points_together = 2*points_b - points_a

        control_loss =0.2* torch.mean(points_together)
        #pdb.set_trace()
        #overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[0][0]))) - torch.squeeze(torch.squeeze(value_function(trajectory[0][-1]).detach())) + control_loss 
 



        #WITH OPTIMAL CONTROL
 
        #optimal_vector = 0.04*torch.matmul(trajectory,torch.matmul(self.R,trajectory)) + 0.4*torch.matmul(trajectory, torch.matmul(self.R,control)) - torch.matmul(trajectory, torch.matmul(self.Q, trajectory))
        #optimal_vector = 0.1*torch.mean(optimal_vector)
        
        #overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[0][0]))) - torch.squeeze(torch.squeeze(value_function(trajectory[0][-1]))).detach() +optimal_vector
        #overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[0][0]))) - 0.5*torch.square(trajectory[0][-1]) +optimal_vector

        #training on optimal solution
        #pdb.set_trace()
        overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[0][0]))) - 0.5*torch.square(trajectory[0][0][0][0])- torch.square(trajectory[0][0][0][1])



        overall_loss = torch.square(overall_loss)# + torch.square(value_function(torch.tensor([[0,0]], dtype = torch.float)))

        return overall_loss#TODO add here the value of value_function(0)

    def policy_improvement(self,trajectory, control, old_control, new_control, value_function):
        traj = torch.squeeze(trajectory, 0)
        
        old_controls = old_control(traj).detach()
        new_controls= new_control(traj)
        difference = old_controls - control

        oc= torch.squeeze(old_controls, 0)
        points_a = torch.matmul(traj, torch.matmul(self.Q, traj.transpose(1,2))) + torch.matmul(oc, torch.matmul(self.R, oc.transpose(1,2)))

        diff = torch.squeeze(difference, 0)
        points_b = torch.matmul(new_controls, torch.matmul(self.R, diff))
        points_together = 2*points_b - points_a
        control_loss =0.2* torch.mean(points_together)
        #overall_loss =  torch.squeeze(torch.squeeze(value_function(trajectory[0][0]).detach())) - torch.squeeze(torch.squeeze(value_function(trajectory[0][-1]).detach())) + control_loss

        '''
        compare_diff = torch.squeeze(-torch.unsqueeze(traj[:,:,1]*traj[:,:,0], 2)- torch.squeeze(control, 1), 0)
        compare_b = torch.matmul(new_controls, torch.matmul(self.R, compare_diff))
        compare_a = torch.matmul(traj, torch.matmul(self.Q, traj.transpose(1,2))) + torch.matmul(-torch.unsqueeze(traj[:,:,1]*traj[:,:,0], 2), torch.matmul(self.R, -torch.unsqueeze(traj[:,:,1]*traj[:,:,0], 2).transpose(1,2)))
        compare_loss = 0.1*torch.mean(2*compare_b -compare_a)
        pdb.set_trace()
        compare = 0.5* traj[0][0][0]**2 + traj[0][0][1]**2 - 0.5* traj[-1][0][0]**2 - traj[-1][0][1]**2 + compare_loss
        '''
        #pdb.set_trace()
        overall_loss = 0.5* traj[0][0][0]**2 + traj[0][0][1]**2 - 0.5* traj[-1][0][0]**2 - traj[-1][0][1]**2  + control_loss

        
        #overall_loss = 0.1*torch.mean(torch.square(new_control(traj)+torch.unsqueeze(traj[:,:,0]*traj[:,:,1],1)))
        #overall_loss = torch.mean(torch.square(new_control(traj)+torch.unsqueeze(traj[:,:,1]*traj[:,:,0],1)))
        #overall_loss =torch.mean(torch.square(new_control(traj)+torch.ones(11).unsqueeze(1).unsqueeze(1))) 
        

        #overall_loss = torch.square(overall_loss)
        overall_loss = torch.abs(overall_loss)
        return overall_loss

if __name__ == '__main__':
    Writer = SummaryWriter()
    error = error()
    dataset = dgl.Dataset()
    output = dataset.create_dataset_different_control_and_starts()
    train_loader = DataLoader(dataset = output, batch_size = 1, shuffle =True )

    ''' 
    ##########
    #plotting of the trajectories
    ##########
    for i,piece in enumerate(output):
        for step in range(piece[0].size()[0]):
            Writer.add_scalars('trajectories', {'first coord'+str(i): piece[0][step][0][0], 'second coord'+str(i): piece[0][step][0][1]}, step)
        #Writer.add_histogram('trajectories', 
    '''

    print("##################################")
    old_control = model.actor(stabilizing = False, control_dim = 1, space_dim = 2)
    new_control = model.actor(stabilizing = False, control_dim = 1, space_dim = 2)
    value_function = model.critic(positive = True, space_dim = 2)
    costs = dgl.cost_functional()

    control_optimizer = optim.SGD(new_control.parameters(), lr=0.5)#0.5
    value_optimizer = optim.SGD(value_function.parameters(), lr=0.2)#ideally 0.3, 0.2 is better
    #control_optimizer = optim.Adam(new_control.parameters(), lr=0.05)
    #value_optimizer = optim.Adam(value_function.parameters(), lr=0.02)

    #lmbda = lambda epoch :0.8# 0.996
    #scheduler = optim.lr_scheduler.MultiplicativeLR(value_optimizer, lr_lambda = lmbda)
 
    #Training
    j = 0
    for epoch in range(10):
        print("epoch: ", epoch)
        for x, u in train_loader:
            control_optimizer.zero_grad()
            value_optimizer.zero_grad()

            value_error= error.value_iteration(x, u, old_control, new_control, value_function)
            assert value_error !=  0
            value_error.backward()
            value_optimizer.step()

            control_optimizer.zero_grad()
            value_optimizer.zero_grad()

            policy_error = error.policy_improvement(x, u, old_control, new_control, value_function)
            #assert policy_error !=  0
            policy_error.backward()
            control_optimizer.step()

            if j%100 == 0:
                print(policy_error)
                Writer.add_scalars('errors', {'policy error': policy_error,'value_error':value_error}, j)
                old_control = deepcopy(new_control)
            j +=1
        #scheduler.step()
    '''
    index = np.linspace(0,1,100)
    mesh = np.transpose(np.meshgrid(index, index))
    mesh = mesh.reshape(10000,2)
    z = np.array([[-x_1*x_2] for (x_1, x_2) in mesh])

    pdb.set_trace()
    opt_control = np.hstack((mesh, z))
    Writer.add_embedding(opt_control, tag= 'optimal control')
    Writer.close()
    '''
    int_const = value_function(torch.tensor([[0,0]], dtype= torch.float)).detach()
    op_val_img = np.zeros((3,100,100))
    op_con_img = np.zeros((3,100,100))
    con_img = np.zeros((3,100,100))
    val_img = np.zeros((3,100,100))
    for x_1 in range(100):
        for x_2 in range(100):
            #print("tuple: ", x_1/1000, x_2/1000)
            op_con_img[0][x_1][x_2] = -x_1/1000 * x_2/1000 
            op_val_img[0][x_1][x_2] = 0.5*(x_1/1000) **2 + (x_2/1000 )**2
            val_img[0][x_1][x_2] = value_function(torch.tensor([[x_1/1000 , x_2/1000 ]], dtype = torch.float)).detach() - int_const
            con_img[0][x_1][x_2] = old_control(torch.tensor([[x_1/1000 , x_2/1000]], dtype = torch.float)).detach()
    #rescaling the images:

    val_max = max(np.amax(val_img), np.amax(op_val_img))
    val_min = min(np.amin(val_img), np.amin(op_val_img))
    con_max = max(np.amax(con_img), np.amax(op_con_img))
    con_min = min(np.amin(con_img), np.amin(op_con_img))
    
    print(np.amin(op_con_img))
    print('con max, con min', con_max, con_min)

    print("control at zero:",new_control(torch.tensor([[0,0]], dtype= torch.float)))
    print("control at 1,1:",new_control(torch.tensor([[0.1,0.1]], dtype= torch.float)))

    

    print(con_img[0], 'con img')
    print(op_con_img[0], 'op con img')

    val_img = (val_img-val_min)/(val_max - val_min)
    con_img = (con_img-con_min)/(con_max - con_min)
    op_val_img = (op_val_img-val_min)/(val_max-val_min)
    op_con_img = (op_con_img-con_min)/(con_max-con_min)
    
    Writer.add_image('optimal value', op_val_img , 0)
    Writer.add_image('optimal control function', op_con_img , 0)
    Writer.add_image('value_function', val_img , 0)
    Writer.add_image('control_function', con_img , 0)
    Writer.close()
    print(con_img[0], 'con img')
    print(op_con_img[0], 'op con img')
    print('done')
    
